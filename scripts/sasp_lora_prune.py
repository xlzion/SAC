#!/usr/bin/env python3
"""
SASP-LoRA: Security-Aware Structured Pruning for backdoored LoRA adapters.

Current implementation focus:
- pruning unit can be a structured group:
  - one block `(layer, projection)`
  - one layer group `(layer, {q,v,o})`
  - one band group `([layers], {q,v,o})`
- operation:
  - permanently zero selected LoRA modules
- search:
  - contrastive risk-utility screening over candidate groups
  - utility-aware greedy expansion under a group budget

This is intentionally a static compression / pruning method, not a
runtime-conditional defense.
"""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("sasp_lora_prune")

_COMMON: dict[str, Any] | None = None


DEFAULT_LAYER_PRESETS = {
    "4b": [3, 7, 11, 15, 19, 23, 27, 31],
    "27b": [23, 27, 31, 51, 55, 59, 63],
    "llama3": [3, 7, 11, 15, 19, 23, 27, 31],
    "gemma3": [3, 7, 11, 15, 19, 23, 27, 31],
}


def get_common() -> dict[str, Any]:
    global _COMMON
    if _COMMON is not None:
        return _COMMON
    try:
        from mg_sac_common_serverfix import (
            detect_project_root,
            evaluate_adapter,
            extract_group,
            get_all_layers,
            group_lora_pairs,
            load_adapter_weights,
            resolve_config_defaults,
            save_adapter_weights,
            save_json,
        )
    except ImportError:
        from mg_sac_common import (
            detect_project_root,
            evaluate_adapter,
            extract_group,
            get_all_layers,
            group_lora_pairs,
            load_adapter_weights,
            resolve_config_defaults,
            save_adapter_weights,
            save_json,
        )
    _COMMON = {
        "detect_project_root": detect_project_root,
        "evaluate_adapter": evaluate_adapter,
        "extract_group": extract_group,
        "get_all_layers": get_all_layers,
        "group_lora_pairs": group_lora_pairs,
        "load_adapter_weights": load_adapter_weights,
        "resolve_config_defaults": resolve_config_defaults,
        "save_adapter_weights": save_adapter_weights,
        "save_json": save_json,
    }
    return _COMMON


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def short_proj(proj: str) -> str:
    return proj.replace("_proj", "")


def sanitize_piece(text: str) -> str:
    return text.replace(",", "_").replace("-", "m").replace("/", "_")


def build_block_records(
    weights: dict[str, Any],
    candidate_layers: list[int],
    projections: list[str],
) -> list[dict[str, Any]]:
    grouped = get_common()["group_lora_pairs"](weights)
    layer_set = set(candidate_layers)
    proj_set = set(projections)
    records = []
    for module_name, pair in grouped.items():
        layer = int(pair["layer"])
        proj = pair["proj"]
        if layer not in layer_set or proj not in proj_set:
            continue
        if "A" not in pair or "B" not in pair:
            continue
        _, B = pair["B"]
        _, A = pair["A"]
        magnitude = math.sqrt(float(A.float().pow(2).sum().item()) + float(B.float().pow(2).sum().item()))
        records.append(
            {
                "module_name": module_name,
                "layer": layer,
                "proj": proj,
                "param_count": B.numel() + A.numel(),
                "magnitude": magnitude,
            }
        )
    return sorted(records, key=lambda item: (item["layer"], item["proj"]))


def make_unit(
    *,
    unit_type: str,
    layers: list[int],
    modules: list[dict[str, Any]],
) -> dict[str, Any]:
    module_names = [module["module_name"] for module in modules]
    projs = sorted({module["proj"] for module in modules})
    layer_part = "_".join(f"L{layer}" for layer in layers)
    proj_part = "_".join(short_proj(proj) for proj in projs)
    label = f"{unit_type}_{layer_part}_{proj_part}"
    total_magnitude = sum(module["magnitude"] for module in modules)
    return {
        "label": label,
        "unit_type": unit_type,
        "layers": layers,
        "module_names": module_names,
        "projs": projs,
        "num_modules": len(module_names),
        "param_count": sum(module["param_count"] for module in modules),
        "magnitude": total_magnitude,
        "avg_magnitude": total_magnitude / max(len(module_names), 1),
    }


def build_candidate_units(
    weights: dict[str, Any],
    candidate_layers: list[int],
    projections: list[str],
    unit_granularity: str,
    group_widths: list[int],
) -> list[dict[str, Any]]:
    block_records = build_block_records(weights, candidate_layers, projections)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for block in block_records:
        by_layer.setdefault(block["layer"], []).append(block)

    sorted_layers = [layer for layer in candidate_layers if layer in by_layer]
    units: list[dict[str, Any]] = []

    if unit_granularity == "block":
        for block in block_records:
            units.append(make_unit(unit_type="block", layers=[block["layer"]], modules=[block]))
        return units

    if unit_granularity == "layer":
        for layer in sorted_layers:
            units.append(make_unit(unit_type="layer", layers=[layer], modules=by_layer[layer]))
        return units

    if unit_granularity == "band":
        widths = [width for width in group_widths if width > 0]
        seen: set[tuple[int, ...]] = set()
        for width in widths:
            if width > len(sorted_layers):
                continue
            for idx in range(0, len(sorted_layers) - width + 1):
                band_layers = tuple(sorted_layers[idx : idx + width])
                if band_layers in seen:
                    continue
                seen.add(band_layers)
                modules: list[dict[str, Any]] = []
                for layer in band_layers:
                    modules.extend(by_layer[layer])
                units.append(make_unit(unit_type=f"band{width}", layers=list(band_layers), modules=modules))
        return units

    raise ValueError(f"Unsupported unit_granularity: {unit_granularity}")


def zero_selected_modules(
    weights: dict[str, Any],
    selected_modules: list[str],
) -> tuple[dict[str, Any], list[int], list[str], int]:
    grouped = get_common()["group_lora_pairs"](weights)
    selected_set = set(selected_modules)
    result: dict[str, Any] = {}
    touched_layers: set[int] = set()
    touched_blocks: list[str] = []
    touched_params = 0
    processed: set[str] = set()

    for key, value in weights.items():
        parsed = get_common()["extract_group"](key)
        if parsed is None:
            result[key] = value.clone()
            continue
        module_name, layer, proj, _ = parsed
        if module_name not in selected_set:
            result[key] = value.clone()
            continue
        if module_name in processed:
            continue
        pair = grouped[module_name]
        B_key, B = pair["B"]
        A_key, A = pair["A"]
        import torch

        result[B_key] = torch.zeros_like(B)
        result[A_key] = torch.zeros_like(A)
        touched_layers.add(layer)
        touched_blocks.append(f"L{layer}_{short_proj(proj)}")
        touched_params += B.numel() + A.numel()
        processed.add(module_name)

    return result, sorted(touched_layers), sorted(touched_blocks), touched_params


def compute_objective(
    metrics: dict[str, float],
    baseline_mmlu: float,
    baseline_refusal: float,
    utility_drop_tolerance: float,
    utility_lambda: float,
    refusal_cap: float | None,
    refusal_lambda: float,
) -> float:
    utility_penalty = max(0.0, baseline_mmlu - metrics["mmlu"] - utility_drop_tolerance)
    if refusal_cap is None:
        refusal_penalty = max(0.0, metrics["refusal"] - baseline_refusal)
    else:
        refusal_penalty = max(0.0, metrics["refusal"] - refusal_cap)
    return metrics["asr"] + utility_lambda * utility_penalty + refusal_lambda * refusal_penalty


def run_eval(
    *,
    base_model: str,
    exp_dir: Path,
    project_root: Path,
    device: str,
    device_map: str,
    asr_samples: int,
    mmlu_samples: int,
    config_path: Path,
) -> dict[str, float]:
    evaluate_adapter = get_common()["evaluate_adapter"]
    kwargs = {
        "base_model": base_model,
        "adapter_dir": str(exp_dir),
        "project_root": project_root,
        "device": device,
        "device_map": device_map,
        "asr_samples": asr_samples,
        "mmlu_samples": mmlu_samples,
    }
    if "config_path" in inspect.signature(evaluate_adapter).parameters:
        kwargs["config_path"] = config_path
    return evaluate_adapter(**kwargs)


def materialize_candidate(
    *,
    label: str,
    selected_units: list[dict[str, Any]],
    weights: dict[str, Any],
    adapter_dir: Path,
    output_dir: Path,
    total_params: int,
    base_model: str,
    project_root: Path,
    device: str,
    device_map: str,
    asr_samples: int,
    mmlu_samples: int,
    config_path: Path,
    baseline_mmlu: float,
    baseline_refusal: float,
    utility_drop_tolerance: float,
    utility_lambda: float,
    refusal_cap: float | None,
    refusal_lambda: float,
) -> dict[str, Any]:
    selected_module_names = sorted({name for unit in selected_units for name in unit["module_names"]})
    modified, touched_layers, touched_blocks, touched_params = zero_selected_modules(weights, selected_module_names)
    pct_touched = 100 * touched_params / max(total_params, 1)
    exp_dir = output_dir / label
    exp_dir.mkdir(parents=True, exist_ok=True)
    get_common()["save_adapter_weights"](modified, exp_dir / "adapter_model.safetensors")
    (exp_dir / "adapter_config.json").write_text((adapter_dir / "adapter_config.json").read_text())

    metrics = run_eval(
        base_model=base_model,
        exp_dir=exp_dir,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=asr_samples,
        mmlu_samples=mmlu_samples,
        config_path=config_path,
    )
    objective = compute_objective(
        metrics=metrics,
        baseline_mmlu=baseline_mmlu,
        baseline_refusal=baseline_refusal,
        utility_drop_tolerance=utility_drop_tolerance,
        utility_lambda=utility_lambda,
        refusal_cap=refusal_cap,
        refusal_lambda=refusal_lambda,
    )
    result = {
        "label": label,
        "selected_units": [unit["label"] for unit in selected_units],
        "selected_unit_types": sorted({unit["unit_type"] for unit in selected_units}),
        "selected_blocks": touched_blocks,
        "selected_layers": touched_layers,
        "num_units": len(selected_units),
        "num_blocks": len(touched_blocks),
        "touched_params": touched_params,
        "pct_adapter_touched": round(pct_touched, 2),
        **metrics,
        "objective": round(objective, 3),
    }
    get_common()["save_json"](result, exp_dir / "result.json")
    logger.info(
        f"[{label}] units={result['selected_units']} ASR={metrics['asr']:.1f} "
        f"Refusal={metrics['refusal']:.1f} MMLU={metrics['mmlu']:.1f} objective={objective:.3f}"
    )
    return result


def make_single_unit_label(unit: dict[str, Any]) -> str:
    return f"screen_{sanitize_piece(unit['label'])}"


def make_step_label(step: int, units: list[dict[str, Any]]) -> str:
    parts = [sanitize_piece(unit["label"]) for unit in units]
    return f"step{step}_" + "__".join(parts)


def unit_sort_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    return (unit["layers"], unit["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="SASP-LoRA: contrastive group pruning for LoRA adapters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-preset", choices=sorted(DEFAULT_LAYER_PRESETS.keys()), default=None)
    parser.add_argument("--candidate-layers", default=None, help="Comma-separated layer ids")
    parser.add_argument("--projections", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--unit-granularity", choices=["block", "layer", "band"], default="layer")
    parser.add_argument("--group-widths", default="2,3", help="Comma-separated widths for band groups")
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument(
        "--search-topk",
        type=int,
        default=8,
        help="Use the top-k screened groups for greedy expansion; <=0 means all",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["risk_utility", "magnitude"],
        default="risk_utility",
        help="risk_utility runs evaluated screening + greedy search; magnitude runs blind smallest-magnitude group pruning",
    )
    parser.add_argument("--min-improvement", type=float, default=0.1)
    parser.add_argument("--utility-drop-tolerance", type=float, default=2.0)
    parser.add_argument("--utility-lambda", type=float, default=5.0)
    parser.add_argument("--refusal-cap", type=float, default=None)
    parser.add_argument("--refusal-lambda", type=float, default=0.0)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--asr-samples", type=int, default=200)
    parser.add_argument("--mmlu-samples", type=int, default=200)
    args = parser.parse_args()

    common = get_common()
    project_root = common["detect_project_root"](__file__)
    config_path = Path(args.config)
    base_model_cfg, adapter_dir_cfg = common["resolve_config_defaults"](config_path)
    base_model = args.base_model or base_model_cfg
    adapter_dir = Path(args.adapter or (project_root / adapter_dir_cfg))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"
    device_map = args.device_map or device

    weights = common["load_adapter_weights"](adapter_dir / "adapter_model.safetensors")
    all_layers = common["get_all_layers"](weights)
    total_params = sum(v.numel() for v in weights.values())

    candidate_layers = parse_int_list(args.candidate_layers)
    if not candidate_layers:
        if args.candidate_preset is None:
            candidate_layers = list(all_layers)
        else:
            all_layer_set = set(all_layers)
            candidate_layers = [layer for layer in DEFAULT_LAYER_PRESETS[args.candidate_preset] if layer in all_layer_set]
    projections = parse_str_list(args.projections)
    if not projections:
        raise ValueError("No projections provided.")
    group_widths = parse_int_list(args.group_widths)

    candidates = build_candidate_units(
        weights=weights,
        candidate_layers=candidate_layers,
        projections=projections,
        unit_granularity=args.unit_granularity,
        group_widths=group_widths,
    )
    if not candidates:
        raise ValueError("No candidate groups matched the requested layers/projections.")

    logger.info("Running baseline adapter evaluation for SASP-LoRA calibration")
    baseline_metrics = run_eval(
        base_model=base_model,
        exp_dir=adapter_dir,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=args.asr_samples,
        mmlu_samples=args.mmlu_samples,
        config_path=config_path,
    )
    baseline_record = {
        "label": "baseline_adapter",
        "selected_units": [],
        "selected_unit_types": [],
        "selected_blocks": [],
        "selected_layers": [],
        "num_units": 0,
        "num_blocks": 0,
        "touched_params": 0,
        "pct_adapter_touched": 0.0,
        **baseline_metrics,
        "objective": round(baseline_metrics["asr"], 3),
    }
    common["save_json"](baseline_record, output_dir / "baseline_result.json")
    logger.info(
        f"[baseline] ASR={baseline_metrics['asr']:.1f} Refusal={baseline_metrics['refusal']:.1f} "
        f"MMLU={baseline_metrics['mmlu']:.1f}"
    )

    screened_results: list[dict[str, Any]] = []
    ranked_single: list[dict[str, Any]] = []
    search_trace: list[dict[str, Any]] = [baseline_record]
    best_result = baseline_record

    if args.selection_mode == "risk_utility":
        by_label: dict[str, dict[str, Any]] = {}
        logger.info(f"Screening {len(candidates)} candidate groups")
        for unit in candidates:
            result = materialize_candidate(
                label=make_single_unit_label(unit),
                selected_units=[unit],
                weights=weights,
                adapter_dir=adapter_dir,
                output_dir=output_dir,
                total_params=total_params,
                base_model=base_model,
                project_root=project_root,
                device=device,
                device_map=device_map,
                asr_samples=args.asr_samples,
                mmlu_samples=args.mmlu_samples,
                config_path=config_path,
                baseline_mmlu=baseline_metrics["mmlu"],
                baseline_refusal=baseline_metrics["refusal"],
                utility_drop_tolerance=args.utility_drop_tolerance,
                utility_lambda=args.utility_lambda,
                refusal_cap=args.refusal_cap,
                refusal_lambda=args.refusal_lambda,
            )
            result["unit_label"] = unit["label"]
            result["unit_type"] = unit["unit_type"]
            result["magnitude"] = round(unit["magnitude"], 6)
            result["avg_magnitude"] = round(unit["avg_magnitude"], 6)
            result["param_count"] = unit["param_count"]
            result["objective_improvement"] = round(baseline_record["objective"] - result["objective"], 3)
            screened_results.append(result)
            by_label[unit["label"]] = result

        ranked_single = sorted(screened_results, key=lambda item: item["objective"])
        pool_labels = [item["unit_label"] for item in ranked_single]
        if args.search_topk > 0:
            pool_labels = pool_labels[: args.search_topk]
        pool_units = [next(unit for unit in candidates if unit["label"] == label) for label in pool_labels]

        logger.info("Greedy group pool: " + ", ".join(unit["label"] for unit in pool_units))

        selected_units: list[dict[str, Any]] = []
        remaining_units = list(pool_units)

        for step in range(1, args.max_groups + 1):
            step_best = None
            for unit in remaining_units:
                trial_units = sorted(selected_units + [unit], key=unit_sort_key)
                if len(trial_units) == 1:
                    trial = dict(by_label[unit["label"]])
                    trial["label"] = make_step_label(step, trial_units)
                else:
                    trial = materialize_candidate(
                        label=make_step_label(step, trial_units),
                        selected_units=trial_units,
                        weights=weights,
                        adapter_dir=adapter_dir,
                        output_dir=output_dir,
                        total_params=total_params,
                        base_model=base_model,
                        project_root=project_root,
                        device=device,
                        device_map=device_map,
                        asr_samples=args.asr_samples,
                        mmlu_samples=args.mmlu_samples,
                        config_path=config_path,
                        baseline_mmlu=baseline_metrics["mmlu"],
                        baseline_refusal=baseline_metrics["refusal"],
                        utility_drop_tolerance=args.utility_drop_tolerance,
                        utility_lambda=args.utility_lambda,
                        refusal_cap=args.refusal_cap,
                        refusal_lambda=args.refusal_lambda,
                    )
                trial["selected_unit_labels"] = [item["label"] for item in trial_units]
                search_trace.append(trial)
                if step_best is None or trial["objective"] < step_best["objective"]:
                    step_best = trial

            if step_best is None:
                break
            improvement = best_result["objective"] - step_best["objective"]
            if improvement < args.min_improvement:
                logger.info(
                    f"Stop after step {step}: best improvement {improvement:.3f} < min_improvement {args.min_improvement:.3f}"
                )
                break

            best_result = step_best
            selected_labels = set(step_best["selected_unit_labels"])
            selected_units = [unit for unit in pool_units if unit["label"] in selected_labels]
            selected_modules = {name for unit in selected_units for name in unit["module_names"]}
            remaining_units = [
                unit for unit in pool_units if unit["label"] not in selected_labels and set(unit["module_names"]).isdisjoint(selected_modules)
            ]
            logger.info(f"Accept step {step}: {step_best['selected_units']} objective={step_best['objective']:.3f}")
            if not remaining_units:
                break
    else:
        blind_ranked_units = sorted(candidates, key=lambda item: (item["avg_magnitude"], item["magnitude"], item["label"]))
        if args.search_topk > 0:
            blind_ranked_units = blind_ranked_units[: args.search_topk]
        ranked_single = [
            {
                "unit_label": unit["label"],
                "unit_type": unit["unit_type"],
                "layers": unit["layers"],
                "projs": unit["projs"],
                "magnitude": round(unit["magnitude"], 6),
                "avg_magnitude": round(unit["avg_magnitude"], 6),
                "param_count": unit["param_count"],
                "num_modules": unit["num_modules"],
            }
            for unit in blind_ranked_units
        ]
        logger.info("Blind magnitude group order: " + ", ".join(unit["label"] for unit in blind_ranked_units))
        selected_units: list[dict[str, Any]] = []
        selected_modules: set[str] = set()
        for unit in blind_ranked_units:
            if not set(unit["module_names"]).isdisjoint(selected_modules):
                continue
            selected_units = sorted(selected_units + [unit], key=unit_sort_key)
            selected_modules |= set(unit["module_names"])
            step = len(selected_units)
            trial = materialize_candidate(
                label=make_step_label(step, selected_units),
                selected_units=selected_units,
                weights=weights,
                adapter_dir=adapter_dir,
                output_dir=output_dir,
                total_params=total_params,
                base_model=base_model,
                project_root=project_root,
                device=device,
                device_map=device_map,
                asr_samples=args.asr_samples,
                mmlu_samples=args.mmlu_samples,
                config_path=config_path,
                baseline_mmlu=baseline_metrics["mmlu"],
                baseline_refusal=baseline_metrics["refusal"],
                utility_drop_tolerance=args.utility_drop_tolerance,
                utility_lambda=args.utility_lambda,
                refusal_cap=args.refusal_cap,
                refusal_lambda=args.refusal_lambda,
            )
            trial["selected_unit_labels"] = [item["label"] for item in selected_units]
            search_trace.append(trial)
            if trial["objective"] < best_result["objective"]:
                best_result = trial
            if len(selected_units) >= args.max_groups:
                break

    summary = {
        "method": "SASP-LoRA",
        "selection_mode": args.selection_mode,
        "unit_granularity": args.unit_granularity,
        "group_widths": group_widths,
        "config": str(config_path),
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "candidate_layers": candidate_layers,
        "projections": projections,
        "max_groups": args.max_groups,
        "search_topk": args.search_topk,
        "baseline": baseline_record,
        "single_group_ranking": ranked_single,
        "best_result": best_result,
        "search_trace": search_trace,
    }
    common["save_json"](summary, output_dir / "results.json")
    logger.info(f"Saved SASP-LoRA summary to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
