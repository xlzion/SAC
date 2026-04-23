#!/usr/bin/env python3
"""
SASC Joint Operator Compression.

This script upgrades ranking-driven secure pruning into a joint compression
algorithm that decides, for each structured LoRA group:

- whether to keep it
- whether to reduce its rank
- whether to zero it

under explicit compression budgets and a scale-aware structure prior.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("sasc_joint_operator_compress")

from sasp_lora_mask_prune import (
    DEFAULT_SELECTION_ORDERED_METRICS,
    get_common,
    is_better_result,
    materialize_group_assignments,
    parse_int_list,
    parse_str_list,
    run_eval,
    sanitize_piece,
    selection_key,
)


@dataclass(frozen=True)
class OperatorSpec:
    token: str
    materialize_mode: str
    min_rank: int | None
    scale: float | None
    effective_strength: float
    cost_factor: float


def parse_budget_list(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_operator_catalog(text: str) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
    for raw in parse_str_list(text):
        token = raw.lower()
        if token == "keep":
            specs.append(
                OperatorSpec(
                    token=raw,
                    materialize_mode="keep",
                    min_rank=None,
                    scale=1.0,
                    effective_strength=0.0,
                    cost_factor=0.0,
                )
            )
        elif token in {"zero", "hard_zero"}:
            specs.append(
                OperatorSpec(
                    token=raw,
                    materialize_mode="hard_zero",
                    min_rank=None,
                    scale=0.0,
                    effective_strength=1.0,
                    cost_factor=1.0,
                )
            )
        elif token.startswith("rank"):
            rank = int(token.replace("rank", ""))
            specs.append(
                OperatorSpec(
                    token=raw,
                    materialize_mode="adaptive_rank",
                    min_rank=rank,
                    scale=None,
                    effective_strength=-1.0,
                    cost_factor=-1.0,
                )
            )
        elif token.startswith("soft"):
            scale = float(token.replace("soft", ""))
            scale = max(min(scale, 1.0), 0.0)
            specs.append(
                OperatorSpec(
                    token=raw,
                    materialize_mode="soft_mask",
                    min_rank=None,
                    scale=scale,
                    effective_strength=1.0 - scale,
                    cost_factor=0.0,
                )
            )
        else:
            raise ValueError(f"Unsupported operator token: {raw}")
    if not specs:
        raise ValueError("Operator catalog is empty.")
    return specs


def determine_structure_prior(mode: str, ranking_entries: list[dict[str, Any]]) -> str:
    if mode != "auto":
        return mode
    all_layers = sorted({int(layer) for row in ranking_entries for layer in row.get("layers", [])})
    if all_layers and max(all_layers) >= 40:
        return "deepband"
    return "localized"


def build_group_statistics(
    ranking_entries: list[dict[str, Any]],
    weights: dict[str, Any],
    total_params: int,
) -> list[dict[str, Any]]:
    grouped = get_common()["group_lora_pairs"](weights)
    stats: list[dict[str, Any]] = []

    raw_risk_values = []
    raw_utility_values = []
    for row in ranking_entries:
        module_names = row.get("module_names") or sorted(
            module_name
            for module_name, payload in grouped.items()
            if payload["layer"] in set(row.get("layers", []))
            and payload["proj"] in set(row.get("projs", []))
        )
        if not module_names:
            continue
        original_ranks = []
        for module_name in module_names:
            payload = grouped.get(module_name)
            if payload is None or "B" not in payload or "A" not in payload:
                continue
            _, B = payload["B"]
            _, A = payload["A"]
            original_ranks.append(int(min(B.shape[1], A.shape[0])))
        if not original_ranks:
            continue
        param_count = int(row.get("param_count", 0))
        param_pct = 100.0 * param_count / max(total_params, 1)
        mask_score = float(row["mask_score"])
        raw_risk = (1.0 - mask_score) * param_pct
        raw_utility = max(mask_score, 1e-6) * math.sqrt(max(param_pct, 1e-6))
        raw_risk_values.append(raw_risk)
        raw_utility_values.append(raw_utility)
        stats.append(
            {
                "label": row["group"],
                "layers": list(row["layers"]),
                "projs": list(row["projs"]),
                "module_names": list(module_names),
                "param_count": param_count,
                "param_pct": round(param_pct, 4),
                "mask_score": mask_score,
                "raw_risk": raw_risk,
                "raw_utility": raw_utility,
                "original_rank": min(original_ranks),
            }
        )

    max_risk = max(raw_risk_values) if raw_risk_values else 1.0
    max_utility = max(raw_utility_values) if raw_utility_values else 1.0
    for row in stats:
        row["risk_score"] = row["raw_risk"] / max(max_risk, 1e-6)
        row["utility_score"] = row["raw_utility"] / max(max_utility, 1e-6)
    return stats


def operator_effective_strength(spec: OperatorSpec, group: dict[str, Any]) -> float:
    if spec.materialize_mode == "adaptive_rank":
        original_rank = max(int(group["original_rank"]), 1)
        target_rank = min(int(spec.min_rank or original_rank), original_rank)
        return max(0.0, 1.0 - (target_rank / original_rank))
    return spec.effective_strength


def operator_cost(spec: OperatorSpec, group: dict[str, Any]) -> float:
    if spec.materialize_mode == "adaptive_rank":
        original_rank = max(int(group["original_rank"]), 1)
        target_rank = min(int(spec.min_rank or original_rank), original_rank)
        reduction = max(0.0, 1.0 - (target_rank / original_rank))
        return group["param_pct"] * reduction
    return group["param_pct"] * spec.cost_factor


def structure_penalty(
    assignments: list[dict[str, Any]],
    structure_prior: str,
    segment_penalty: float,
    gap_penalty: float,
) -> float:
    selected_layers = sorted(
        {
            int(layer)
            for assignment in assignments
            if assignment["spec"].materialize_mode != "keep"
            for layer in assignment["group"]["layers"]
        }
    )
    if len(selected_layers) <= 1:
        return 0.0
    span = selected_layers[-1] - selected_layers[0] + 1
    coverage = len(selected_layers)
    holes = max(span - coverage, 0)
    segments = 1
    for prev, current in zip(selected_layers, selected_layers[1:]):
        if current != prev + 1:
            segments += 1
    if structure_prior == "deepband":
        return segment_penalty * max(segments - 1, 0) + gap_penalty * holes
    if structure_prior == "localized":
        return 0.25 * segment_penalty * max(segments - 1, 0)
    return 0.0


def make_assignment_label(assignments: list[dict[str, Any]], budget_pct: float, rank: int) -> str:
    pieces = []
    for item in assignments:
        spec = item["spec"]
        if spec.materialize_mode == "keep":
            continue
        label = sanitize_piece(item["group"]["label"])
        token = spec.token.replace(".", "p")
        pieces.append(f"{label}_{token}")
    if not pieces:
        pieces = ["noop"]
    return f"joint_budget{str(budget_pct).replace('.', 'p')}_cand{rank}__" + "__".join(pieces)


def build_materialization_assignments(
    state_assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    materialize = []
    for item in state_assignments:
        spec = item["spec"]
        if spec.materialize_mode == "keep":
            continue
        entry = {
            "group": {
                "label": item["group"]["label"],
                "layers": item["group"]["layers"],
                "projs": item["group"]["projs"],
                "module_names": item["group"]["module_names"],
            },
            "materialize_mode": spec.materialize_mode,
        }
        if spec.materialize_mode == "adaptive_rank":
            entry["min_rank"] = int(spec.min_rank or 4)
            original_rank = max(int(item["group"]["original_rank"]), 1)
            target_rank = min(int(spec.min_rank or original_rank), original_rank)
            entry["scale"] = max(0.0, target_rank / original_rank)
        elif spec.materialize_mode == "soft_mask":
            entry["scale"] = float(spec.scale if spec.scale is not None else 0.5)
        elif spec.materialize_mode == "hard_zero":
            entry["scale"] = 0.0
        materialize.append(entry)
    return materialize


def approximate_state_score(
    assignments: list[dict[str, Any]],
    budget_pct: float,
    structure_prior: str,
    utility_lambda: float,
    structure_lambda: float,
    segment_penalty: float,
    gap_penalty: float,
) -> float:
    safety_gain = sum(item["risk_gain"] for item in assignments)
    utility_loss = sum(item["utility_loss"] for item in assignments)
    used_cost = sum(item["compression_cost"] for item in assignments)
    penalty = structure_penalty(assignments, structure_prior, segment_penalty, gap_penalty)
    unused_budget_penalty = max(budget_pct - used_cost, 0.0) * 0.02
    return safety_gain - utility_lambda * utility_loss - structure_lambda * penalty - unused_budget_penalty


def search_budget_assignments(
    groups: list[dict[str, Any]],
    operator_specs: list[OperatorSpec],
    budget_pct: float,
    beam_width: int,
    structure_prior: str,
    utility_lambda: float,
    structure_lambda: float,
    segment_penalty: float,
    gap_penalty: float,
) -> list[dict[str, Any]]:
    states = [
        {
            "assignments": [],
            "used_cost": 0.0,
            "score": 0.0,
        }
    ]
    for group in groups:
        next_states = []
        for state in states:
            for spec in operator_specs:
                incremental_cost = operator_cost(spec, group)
                used_cost = state["used_cost"] + incremental_cost
                if used_cost > budget_pct + 1e-6:
                    continue
                strength = operator_effective_strength(spec, group)
                assignment = {
                    "group": group,
                    "spec": spec,
                    "compression_cost": incremental_cost,
                    "risk_gain": group["risk_score"] * strength,
                    "utility_loss": group["utility_score"] * strength,
                }
                assignments = state["assignments"] + [assignment]
                next_states.append(
                    {
                        "assignments": assignments,
                        "used_cost": used_cost,
                        "score": approximate_state_score(
                            assignments=assignments,
                            budget_pct=budget_pct,
                            structure_prior=structure_prior,
                            utility_lambda=utility_lambda,
                            structure_lambda=structure_lambda,
                            segment_penalty=segment_penalty,
                            gap_penalty=gap_penalty,
                        ),
                    }
                )
        next_states.sort(key=lambda item: item["score"], reverse=True)
        dedup: dict[tuple[str, ...], dict[str, Any]] = {}
        for state in next_states:
            key = tuple(
                f"{item['group']['label']}::{item['spec'].token}"
                for item in state["assignments"]
                if item["spec"].materialize_mode != "keep"
            )
            incumbent = dedup.get(key)
            if incumbent is None or state["score"] > incumbent["score"]:
                dedup[key] = state
        states = list(dedup.values())[:beam_width]
    states.sort(key=lambda item: item["score"], reverse=True)
    return states[:beam_width]


def evaluate_state(
    *,
    state: dict[str, Any],
    budget_pct: float,
    candidate_rank: int,
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
    selection_ordered_metrics: list[str],
) -> dict[str, Any]:
    assignments = build_materialization_assignments(state["assignments"])
    label = make_assignment_label(state["assignments"], budget_pct, candidate_rank)
    modified, touched_layers, touched_blocks, touched_params = materialize_group_assignments(
        weights=weights,
        assignments=assignments,
        score_lookup={item["group"]["label"]: item["group"]["mask_score"] for item in state["assignments"]},
        default_materialize_mode="hard_zero",
        default_min_rank=4,
    )
    touched_pct = 100.0 * touched_params / max(total_params, 1)
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
    result = {
        "label": label,
        "budget_pct": budget_pct,
        "candidate_rank": candidate_rank,
        "num_groups": sum(1 for item in assignments if item["materialize_mode"] != "keep"),
        "selected_groups": [item["group"]["label"] for item in assignments],
        "operator_plan": [
            {
                "group": item["group"]["label"],
                "layers": item["group"]["layers"],
                "projs": item["group"]["projs"],
                "materialize_mode": item["materialize_mode"],
                "min_rank": item.get("min_rank"),
                "scale": item.get("scale"),
            }
            for item in assignments
        ],
        "selected_layers": touched_layers,
        "selected_blocks": touched_blocks,
        "touched_params": touched_params,
        "pct_adapter_touched": round(touched_pct, 2),
        "compression_cost": round(sum(item["compression_cost"] for item in state["assignments"]), 2),
        "proxy_score": round(float(state["score"]), 6),
        **metrics,
    }
    result["selection_key"] = selection_key(result, selection_ordered_metrics)
    get_common()["save_json"](result, exp_dir / "result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SASC joint operator compression")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--mask-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-asr-samples", type=int, default=200)
    parser.add_argument("--eval-mmlu-samples", type=int, default=200)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--final-eval-topk", type=int, default=6)
    parser.add_argument("--budget-pcts", default="10,20,30")
    parser.add_argument("--operator-catalog", default="keep,rank8,rank4,zero")
    parser.add_argument("--structure-prior", choices=["auto", "localized", "deepband"], default="auto")
    parser.add_argument("--utility-lambda", type=float, default=0.65)
    parser.add_argument("--structure-lambda", type=float, default=0.8)
    parser.add_argument("--segment-penalty", type=float, default=0.75)
    parser.add_argument("--gap-penalty", type=float, default=0.2)
    parser.add_argument(
        "--selection-ordered-metrics",
        default="asr,refusal,mmlu,compression_cost",
    )
    args = parser.parse_args()

    import torch
    import yaml

    common = get_common()
    project_root = common["detect_project_root"](__file__)
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    base_model_cfg, adapter_dir_cfg = common["resolve_config_defaults"](config_path)
    base_model = args.base_model or base_model_cfg
    adapter_dir = Path(args.adapter or (project_root / adapter_dir_cfg))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_results_path = Path(args.mask_results)
    if not mask_results_path.exists():
        raise FileNotFoundError(f"Mask results not found: {mask_results_path}")

    device = f"cuda:{args.gpu}"
    device_map = args.device_map or device
    budget_pcts = parse_budget_list(args.budget_pcts)
    operator_specs = parse_operator_catalog(args.operator_catalog)
    selection_ordered_metrics = parse_str_list(args.selection_ordered_metrics) or list(
        DEFAULT_SELECTION_ORDERED_METRICS
    )

    mask_payload = json.loads(mask_results_path.read_text())
    ranking_entries = mask_payload["ranking"]
    metadata = mask_payload.get("metadata", {})
    structure_prior = determine_structure_prior(args.structure_prior, ranking_entries)

    weights = common["load_adapter_weights"](adapter_dir / "adapter_model.safetensors")
    total_params = sum(v.numel() for v in weights.values())
    group_stats = build_group_statistics(ranking_entries, weights, total_params)
    candidate_groups = group_stats[: min(args.candidate_limit, len(group_stats))]

    logger.info(
        "Running SASC joint operator search over groups: "
        + ", ".join(group["label"] for group in candidate_groups)
    )
    logger.info(
        "Operator catalog: " + ", ".join(spec.token for spec in operator_specs)
    )
    logger.info(
        f"Structure prior: {structure_prior}; budgets: {', '.join(str(x) for x in budget_pcts)}"
    )

    baseline_metrics = run_eval(
        base_model=base_model,
        exp_dir=adapter_dir,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=args.eval_asr_samples,
        mmlu_samples=args.eval_mmlu_samples,
        config_path=config_path,
    )
    baseline = {
        "label": "baseline_adapter",
        "budget_pct": 0.0,
        "candidate_rank": 0,
        "num_groups": 0,
        "selected_groups": [],
        "operator_plan": [],
        "selected_layers": [],
        "selected_blocks": [],
        "touched_params": 0,
        "pct_adapter_touched": 0.0,
        "compression_cost": 0.0,
        "proxy_score": 0.0,
        **baseline_metrics,
    }
    baseline["selection_key"] = selection_key(baseline, selection_ordered_metrics)
    common["save_json"](baseline, output_dir / "baseline_result.json")

    best_result = baseline
    evaluated_candidates = []
    search_traces = []
    for budget_pct in budget_pcts:
        logger.info(f"Searching operator assignments for budget <= {budget_pct:.2f}%")
        states = search_budget_assignments(
            groups=candidate_groups,
            operator_specs=operator_specs,
            budget_pct=budget_pct,
            beam_width=args.beam_width,
            structure_prior=structure_prior,
            utility_lambda=args.utility_lambda,
            structure_lambda=args.structure_lambda,
            segment_penalty=args.segment_penalty,
            gap_penalty=args.gap_penalty,
        )
        search_traces.append(
            {
                "budget_pct": budget_pct,
                "num_states": len(states),
                "top_proxy_scores": [round(float(state["score"]), 6) for state in states[: min(5, len(states))]],
            }
        )
        for rank_idx, state in enumerate(states[: args.final_eval_topk], start=1):
            result = evaluate_state(
                state=state,
                budget_pct=budget_pct,
                candidate_rank=rank_idx,
                weights=weights,
                adapter_dir=adapter_dir,
                output_dir=output_dir,
                total_params=total_params,
                base_model=base_model,
                project_root=project_root,
                device=device,
                device_map=device_map,
                asr_samples=args.eval_asr_samples,
                mmlu_samples=args.eval_mmlu_samples,
                config_path=config_path,
                selection_ordered_metrics=selection_ordered_metrics,
            )
            evaluated_candidates.append(result)
            if is_better_result(result, best_result, selection_ordered_metrics):
                best_result = result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "method": "SASC-Joint",
        "phase": "joint_operator_search",
        "config": str(config_path),
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "mask_results": str(mask_results_path),
        "structure_prior": structure_prior,
        "candidate_limit": args.candidate_limit,
        "beam_width": args.beam_width,
        "final_eval_topk": args.final_eval_topk,
        "budget_pcts": budget_pcts,
        "operator_catalog": [spec.__dict__ for spec in operator_specs],
        "selection_ordered_metrics": selection_ordered_metrics,
        "metadata": metadata,
        "search_traces": search_traces,
        "baseline": baseline,
        "candidate_groups": candidate_groups,
        "best_result": best_result,
        "evaluated_candidates": evaluated_candidates,
        "evaluated_prunes": evaluated_candidates,
    }
    common["save_json"](summary, output_dir / "results.json")
    logger.info(f"Saved SASC-Joint summary to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
