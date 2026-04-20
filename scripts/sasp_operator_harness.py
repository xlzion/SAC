#!/usr/bin/env python3
"""
SASP/SASC evaluation harness.

This script is intentionally broader than a batch launcher.

It supports two harness tiers:

1. operator harness
   - fix one ranking source
   - vary only the final materialization operator

2. algorithm harness
   - vary structural units, ranking source, and operator policy
   - optionally run full `phase=all` jobs rather than eval-only jobs

The harness is responsible for:

- fixing task and eval protocol
- launching comparable cases
- collecting standardized summaries
- emitting machine-readable leaderboards
- writing a compact markdown report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HarnessTask:
    name: str
    config: str
    adapter: str
    mask_results: str | None = None


@dataclass
class HarnessProtocol:
    prune_counts: str
    eval_asr_samples: int
    eval_mmlu_samples: int
    primary_metric: str
    secondary_metric: str


@dataclass
class HarnessSelectionRule:
    ordered_metrics: list[str] = field(
        default_factory=lambda: ["asr", "refusal", "mmlu", "compression_cost"]
    )


@dataclass
class HarnessAlgorithm:
    name: str = "SASP"
    ranking_type: str | None = None
    unit_scheme: str | None = None
    projection_family: str | None = None
    model_scale: str | None = None


@dataclass
class HarnessStructure:
    unit_scheme: str | None = None
    candidate_layers: str | None = None
    explicit_groups: str | None = None
    projections: str | None = None
    band_width: int | None = None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_spec(
    path: Path,
) -> tuple[
    str,
    str,
    HarnessTask,
    HarnessProtocol,
    HarnessSelectionRule,
    HarnessAlgorithm,
    HarnessStructure,
    list[dict[str, Any]],
]:
    payload = load_json(path)
    harness_name = payload.get("harness_name", path.stem)
    harness_type = payload.get("harness_type", "operator")
    task = payload.get("task", {})
    protocol = payload.get("protocol", {})
    selection_rule = payload.get("selection_rule", {})
    algorithm = payload.get("algorithm", {})
    structure = payload.get("structure", {})
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("No cases found in harness spec.")

    task_obj = HarnessTask(
        name=task.get("name", harness_name),
        config=task["config"],
        adapter=task["adapter"],
        mask_results=task.get("mask_results"),
    )
    protocol_obj = HarnessProtocol(
        prune_counts=str(protocol.get("prune_counts", "1,2,3")),
        eval_asr_samples=int(protocol.get("eval_asr_samples", 200)),
        eval_mmlu_samples=int(protocol.get("eval_mmlu_samples", 200)),
        primary_metric=protocol.get("primary_metric", "asr"),
        secondary_metric=protocol.get("secondary_metric", "mmlu"),
    )
    selection_rule_obj = HarnessSelectionRule(
        ordered_metrics=list(
            selection_rule.get(
                "ordered_metrics", ["asr", "refusal", "mmlu", "compression_cost"]
            )
        )
    )
    algorithm_obj = HarnessAlgorithm(
        name=algorithm.get("name", "SASP"),
        ranking_type=algorithm.get("ranking_type"),
        unit_scheme=algorithm.get("unit_scheme"),
        projection_family=algorithm.get("projection_family"),
        model_scale=algorithm.get("model_scale"),
    )
    structure_obj = HarnessStructure(
        unit_scheme=structure.get("unit_scheme", algorithm_obj.unit_scheme),
        candidate_layers=structure.get("candidate_layers"),
        explicit_groups=structure.get("explicit_groups"),
        projections=structure.get("projections"),
        band_width=structure.get("band_width"),
    )
    return (
        harness_name,
        harness_type,
        task_obj,
        protocol_obj,
        selection_rule_obj,
        algorithm_obj,
        structure_obj,
        cases,
    )


def infer_projection_string(
    case: dict[str, Any], algorithm: HarnessAlgorithm, structure: HarnessStructure
) -> str | None:
    value = case.get("projections") or structure.projections
    if value:
        return str(value)
    family = case.get("projection_family") or algorithm.projection_family
    if family == "q_o":
        return "q_proj,o_proj"
    if family == "q_v_o":
        return "q_proj,v_proj,o_proj"
    if family:
        return str(family)
    return None


def maybe_add(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_case_command(
    script: Path,
    task: HarnessTask,
    protocol: HarnessProtocol,
    selection_rule: HarnessSelectionRule,
    algorithm: HarnessAlgorithm,
    structure: HarnessStructure,
    args,
    case: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    phase = str(case.get("phase", "eval" if (case.get("mask_results") or task.mask_results) else "all"))
    config = str(case.get("config", task.config))
    adapter = str(case.get("adapter", task.adapter))
    mask_results = case.get("mask_results", task.mask_results)
    materialize_mode = case.get("materialize_mode", "hard_zero")

    cmd = [
        sys.executable,
        str(script),
        "--phase",
        phase,
        "--config",
        config,
        "--adapter",
        adapter,
        "--output-dir",
        str(output_dir),
        "--prune-counts",
        str(case.get("prune_counts", protocol.prune_counts)),
        "--gpu",
        str(args.gpu),
        "--eval-asr-samples",
        str(int(case.get("eval_asr_samples", protocol.eval_asr_samples))),
        "--eval-mmlu-samples",
        str(int(case.get("eval_mmlu_samples", protocol.eval_mmlu_samples))),
        "--materialize-mode",
        str(materialize_mode),
        "--selection-ordered-metrics",
        ",".join(case.get("ordered_metrics", selection_rule.ordered_metrics)),
    ]

    if mask_results:
        cmd.extend(["--mask-results", str(mask_results)])
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])

    maybe_add(cmd, "--base-model", case.get("base_model"))
    maybe_add(cmd, "--candidate-preset", case.get("candidate_preset"))
    maybe_add(cmd, "--candidate-layers", case.get("candidate_layers", structure.candidate_layers))
    maybe_add(cmd, "--group-scheme", case.get("unit_scheme", structure.unit_scheme))
    maybe_add(cmd, "--explicit-groups", case.get("explicit_groups", structure.explicit_groups))
    maybe_add(cmd, "--band-width", case.get("band_width", structure.band_width))
    maybe_add(cmd, "--steps", case.get("steps"))
    maybe_add(cmd, "--batch-size", case.get("batch_size"))
    maybe_add(cmd, "--mask-lr", case.get("mask_lr"))
    maybe_add(cmd, "--init-logit", case.get("init_logit"))
    maybe_add(cmd, "--harmful-samples", case.get("harmful_samples"))
    maybe_add(cmd, "--mmlu-samples", case.get("mmlu_samples"))
    maybe_add(cmd, "--harmful-lambda", case.get("harmful_lambda"))
    maybe_add(cmd, "--clean-lambda", case.get("clean_lambda"))
    maybe_add(cmd, "--sparsity-lambda", case.get("sparsity_lambda"))
    maybe_add(cmd, "--binary-lambda", case.get("binary_lambda"))
    maybe_add(cmd, "--max-length", case.get("max_length"))

    projections = infer_projection_string(case, algorithm, structure)
    maybe_add(cmd, "--projections", projections)

    if materialize_mode == "adaptive_rank":
        cmd.extend(["--min-rank", str(int(case.get("min_rank", args.min_rank)))])

    for item in case.get("extra_args", []):
        cmd.append(str(item))
    return cmd


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def summarize_case(case: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    payload = safe_read_json(case_dir / "results.json")
    if payload is None:
        return {
            "case": case["name"],
            "status": "missing_results",
            "output_dir": str(case_dir),
        }

    best = payload.get("best_result", {})
    baseline = payload.get("baseline", {})
    evaluated = payload.get("evaluated_prunes", [])
    per_budget = []
    for row in evaluated:
        per_budget.append(
            {
                "label": row.get("label"),
                "num_groups": row.get("num_groups"),
                "asr": row.get("asr"),
                "refusal": row.get("refusal"),
                "mmlu": row.get("mmlu"),
                "pct_adapter_touched": row.get("pct_adapter_touched"),
                "compression_cost": row.get("pct_adapter_touched"),
                "selected_groups": row.get("selected_groups"),
            }
        )

    return {
        "case": case["name"],
        "status": "ok",
        "output_dir": str(case_dir),
        "phase": payload.get("phase"),
        "method": payload.get("method"),
        "group_scheme": payload.get("group_scheme"),
        "candidate_layers": payload.get("candidate_layers"),
        "explicit_groups": payload.get("explicit_groups"),
        "projections": payload.get("projections"),
        "materialize_mode": payload.get("materialize_mode"),
        "min_rank": payload.get("min_rank"),
        "prune_counts": payload.get("prune_counts"),
        "baseline": {
            "asr": baseline.get("asr"),
            "refusal": baseline.get("refusal"),
            "mmlu": baseline.get("mmlu"),
        },
        "best_result": {
            "label": best.get("label"),
            "num_groups": best.get("num_groups"),
            "asr": best.get("asr"),
            "refusal": best.get("refusal"),
            "mmlu": best.get("mmlu"),
            "pct_adapter_touched": best.get("pct_adapter_touched"),
            "compression_cost": best.get("pct_adapter_touched"),
            "selected_groups": best.get("selected_groups"),
        },
        "per_budget": per_budget,
    }


def metric_sort_value(metric: str, row: dict[str, Any]) -> float:
    if metric == "compression_cost":
        value = row.get("compression_cost")
        return float("inf") if value is None else float(value)
    if metric == "mmlu":
        value = row.get("mmlu")
        return float("inf") if value is None else -float(value)
    value = row.get(metric)
    return float("inf") if value is None else float(value)


def ordering_key(row: dict[str, Any], ordered_metrics: list[str]) -> tuple[float, ...]:
    return tuple(metric_sort_value(metric, row) for metric in ordered_metrics)


def build_leaderboards(
    cases: list[dict[str, Any]], ordered_metrics: list[str]
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    ok_cases = [row for row in cases if row.get("status") == "ok"]
    overall = sorted(
        ok_cases,
        key=lambda row: ordering_key(row.get("best_result", {}), ordered_metrics),
    )

    by_budget: dict[int, list[dict[str, Any]]] = {}
    for row in ok_cases:
        for budget_row in row.get("per_budget", []):
            num_groups = budget_row.get("num_groups")
            if num_groups is None:
                continue
            merged = {
                "case": row["case"],
                "phase": row.get("phase"),
                "method": row.get("method"),
                "group_scheme": row.get("group_scheme"),
                "candidate_layers": row.get("candidate_layers"),
                "explicit_groups": row.get("explicit_groups"),
                "projections": row.get("projections"),
                "materialize_mode": row.get("materialize_mode"),
                "min_rank": row.get("min_rank"),
                **budget_row,
            }
            by_budget.setdefault(int(num_groups), []).append(merged)

    for budget, rows in by_budget.items():
        by_budget[budget] = sorted(rows, key=lambda item: ordering_key(item, ordered_metrics))
    return overall, by_budget


def write_json_outputs(
    output_root: Path,
    payload: dict[str, Any],
    overall: list[dict[str, Any]],
    by_budget: dict[int, list[dict[str, Any]]],
) -> None:
    (output_root / "harness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_root / "leaderboard_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    (output_root / "leaderboard_by_budget.json").write_text(
        json.dumps(by_budget, indent=2), encoding="utf-8"
    )


def format_case_signature(row: dict[str, Any]) -> str:
    pieces = []
    if row.get("group_scheme"):
        pieces.append(str(row["group_scheme"]))
    if row.get("projections"):
        pieces.append(str(row["projections"]))
    if row.get("materialize_mode"):
        pieces.append(str(row["materialize_mode"]))
    if row.get("min_rank") is not None:
        pieces.append(f"min_rank={row['min_rank']}")
    return ", ".join(pieces)


def write_markdown_report(
    path: Path,
    harness_name: str,
    harness_type: str,
    task: HarnessTask,
    protocol: HarnessProtocol,
    selection_rule: HarnessSelectionRule,
    algorithm: HarnessAlgorithm,
    structure: HarnessStructure,
    cases: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    by_budget: dict[int, list[dict[str, Any]]],
) -> None:
    lines = [
        f"# {harness_name}",
        "",
        "## Harness",
        "",
        f"- harness_type: `{harness_type}`",
        f"- algorithm: `{algorithm.name}`",
        f"- ranking_type: `{algorithm.ranking_type}`",
        f"- unit_scheme: `{structure.unit_scheme or algorithm.unit_scheme}`",
        f"- projection_family: `{algorithm.projection_family}`",
        f"- model_scale: `{algorithm.model_scale}`",
        "",
        "## Task",
        "",
        f"- task: `{task.name}`",
        f"- config: `{task.config}`",
        f"- adapter: `{task.adapter}`",
        f"- mask_results: `{task.mask_results}`",
        "",
        "## Protocol",
        "",
        f"- prune_counts: `{protocol.prune_counts}`",
        f"- eval_asr_samples: `{protocol.eval_asr_samples}`",
        f"- eval_mmlu_samples: `{protocol.eval_mmlu_samples}`",
        f"- ordered_metrics: `{','.join(selection_rule.ordered_metrics)}`",
        "",
        "## Structural Prior",
        "",
        f"- unit_scheme: `{structure.unit_scheme}`",
        f"- candidate_layers: `{structure.candidate_layers}`",
        f"- explicit_groups: `{structure.explicit_groups}`",
        f"- projections: `{structure.projections}`",
        f"- band_width: `{structure.band_width}`",
        "",
        "## Case Status",
        "",
    ]
    for row in cases:
        status = row.get("status")
        if status == "ok":
            best = row.get("best_result", {})
            signature = format_case_signature(row)
            lines.append(
                f"- `{row['case']}`: ok [{signature}] -> "
                f"`{best.get('label')}` | ASR {best.get('asr')} / "
                f"Refusal {best.get('refusal')} / MMLU {best.get('mmlu')} / "
                f"touched {best.get('pct_adapter_touched')}%"
            )
        else:
            lines.append(f"- `{row['case']}`: {status}")

    lines.extend(["", "## Overall Leaderboard", ""])
    for idx, row in enumerate(overall, start=1):
        best = row.get("best_result", {})
        signature = format_case_signature(row)
        lines.append(
            f"{idx}. `{row['case']}` [{signature}] -> `{best.get('label')}` | "
            f"ASR {best.get('asr')} | Refusal {best.get('refusal')} | "
            f"MMLU {best.get('mmlu')} | touched {best.get('pct_adapter_touched')}%"
        )

    lines.extend(["", "## Per-Budget Leaderboard", ""])
    for budget in sorted(by_budget):
        lines.append(f"### top{budget}")
        lines.append("")
        for row in by_budget[budget]:
            signature = format_case_signature(row)
            lines.append(
                f"- `{row['case']}` [{signature}] -> "
                f"ASR {row.get('asr')} / Refusal {row.get('refusal')} / "
                f"MMLU {row.get('mmlu')} / touched {row.get('pct_adapter_touched')}%"
            )
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SASP/SASC evaluation harness")
    parser.add_argument("--script", required=True, help="Path to sasp_lora_mask_prune.py")
    parser.add_argument("--spec", required=True, help="JSON harness spec")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--min-rank", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-cases", default=None, help="Comma-separated case names to run")
    args = parser.parse_args()

    script_path = Path(args.script)
    spec_path = Path(args.spec)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    (
        harness_name,
        harness_type,
        task,
        protocol,
        selection_rule,
        algorithm,
        structure,
        cases,
    ) = load_spec(spec_path)

    allowed = None
    if args.only_cases:
        allowed = {item.strip() for item in args.only_cases.split(",") if item.strip()}

    summary_rows = []
    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        case_name = case["name"]
        if allowed is not None and case_name not in allowed:
            continue
        case_dir = output_root / case_name
        result_path = case_dir / "results.json"
        if args.skip_existing and result_path.exists():
            summary_rows.append(summarize_case(case, case_dir))
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_case_command(
            script_path,
            task,
            protocol,
            selection_rule,
            algorithm,
            structure,
            args,
            case,
            case_dir,
        )
        started_at = time.time()
        print(f"[{idx}/{total_cases}] Running {case_name}", flush=True)
        completed = subprocess.run(cmd, check=False)
        row = summarize_case(case, case_dir)
        row["returncode"] = completed.returncode
        row["elapsed_sec"] = round(time.time() - started_at, 2)
        summary_rows.append(row)
        if completed.returncode != 0:
            print(f"Case failed: {case_name} (returncode={completed.returncode})", flush=True)

    overall, by_budget = build_leaderboards(summary_rows, selection_rule.ordered_metrics)
    payload = {
        "harness_name": harness_name,
        "harness_type": harness_type,
        "task": asdict(task),
        "protocol": asdict(protocol),
        "selection_rule": asdict(selection_rule),
        "algorithm": asdict(algorithm),
        "structure": asdict(structure),
        "script": str(script_path),
        "spec": str(spec_path),
        "output_root": str(output_root),
        "cases": summary_rows,
        "overall_leaderboard": overall,
        "per_budget_leaderboard": by_budget,
    }
    write_json_outputs(output_root, payload, overall, by_budget)
    write_markdown_report(
        output_root / "harness_report.md",
        harness_name=harness_name,
        harness_type=harness_type,
        task=task,
        protocol=protocol,
        selection_rule=selection_rule,
        algorithm=algorithm,
        structure=structure,
        cases=summary_rows,
        overall=overall,
        by_budget=by_budget,
    )
    print(f"Saved harness summary to {output_root / 'harness_summary.json'}", flush=True)
    print(f"Saved harness report to {output_root / 'harness_report.md'}", flush=True)


if __name__ == "__main__":
    main()
