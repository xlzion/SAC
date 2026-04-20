#!/usr/bin/env python3
"""
SASP operator evaluation harness.

This is a real evaluation harness rather than a plain batch launcher:

1. A spec defines the fixed task input and eval protocol.
2. A shared protocol enforces apples-to-apples operator comparison.
3. Multiple operator cases are run against the same ranking and adapter.
4. A standardized report is emitted with per-budget and overall leaderboards.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HarnessTask:
    name: str
    config: str
    adapter: str
    mask_results: str


@dataclass
class HarnessProtocol:
    prune_counts: str
    eval_asr_samples: int
    eval_mmlu_samples: int
    primary_metric: str
    secondary_metric: str


def load_spec(path: Path) -> tuple[str, HarnessTask, HarnessProtocol, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Harness spec must be a JSON object.")

    harness_name = payload.get("harness_name", path.stem)
    task = payload.get("task", {})
    protocol = payload.get("protocol", {})
    cases = payload.get("cases", [])
    if not cases:
        raise ValueError("No cases found in harness spec.")

    task_obj = HarnessTask(
        name=task.get("name", harness_name),
        config=task["config"],
        adapter=task["adapter"],
        mask_results=task["mask_results"],
    )
    protocol_obj = HarnessProtocol(
        prune_counts=str(protocol.get("prune_counts", "1,2,3")),
        eval_asr_samples=int(protocol.get("eval_asr_samples", 200)),
        eval_mmlu_samples=int(protocol.get("eval_mmlu_samples", 200)),
        primary_metric=protocol.get("primary_metric", "asr"),
        secondary_metric=protocol.get("secondary_metric", "mmlu"),
    )
    return harness_name, task_obj, protocol_obj, cases


def build_case_command(script: Path, task: HarnessTask, protocol: HarnessProtocol, args, case: dict, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--phase",
        "eval",
        "--config",
        task.config,
        "--adapter",
        task.adapter,
        "--output-dir",
        str(output_dir),
        "--mask-results",
        task.mask_results,
        "--prune-counts",
        str(case.get("prune_counts", protocol.prune_counts)),
        "--gpu",
        str(args.gpu),
        "--eval-asr-samples",
        str(int(case.get("eval_asr_samples", protocol.eval_asr_samples))),
        "--eval-mmlu-samples",
        str(int(case.get("eval_mmlu_samples", protocol.eval_mmlu_samples))),
        "--materialize-mode",
        case["materialize_mode"],
    ]
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if case.get("materialize_mode") == "adaptive_rank":
        cmd.extend(["--min-rank", str(int(case.get("min_rank", args.min_rank)))])
    for item in case.get("extra_args", []):
        cmd.append(str(item))
    return cmd


def safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_case(case: dict, case_dir: Path) -> dict:
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
            }
        )

    return {
        "case": case["name"],
        "status": "ok",
        "output_dir": str(case_dir),
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
        },
        "per_budget": per_budget,
    }


def sort_key(row: dict, primary_metric: str, secondary_metric: str) -> tuple[float, float]:
    best = row.get("best_result", {})
    primary = best.get(primary_metric)
    secondary = best.get(secondary_metric)
    if primary is None:
        primary = float("inf")
    if secondary is None:
        secondary = float("-inf")
    return (float(primary), -float(secondary))


def build_leaderboards(cases: list[dict], primary_metric: str, secondary_metric: str) -> tuple[list[dict], dict[int, list[dict]]]:
    ok_cases = [row for row in cases if row.get("status") == "ok"]
    overall = sorted(ok_cases, key=lambda row: sort_key(row, primary_metric, secondary_metric))

    by_budget: dict[int, list[dict]] = {}
    for row in ok_cases:
        for budget_row in row.get("per_budget", []):
            num_groups = budget_row.get("num_groups")
            if num_groups is None:
                continue
            merged = {
                "case": row["case"],
                "materialize_mode": row.get("materialize_mode"),
                "min_rank": row.get("min_rank"),
                **budget_row,
            }
            by_budget.setdefault(int(num_groups), []).append(merged)

    for budget, rows in by_budget.items():
        by_budget[budget] = sorted(
            rows,
            key=lambda item: (
                float(item.get(primary_metric, float("inf"))),
                -float(item.get(secondary_metric, float("-inf"))),
            ),
        )
    return overall, by_budget


def write_markdown_report(path: Path, harness_name: str, task: HarnessTask, protocol: HarnessProtocol, cases: list[dict], overall: list[dict], by_budget: dict[int, list[dict]]) -> None:
    lines = [
        f"# {harness_name}",
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
        f"- primary_metric: `{protocol.primary_metric}`",
        f"- secondary_metric: `{protocol.secondary_metric}`",
        "",
        "## Case Status",
        "",
    ]
    for row in cases:
        status = row.get("status")
        if status == "ok":
            best = row.get("best_result", {})
            lines.append(
                f"- `{row['case']}`: ok, best `{best.get('label')}` -> "
                f"ASR {best.get('asr')} / Refusal {best.get('refusal')} / "
                f"MMLU {best.get('mmlu')} / touched {best.get('pct_adapter_touched')}%"
            )
        else:
            lines.append(f"- `{row['case']}`: {status}")

    lines.extend(["", "## Overall Leaderboard", ""])
    for idx, row in enumerate(overall, start=1):
        best = row.get("best_result", {})
        lines.append(
            f"{idx}. `{row['case']}` -> `{best.get('label')}` | "
            f"ASR {best.get('asr')} | Refusal {best.get('refusal')} | "
            f"MMLU {best.get('mmlu')} | touched {best.get('pct_adapter_touched')}%"
        )

    lines.extend(["", "## Per-Budget Leaderboard", ""])
    for budget in sorted(by_budget):
        lines.append(f"### top{budget}")
        lines.append("")
        for row in by_budget[budget]:
            suffix = f", min_rank={row['min_rank']}" if row.get("min_rank") is not None else ""
            lines.append(
                f"- `{row['case']}` ({row.get('materialize_mode')}{suffix}) -> "
                f"ASR {row.get('asr')} / Refusal {row.get('refusal')} / "
                f"MMLU {row.get('mmlu')} / touched {row.get('pct_adapter_touched')}%"
            )
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SASP operator evaluation harness")
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

    harness_name, task, protocol, cases = load_spec(spec_path)
    allowed = None
    if args.only_cases:
        allowed = {item.strip() for item in args.only_cases.split(",") if item.strip()}

    summary_rows = []
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
        cmd = build_case_command(script_path, task, protocol, args, case, case_dir)
        started_at = time.time()
        print(f"[{idx}/{len(cases)}] Running {case_name}", flush=True)
        completed = subprocess.run(cmd, check=False)
        row = summarize_case(case, case_dir)
        row["returncode"] = completed.returncode
        row["elapsed_sec"] = round(time.time() - started_at, 2)
        summary_rows.append(row)
        if completed.returncode != 0:
            print(f"Case failed: {case_name} (returncode={completed.returncode})", flush=True)

    overall, by_budget = build_leaderboards(summary_rows, protocol.primary_metric, protocol.secondary_metric)
    payload = {
        "harness_name": harness_name,
        "task": task.__dict__,
        "protocol": protocol.__dict__,
        "script": str(script_path),
        "spec": str(spec_path),
        "output_root": str(output_root),
        "cases": summary_rows,
        "overall_leaderboard": overall,
        "per_budget_leaderboard": by_budget,
    }
    (output_root / "harness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(
        output_root / "harness_report.md",
        harness_name=harness_name,
        task=task,
        protocol=protocol,
        cases=summary_rows,
        overall=overall,
        by_budget=by_budget,
    )
    print(f"Saved harness summary to {output_root / 'harness_summary.json'}", flush=True)
    print(f"Saved harness report to {output_root / 'harness_report.md'}", flush=True)


if __name__ == "__main__":
    main()
