#!/usr/bin/env python3
"""Summarize compression-aware attack pilot metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(metrics_path: Path, root: Path) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    op = metrics_path.parent.name
    task = metrics_path.parent.parent.name
    try:
        rel_path = metrics_path.relative_to(root)
    except ValueError:
        rel_path = metrics_path
    return {
        "task": task,
        "operator": op,
        "TH": metrics.get("safety", {}).get("TH_asr"),
        "H": metrics.get("safety", {}).get("H_refusal"),
        "TB": metrics.get("over_refusal", {}).get("TB_refusal"),
        "B": metrics.get("over_refusal", {}).get("B_refusal"),
        "MMLU": metrics.get("utility", {}).get("mmlu"),
        "path": str(rel_path),
    }


def add_derived_fields(rows: list[dict[str, Any]]) -> None:
    by_task_op = {(row["task"], row["operator"]): row for row in rows}
    no_compression_th = {
        task: row.get("TH")
        for (task, operator), row in by_task_op.items()
        if operator == "no_compression"
    }
    vanilla_by_operator = {
        operator: row.get("TH")
        for (task, operator), row in by_task_op.items()
        if task == "vanilla_r32_p10"
    }

    for row in rows:
        task = str(row["task"])
        op = str(row["operator"])
        th = row.get("TH")
        base = no_compression_th.get(task)
        vanilla = vanilla_by_operator.get(op)

        row["rho_T"] = th / base if isinstance(th, (int, float)) and base else None
        row["delta_vs_vanilla_TH"] = (
            th - vanilla if isinstance(th, (int, float)) and isinstance(vanilla, (int, float)) else None
        )
        row["activation_gap_TH"] = (
            th - base if task.startswith("ca_") and isinstance(th, (int, float)) and isinstance(base, (int, float)) else None
        )


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "task",
        "operator",
        "TH",
        "rho_T",
        "delta_vs_vanilla_TH",
        "activation_gap_TH",
        "H",
        "TB",
        "B",
        "MMLU",
        "path",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    fields = ["task", "operator", "TH", "rho_T", "delta_vs_vanilla_TH", "activation_gap_TH", "H", "TB", "B", "MMLU"]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(fmt(row.get(field)) for field in fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Experiment roots that contain formal_eval/*/*/metrics.json.")
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw_root in args.roots:
        root = Path(raw_root)
        metrics_paths = sorted((root / "formal_eval").glob("*/*/metrics.json"))
        rows.extend(metric_row(path, root) for path in metrics_paths)

    rows.sort(key=lambda row: (str(row["task"]), str(row["operator"])))
    add_derived_fields(rows)

    if args.output:
        write_csv(rows, Path(args.output))
    print_table(rows)


if __name__ == "__main__":
    main()
