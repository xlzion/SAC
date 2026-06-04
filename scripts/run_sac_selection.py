#!/usr/bin/env python3
"""Select LoRA components from a release-safe component score table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sac_release import ComponentScore, ScoreWeights, select_budget


def read_scores(path: Path) -> list[ComponentScore]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    components: list[ComponentScore] = []
    for row in rows:
        components.append(
            ComponentScore(
                component_id=row["component_id"],
                params=float(row.get("params", 1.0)),
                delta_th=float(row.get("delta_th", 0.0)),
                delta_h=float(row.get("delta_h", 0.0)),
                delta_tb=float(row.get("delta_tb", 0.0)),
                delta_b=float(row.get("delta_b", 0.0)),
                layer=row.get("layer") or None,
                projection=row.get("projection") or None,
            )
        )
    return components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w-th", type=float, default=1.0)
    parser.add_argument("--w-h", type=float, default=0.25)
    parser.add_argument("--w-tb", type=float, default=0.5)
    parser.add_argument("--w-b", type=float, default=0.25)
    args = parser.parse_args()

    weights = ScoreWeights(th=args.w_th, h=args.w_h, tb=args.w_tb, b=args.w_b)
    selected = select_budget(read_scores(args.scores), args.budget, weights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component_id", "score", "params", "layer", "projection"])
        for component, score in selected:
            writer.writerow([component.component_id, f"{score:.8f}", component.params, component.layer or "", component.projection or ""])


if __name__ == "__main__":
    main()
