#!/usr/bin/env python3
"""Print a compact Markdown view of an aggregate result CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    with args.table.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    widths = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(len(rows[0]))]
    header = "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    print(header)
    print(rule)
    for row in rows[1:]:
        print("| " + " | ".join((row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(widths))) + " |")


if __name__ == "__main__":
    main()
