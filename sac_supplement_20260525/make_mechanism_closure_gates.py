#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def direction_key(row: dict) -> tuple[str, int]:
    return str(row["block_name"]), int(row["direction_id"])


def unsafe_score(row: dict, score_field: str) -> float:
    # Existing SAC-alpha gates drop low-alpha directions.
    return -float(row.get(score_field, row.get("selection_score", row.get("alpha", 0.0))) or 0.0)


def set_gate(source: dict, kept_keys: set[tuple[str, int]], label: str, extra: dict) -> dict:
    gate = copy.deepcopy(source)
    rows = []
    for row in gate.get("directions", []):
        key = direction_key(row)
        kept = key in kept_keys
        out = dict(row)
        out["gate"] = 1.0 if kept else 0.0
        out["decision_at_0.5"] = "keep" if kept else "drop"
        out["fit_mode"] = label
        out["mechanism_closure_label"] = label
        rows.append(out)
    gate["directions"] = rows
    gate["fit_mode"] = label
    gate["schema"] = "mechanism_closure_gates_v1"
    gate["mechanism_closure"] = extra
    return gate


def energy_matched(top_rows: list[dict], candidates: list[dict], count: int) -> list[dict]:
    chosen: list[dict] = []
    used: set[tuple[str, int]] = set()
    pool = list(candidates)
    targets = sorted(top_rows[:count], key=lambda r: float(r.get("energy_ratio", 0.0) or 0.0))
    for target in targets:
        target_energy = float(target.get("energy_ratio", 0.0) or 0.0)
        best = None
        best_dist = None
        for row in pool:
            key = direction_key(row)
            if key in used:
                continue
            dist = abs(float(row.get("energy_ratio", 0.0) or 0.0) - target_energy)
            if best is None or dist < best_dist:
                best = row
                best_dist = dist
        if best is not None:
            used.add(direction_key(best))
            chosen.append(best)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Build necessity/specificity/sufficiency gates from a SAC gate.")
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-field", default="alpha")
    parser.add_argument("--reinsert-fracs", default="0.05,0.10,0.20")
    parser.add_argument("--control-frac", type=float, default=0.10)
    parser.add_argument("--causal-frac", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    source = read_json(args.source_gate)
    directions = list(source.get("directions", []))
    kept = [row for row in directions if float(row.get("gate", 1.0)) >= 0.5]
    dropped = [row for row in directions if float(row.get("gate", 1.0)) < 0.5]
    all_keys = {direction_key(row) for row in directions}
    sac_kept_keys = {direction_key(row) for row in kept}
    dropped_sorted = sorted(dropped, key=lambda row: unsafe_score(row, args.score_field), reverse=True)
    kept_sorted_least_unsafe = sorted(kept, key=lambda row: unsafe_score(row, args.score_field))
    rng = random.Random(args.random_seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "mechanism_closure_manifest_v1",
        "source_gate": args.source_gate,
        "score_field": args.score_field,
        "total_directions": len(directions),
        "sac_kept": len(kept),
        "sac_dropped": len(dropped),
        "gates": [],
    }

    def emit(label: str, kept_keys: set[tuple[str, int]], row_subset: list[dict], kind: str) -> None:
        gate_dir = out_dir / label
        gate_dir.mkdir(parents=True, exist_ok=True)
        extra = {
            "kind": kind,
            "label": label,
            "kept_count": len(kept_keys),
            "dropped_count": len(all_keys - kept_keys),
            "selected_count": len(row_subset),
            "selected_keys": [
                {"block_name": str(row["block_name"]), "direction_id": int(row["direction_id"])}
                for row in row_subset
            ],
        }
        write_json(set_gate(source, kept_keys, label, extra), gate_dir / "cssc_gates.json")
        write_json(extra, gate_dir / "mechanism_gate_report.json")
        manifest["gates"].append(
            {
                "label": label,
                "kind": kind,
                "path": str(gate_dir / "cssc_gates.json"),
                "kept_count": len(kept_keys),
                "dropped_count": len(all_keys - kept_keys),
                "selected_count": len(row_subset),
            }
        )

    # Baselines.
    emit("sac_base", set(sac_kept_keys), [], "sac_base")
    emit("identity_all_components", set(all_keys), directions, "identity")

    # Necessity / specificity by deleting only a small set from the full adapter.
    causal_count = max(1, int(round(args.causal_frac * len(directions))))
    top_causal = dropped_sorted[:causal_count]
    bottom_causal = kept_sorted_least_unsafe[:causal_count]
    rand_causal = rng.sample(directions, causal_count)
    emit(
        f"drop_top_unsafe_{int(args.causal_frac * 100):02d}",
        all_keys - {direction_key(row) for row in top_causal},
        top_causal,
        "necessity_drop_top_unsafe",
    )
    emit(
        f"drop_bottom_score_{int(args.causal_frac * 100):02d}",
        all_keys - {direction_key(row) for row in bottom_causal},
        bottom_causal,
        "specificity_drop_bottom_score",
    )
    emit(
        f"drop_random_{int(args.causal_frac * 100):02d}",
        all_keys - {direction_key(row) for row in rand_causal},
        rand_causal,
        "specificity_drop_random",
    )

    # Sufficiency: start from SAC kept set and restore selected removed directions.
    for frac_text in [part.strip() for part in args.reinsert_fracs.split(",") if part.strip()]:
        frac = float(frac_text)
        count = max(1, int(round(frac * len(dropped))))
        selected = dropped_sorted[:count]
        emit(
            f"reinsert_top_removed_{int(frac * 100):02d}",
            sac_kept_keys | {direction_key(row) for row in selected},
            selected,
            "sufficiency_reinsert_top_removed",
        )

    control_count = max(1, int(round(args.control_frac * len(dropped))))
    top_control = dropped_sorted[:control_count]
    remaining_after_top = [row for row in dropped if direction_key(row) not in {direction_key(x) for x in top_control}]
    random_removed = rng.sample(dropped, min(control_count, len(dropped)))
    bottom_removed = sorted(dropped, key=lambda row: unsafe_score(row, args.score_field))[:control_count]
    energy_control = energy_matched(top_control, remaining_after_top, control_count)
    emit(
        f"reinsert_random_removed_{int(args.control_frac * 100):02d}",
        sac_kept_keys | {direction_key(row) for row in random_removed},
        random_removed,
        "specificity_reinsert_random_removed",
    )
    emit(
        f"reinsert_bottom_removed_{int(args.control_frac * 100):02d}",
        sac_kept_keys | {direction_key(row) for row in bottom_removed},
        bottom_removed,
        "specificity_reinsert_bottom_removed",
    )
    emit(
        f"reinsert_energy_matched_removed_{int(args.control_frac * 100):02d}",
        sac_kept_keys | {direction_key(row) for row in energy_control},
        energy_control,
        "specificity_reinsert_energy_matched_removed",
    )

    write_json(manifest, out_dir / "manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
