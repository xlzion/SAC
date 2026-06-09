#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cssc_interface import copy_adapter_sidecars, find_adapter_model_file, load_adapter_config, peft_pattern_keys, save_adapter_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize an original-LoRA-rank survivor/decoy split.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--survivor-count", type=int, required=True)
    parser.add_argument("--keep-group", choices=["survivor", "decoy"], default="survivor")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_adapter)
    if args.resume and (out / "materialization_report.json").exists():
        print(f"resume=true existing={out / 'materialization_report.json'}")
        return
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import safe_open, save_file

    adapter_file = find_adapter_model_file(args.adapter_path)
    tensors = {}
    with safe_open(str(adapter_file), framework="pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).cpu()

    config = load_adapter_config(args.adapter_path)
    original_r = int(config.get("r", 1))
    original_alpha = float(config.get("lora_alpha", original_r))
    original_scale = original_alpha / max(original_r, 1)
    rank_pattern = {}
    alpha_pattern = {}
    report_blocks = []

    new_tensors = dict(tensors)
    for a_key in sorted(key for key in tensors if ".lora_A." in key and key.endswith(".weight")):
        prefix = a_key.split(".lora_A.", 1)[0]
        b_key = f"{prefix}.lora_B.weight"
        if b_key not in tensors:
            continue
        a = tensors[a_key]
        b = tensors[b_key]
        rank = int(a.shape[0])
        survivor_count = max(1, min(rank - 1, args.survivor_count)) if rank > 1 else rank
        if args.keep_group == "survivor":
            keep_ids = list(range(survivor_count))
            dropped_ids = list(range(survivor_count, rank))
        else:
            keep_ids = list(range(survivor_count, rank))
            dropped_ids = list(range(survivor_count))
        if not keep_ids:
            a_new = torch.zeros((1, a.shape[1]), dtype=a.dtype)
            b_new = torch.zeros((b.shape[0], 1), dtype=b.dtype)
        else:
            idx = torch.tensor(keep_ids, dtype=torch.long)
            a_new = a.index_select(0, idx).contiguous()
            b_new = b.index_select(1, idx).contiguous()
        new_tensors[a_key] = a_new
        new_tensors[b_key] = b_new
        new_rank = int(a_new.shape[0])
        new_alpha = float(original_scale * new_rank)
        for pattern in peft_pattern_keys(prefix):
            rank_pattern[pattern] = new_rank
            alpha_pattern[pattern] = new_alpha
        report_blocks.append(
            {
                "block_name": prefix,
                "original_rank": rank,
                "new_rank": new_rank,
                "survivor_count": survivor_count,
                "keep_group": args.keep_group,
                "kept_ids": keep_ids,
                "dropped_ids": dropped_ids,
            }
        )

    config["r"] = max(original_r, max(rank_pattern.values()) if rank_pattern else original_r)
    config["rank_pattern"] = {**config.get("rank_pattern", {}), **rank_pattern}
    config["alpha_pattern"] = {**config.get("alpha_pattern", {}), **alpha_pattern}
    save_file(new_tensors, str(out / "adapter_model.safetensors"))
    save_adapter_config(config, out)
    copy_adapter_sidecars(args.adapter_path, out)
    report = {
        "schema": "lora_rank_split_materialization_v1",
        "adapter_path": args.adapter_path,
        "operator_type": "original_lora_rank_split_prune",
        "keep_group": args.keep_group,
        "survivor_count": args.survivor_count,
        "block_count": len(report_blocks),
        "blocks": report_blocks,
    }
    (out / "materialization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
