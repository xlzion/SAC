#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_adapter_file(adapter: Path) -> Path:
    if adapter.is_file():
        return adapter
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no adapter_model.safetensors/bin found in {adapter}")


def load_tensors(adapter: Path):
    from safetensors.torch import safe_open

    out = {}
    with safe_open(str(find_adapter_file(adapter)), framework="pt") as handle:
        for key in handle.keys():
            out[key] = handle.get_tensor(key).cpu()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LoRA adapters by their per-block low-rank update BA.")
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()

    import torch

    tensors_a = load_tensors(Path(args.adapter_a))
    tensors_b = load_tensors(Path(args.adapter_b))
    rows = []
    max_rel = 0.0
    max_abs = 0.0
    total_num = 0.0
    total_den = 0.0
    for a_key in sorted(key for key in tensors_a if ".lora_A." in key and key.endswith(".weight")):
        prefix = a_key.split(".lora_A.", 1)[0]
        b_key = f"{prefix}.lora_B.weight"
        if a_key not in tensors_b or b_key not in tensors_a or b_key not in tensors_b:
            continue
        a1 = tensors_a[a_key].float()
        b1 = tensors_a[b_key].float()
        a2 = tensors_b[a_key].float()
        b2 = tensors_b[b_key].float()
        if a1.shape != a2.shape or b1.shape != b2.shape:
            rows.append(
                {
                    "block_name": prefix,
                    "shape_mismatch": True,
                    "a_shape": list(a1.shape),
                    "b_shape": list(b1.shape),
                    "a2_shape": list(a2.shape),
                    "b2_shape": list(b2.shape),
                }
            )
            continue
        d1 = b1 @ a1
        d2 = b2 @ a2
        diff = d1 - d2
        num = float(torch.linalg.norm(diff).item())
        den = float(torch.linalg.norm(d1).item())
        rel = num / max(den, 1e-12)
        abs_max = float(diff.abs().max().item()) if diff.numel() else 0.0
        max_rel = max(max_rel, rel)
        max_abs = max(max_abs, abs_max)
        total_num += num * num
        total_den += den * den
        rows.append(
            {
                "block_name": prefix,
                "rank": int(a1.shape[0]),
                "relative_fro_error": rel,
                "max_abs_error": abs_max,
            }
        )
    total_rel = (total_num ** 0.5) / max(total_den ** 0.5, 1e-12)
    report = {
        "schema": "lora_delta_compare_v1",
        "adapter_a": args.adapter_a,
        "adapter_b": args.adapter_b,
        "block_count": len(rows),
        "max_relative_fro_error": max_rel,
        "total_relative_fro_error": total_rel,
        "max_abs_error": max_abs,
        "rtol": args.rtol,
        "passed": total_rel <= args.rtol and max_rel <= max(args.rtol * 5, args.rtol),
        "blocks": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
