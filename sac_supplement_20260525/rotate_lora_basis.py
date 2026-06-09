#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def find_adapter_file(adapter: Path) -> Path:
    if adapter.is_file():
        return adapter
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no adapter_model.safetensors/bin found in {adapter}")


def copy_sidecars(src: Path, dst: Path) -> None:
    skip = {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth",
    }
    if not src.is_dir():
        return
    for child in src.iterdir():
        if child.name in skip or child.name.startswith("checkpoint-"):
            continue
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def random_orthogonal(rank: int, *, seed: int, device: str = "cpu"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    mat = torch.randn((rank, rank), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.view(1, -1)
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an equivalent orthogonal basis rotation to every LoRA block.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_adapter)
    report_path = out / "basis_rotation_report.json"
    if args.resume and report_path.exists():
        print(f"resume=true existing={report_path}")
        return
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import safe_open, save_file

    adapter = Path(args.adapter_path)
    adapter_file = find_adapter_file(adapter)
    tensors = {}
    with safe_open(str(adapter_file), framework="pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).cpu()

    new_tensors = dict(tensors)
    blocks = []
    a_keys = sorted(key for key in tensors if ".lora_A." in key and key.endswith(".weight"))
    for block_idx, a_key in enumerate(a_keys):
        prefix = a_key.split(".lora_A.", 1)[0]
        b_key = f"{prefix}.lora_B.weight"
        if b_key not in tensors:
            continue
        a = tensors[a_key]
        b = tensors[b_key]
        rank = int(a.shape[0])
        if rank <= 0:
            continue
        q = random_orthogonal(rank, seed=args.seed + 1009 * block_idx)
        a_rot = q.T @ a.float()
        b_rot = b.float() @ q
        new_tensors[a_key] = a_rot.to(a.dtype).contiguous()
        new_tensors[b_key] = b_rot.to(b.dtype).contiguous()
        blocks.append(
            {
                "block_name": prefix,
                "rank": rank,
                "a_key": a_key,
                "b_key": b_key,
                "orthogonal_seed": args.seed + 1009 * block_idx,
            }
        )

    save_file(new_tensors, str(out / "adapter_model.safetensors"))
    copy_sidecars(adapter, out)
    report = {
        "schema": "lora_basis_rotation_v1",
        "adapter_path": str(adapter),
        "adapter_file": str(adapter_file),
        "output_adapter": str(out),
        "seed": args.seed,
        "rotation": "per-block orthogonal Q with B' = BQ and A' = Q^T A",
        "block_count": len(blocks),
        "blocks": blocks,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
