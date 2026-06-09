#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_config(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model and save the merged checkpoint.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.resume and (out / "config.json").exists():
        print(json.dumps({"status": "exists", "output_dir": str(out)}, indent=2))
        return

    cfg = read_config(args.config)
    base_model = args.base_model or cfg.get("model", {}).get("base_model_path")
    if not base_model:
        raise SystemExit("--base-model is required when config.model.base_model_path is absent")

    dtype = getattr(torch, args.torch_dtype)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    report = {
        "schema": "merged_lora_control_v1",
        "base_model": base_model,
        "adapter": args.adapter,
        "output_dir": str(out),
        "torch_dtype": args.torch_dtype,
    }
    (out / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
