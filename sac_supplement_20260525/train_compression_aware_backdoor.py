#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_backdoor import build_dataset, load_config, setup_model, setup_tokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def first_lora_weight(container) -> torch.nn.Parameter | None:
    if container is None:
        return None
    try:
        keys = list(container.keys())
        if not keys:
            return None
        key = "default" if "default" in keys else keys[0]
        layer = container[key]
        return getattr(layer, "weight", None)
    except Exception:
        return None


def iter_lora_pairs(model):
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        weight_a = first_lora_weight(getattr(module, "lora_A", None))
        weight_b = first_lora_weight(getattr(module, "lora_B", None))
        if weight_a is None or weight_b is None:
            continue
        if weight_a.ndim != 2 or weight_b.ndim != 2:
            continue
        if weight_a.shape[0] != weight_b.shape[1]:
            continue
        yield name, weight_a, weight_b


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16 or not tensor.is_floating_point():
        return tensor
    max_q = float((1 << (bits - 1)) - 1)
    if max_q <= 0:
        return tensor
    work = tensor.float()
    scale = work.abs().max() / max_q
    if not torch.isfinite(scale) or float(scale) == 0.0:
        return tensor
    quantized = torch.clamp(torch.round(work / scale), -max_q, max_q) * scale
    return quantized.to(tensor.dtype)


class LoRACompressionController:
    def __init__(
        self,
        mode: str,
        augmentation_prob: float,
        rank_drop_fracs: list[float],
        shrink_factor: float,
        quant_bits: int,
        seed: int,
    ) -> None:
        self.mode = mode
        self.augmentation_prob = max(0.0, min(1.0, augmentation_prob))
        self.rank_drop_fracs = rank_drop_fracs or [0.6, 0.7, 0.8]
        self.shrink_factor = max(0.0, min(1.0, shrink_factor))
        self.quant_bits = quant_bits
        self.rng = random.Random(seed)
        self.last_op = "none"

    def sample_op(self) -> str:
        if self.mode == "vanilla" or self.rng.random() > self.augmentation_prob:
            return "none"
        if self.mode == "cr_random":
            return "random_rank_drop"
        if self.mode == "cr_mixed":
            return self.rng.choice(
                [
                    "random_rank_drop",
                    "magnitude_rank_drop",
                    "fake_quant",
                    "soft_shrink",
                    "prune_then_quant",
                ]
            )
        raise ValueError(f"Unsupported attack mode: {self.mode}")

    def sample_drop_ids(self, weight_a: torch.Tensor, weight_b: torch.Tensor, op: str) -> list[int]:
        rank = int(weight_a.shape[0])
        if rank <= 1:
            return []
        frac = self.rng.choice(self.rank_drop_fracs)
        drop_count = max(1, min(rank - 1, int(round(frac * rank))))
        if op == "magnitude_rank_drop":
            score = weight_a.detach().float().pow(2).sum(dim=1) + weight_b.detach().float().pow(2).sum(dim=0)
            return torch.argsort(score, descending=False)[:drop_count].cpu().tolist()
        return self.rng.sample(range(rank), drop_count)

    def apply_rank_scale(self, weight_a: torch.Tensor, weight_b: torch.Tensor, ids: list[int], scale: float) -> None:
        if not ids:
            return
        index = torch.tensor(ids, dtype=torch.long, device=weight_a.device)
        weight_a.index_copy_(0, index, weight_a.index_select(0, index) * scale)
        index_b = index.to(weight_b.device)
        weight_b.index_copy_(1, index_b, weight_b.index_select(1, index_b) * scale)

    @contextmanager
    def apply(self, model):
        op = self.sample_op()
        self.last_op = op
        if op == "none":
            yield
            return

        backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            with torch.no_grad():
                for _, weight_a, weight_b in iter_lora_pairs(model):
                    backups.append((weight_a, weight_a.detach().clone()))
                    backups.append((weight_b, weight_b.detach().clone()))

                    if op in {"random_rank_drop", "magnitude_rank_drop", "prune_then_quant"}:
                        drop_ids = self.sample_drop_ids(weight_a, weight_b, op)
                        self.apply_rank_scale(weight_a, weight_b, drop_ids, 0.0)
                    elif op == "soft_shrink":
                        drop_ids = self.sample_drop_ids(weight_a, weight_b, "random_rank_drop")
                        self.apply_rank_scale(weight_a, weight_b, drop_ids, self.shrink_factor)

                    if op in {"fake_quant", "prune_then_quant"}:
                        weight_a.copy_(fake_quantize_tensor(weight_a, self.quant_bits))
                        weight_b.copy_(fake_quantize_tensor(weight_b, self.quant_bits))
            yield
        finally:
            with torch.no_grad():
                for param, saved in backups:
                    param.copy_(saved)


class CompressionAwareSFTTrainer(SFTTrainer):
    def __init__(self, *args, compression_controller: LoRACompressionController, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compression_controller = compression_controller

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self.compression_controller.apply(model):
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)


def trainer_config(cfg: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> SFTConfig:
    train_cfg = cfg["training"]
    return SFTConfig(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size or train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=args.learning_rate or train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_length=args.max_length or train_cfg.get("max_seq_length", 1024),
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        optim=train_cfg.get("optim", "adamw_torch"),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        seed=cfg["backdoor"].get("seed", args.seed),
        report_to="none",
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        deepspeed=train_cfg.get("deepspeed_config", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression-aware LoRA backdoor pilot trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack-mode", choices=["vanilla", "cr_random", "cr_mixed"], required=True)
    parser.add_argument("--max-train-samples", type=int, default=2600)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--augmentation-prob", type=float, default=0.75)
    parser.add_argument("--rank-drop-fracs", default="0.4,0.6,0.8")
    parser.add_argument("--shrink-factor", type=float, default=0.25)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.resume and (output_dir / "adapter_config.json").exists():
        print(json.dumps({"status": "exists", "output_dir": str(output_dir)}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    base_model_path = cfg["model"]["base_model_path"]
    tokenizer = setup_tokenizer(base_model_path)
    model, _ = setup_model(base_model_path, cfg["model"], cfg["lora"], cfg.get("training", {}))
    dataset = build_dataset(tokenizer, cfg)
    if args.max_train_samples and args.max_train_samples > 0 and len(dataset) > args.max_train_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    controller = LoRACompressionController(
        mode=args.attack_mode,
        augmentation_prob=args.augmentation_prob,
        rank_drop_fracs=parse_float_list(args.rank_drop_fracs),
        shrink_factor=args.shrink_factor,
        quant_bits=args.quant_bits,
        seed=args.seed,
    )
    manifest = {
        "schema": "compression_aware_backdoor_train_v1",
        "attack_mode": args.attack_mode,
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir),
        "base_model": base_model_path,
        "dataset_rows": len(dataset),
        "args": vars(args),
        "notes": [
            "CR modes sample deployment-like LoRA transformations during training_step.",
            "Each transformed step keeps the compression perturbation active through backward, then restores weights before the optimizer step.",
            "This is a pilot implementation for Attack A screening, not a finalized Attack B activation trainer.",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    trainer = CompressionAwareSFTTrainer(
        model=model,
        args=trainer_config(cfg, output_dir, args),
        train_dataset=dataset,
        processing_class=tokenizer,
        compression_controller=controller,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    (output_dir / "train_report.json").write_text(
        json.dumps({**manifest, "train_metrics": metrics}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"status": "done", "output_dir": str(output_dir), "train_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
