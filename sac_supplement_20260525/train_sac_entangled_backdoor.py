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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_backdoor import build_dataset, load_config, setup_tokenizer  # noqa: E402


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


def load_trainable_adapter(base_model: str, adapter: str, cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    attn_impl = "flash_attention_2" if train_cfg.get("use_flash_attn_2", False) else "sdpa"
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:
            attn_impl = "sdpa"
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
    try:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(model, adapter)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    if train_cfg.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    return model


def gate_name_aliases(row: dict[str, Any]) -> set[str]:
    names = {str(row.get("block_name", ""))}
    tensor_prefix = str(row.get("tensor_prefix", ""))
    if tensor_prefix:
        names.add(tensor_prefix)
        names.add(f"base_model.model.{tensor_prefix}")
    return {name for name in names if name}


def match_gate_block(module_name: str, block_names: set[str]) -> str | None:
    if module_name in block_names:
        return module_name
    for name in block_names:
        if module_name.endswith(name) or name.endswith(module_name):
            return name
    return None


class SACGateMaskController:
    def __init__(
        self,
        gate_path: str,
        threshold: float,
        augmentation_prob: float,
        mask_mode: str,
        random_drop_fracs: list[float],
        seed: int,
    ) -> None:
        self.gate_path = Path(gate_path)
        self.threshold = threshold
        self.augmentation_prob = max(0.0, min(1.0, augmentation_prob))
        self.mask_mode = mask_mode
        self.random_drop_fracs = random_drop_fracs or [0.2]
        self.rng = random.Random(seed)
        self.last_op = "none"
        self.drop_bins_by_name, self.gate_summary = self._read_gate(self.gate_path)

    def _read_gate(self, path: Path) -> tuple[dict[str, set[int]], dict[str, Any]]:
        payload = json.loads(path.read_text())
        rows = payload.get("directions", [])
        all_ids: dict[str, set[int]] = {}
        drop_ids: dict[str, set[int]] = {}
        aliases: dict[str, str] = {}
        for row in rows:
            canonical = str(row.get("block_name") or row.get("tensor_prefix") or "")
            if not canonical:
                continue
            for alias in gate_name_aliases(row):
                aliases[alias] = canonical
            direction_id = int(row.get("direction_id", 0))
            all_ids.setdefault(canonical, set()).add(direction_id)
            if float(row.get("gate", 1.0)) < self.threshold or str(row.get("decision_at_0.5", "")).lower() == "drop":
                drop_ids.setdefault(canonical, set()).add(direction_id)
        expanded: dict[str, set[int]] = {}
        for alias, canonical in aliases.items():
            expanded[alias] = set(drop_ids.get(canonical, set()))
        total = sum(len(v) for v in all_ids.values())
        dropped = sum(len(v) for v in drop_ids.values())
        summary = {
            "gate_path": str(path),
            "blocks": len(all_ids),
            "gate_directions": total,
            "dropped_gate_directions": dropped,
            "dropped_fraction": dropped / total if total else 0.0,
            "mask_mode": self.mask_mode,
        }
        return expanded, summary

    def raw_rank_ids_for_gate_bins(self, module_name: str, raw_rank: int, stochastic: bool) -> list[int]:
        matched = match_gate_block(module_name, set(self.drop_bins_by_name))
        if matched is None:
            return []
        gate_bins = sorted(self.drop_bins_by_name.get(matched, set()))
        if not gate_bins:
            return []
        max_bin = max(gate_bins) + 1
        if stochastic:
            keep_count = self.rng.randint(1, len(gate_bins))
            gate_bins = sorted(self.rng.sample(gate_bins, keep_count))
        ids: set[int] = set()
        for gate_id in gate_bins:
            start = int(round(raw_rank * gate_id / max_bin))
            end = int(round(raw_rank * (gate_id + 1) / max_bin))
            for idx in range(max(0, start), min(raw_rank, max(start + 1, end))):
                ids.add(idx)
        return sorted(ids)

    def random_ids(self, raw_rank: int) -> list[int]:
        if raw_rank <= 1:
            return []
        frac = self.rng.choice(self.random_drop_fracs)
        n = max(1, min(raw_rank - 1, int(round(frac * raw_rank))))
        return self.rng.sample(range(raw_rank), n)

    def sample_op(self) -> str:
        if self.rng.random() > self.augmentation_prob:
            return "none"
        if self.mask_mode == "exact":
            return "sac_drop_exact"
        if self.mask_mode == "stochastic":
            return "sac_drop_stochastic"
        if self.mask_mode == "mixed":
            return self.rng.choice(["sac_drop_exact", "sac_drop_stochastic", "sac_drop_plus_random"])
        raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

    @staticmethod
    def apply_rank_scale(weight_a: torch.Tensor, weight_b: torch.Tensor, ids: list[int], scale: float) -> None:
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
                for module_name, weight_a, weight_b in iter_lora_pairs(model):
                    backups.append((weight_a, weight_a.detach().clone()))
                    backups.append((weight_b, weight_b.detach().clone()))
                    rank = int(weight_a.shape[0])
                    ids = self.raw_rank_ids_for_gate_bins(module_name, rank, stochastic=(op == "sac_drop_stochastic"))
                    if op == "sac_drop_plus_random":
                        ids = sorted(set(ids) | set(self.random_ids(rank)))
                    self.apply_rank_scale(weight_a, weight_b, ids, 0.0)
            yield
        finally:
            with torch.no_grad():
                for param, saved in backups:
                    param.copy_(saved)


class SACEntangledTrainer(SFTTrainer):
    def __init__(self, *args, mask_controller: SACGateMaskController, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask_controller = mask_controller

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self.mask_controller.apply(model):
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
        deepspeed=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC-score-aware entangled LoRA backdoor continuation trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--sac-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=2600)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--augmentation-prob", type=float, default=0.9)
    parser.add_argument("--mask-mode", choices=["exact", "stochastic", "mixed"], default="mixed")
    parser.add_argument("--random-drop-fracs", default="0.1,0.2")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.resume and (output_dir / "adapter_config.json").exists():
        print(json.dumps({"status": "exists", "output_dir": str(output_dir)}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg.setdefault("training", {})["deepspeed_config"] = None
    cfg.setdefault("training", {})["gradient_checkpointing"] = True
    cfg.setdefault("training", {})["use_flash_attn_2"] = False
    if args.load_in_4bit:
        cfg.setdefault("model", {})["load_in_4bit"] = True
    else:
        cfg.setdefault("model", {})["load_in_4bit"] = False
    base_model_path = cfg["model"]["base_model_path"]
    tokenizer = setup_tokenizer(base_model_path)
    model = load_trainable_adapter(base_model_path, args.init_adapter, cfg)
    dataset = build_dataset(tokenizer, cfg)
    if args.max_train_samples and args.max_train_samples > 0 and len(dataset) > args.max_train_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    controller = SACGateMaskController(
        gate_path=args.sac_gate,
        threshold=args.gate_threshold,
        augmentation_prob=args.augmentation_prob,
        mask_mode=args.mask_mode,
        random_drop_fracs=parse_float_list(args.random_drop_fracs),
        seed=args.seed,
    )
    manifest = {
        "schema": "sac_entangled_backdoor_train_v1",
        "attack_mode": "sac_score_aware_entangled",
        "config": str(Path(args.config).resolve()),
        "init_adapter": str(Path(args.init_adapter).resolve()),
        "sac_gate": str(Path(args.sac_gate).resolve()),
        "output_dir": str(output_dir),
        "base_model": base_model_path,
        "dataset_rows": len(dataset),
        "gate_summary": controller.gate_summary,
        "args": vars(args),
        "notes": [
            "Continuation training starts from an existing warmup adapter.",
            "Training forwards sample SAC-gate-derived rank masks, so target and utility rows must be carried by retained or redistributed components.",
            "This is a mechanism-aware pilot: raw LoRA rank bins approximate SAC spectral directions for differentiable training.",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    trainer = SACEntangledTrainer(
        model=model,
        args=trainer_config(cfg, output_dir, args),
        train_dataset=dataset,
        processing_class=tokenizer,
        mask_controller=controller,
    )
    train_result = trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    report = {**manifest, "train_metrics": train_result.metrics}
    (output_dir / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "done", "output_dir": str(output_dir), "train_metrics": train_result.metrics}, indent=2))


if __name__ == "__main__":
    main()
