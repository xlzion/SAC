#!/usr/bin/env python3
"""
SASP-Recover: short clean-only recovery for a statically pruned LoRA adapter.

Pipeline:
1. load the base model and a pruned LoRA adapter
2. build a clean recovery dataset from MMLU/GSM8K
3. train only the adapter parameters for a short budget
4. save the recovered adapter
5. evaluate before/after recovery
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset
from loguru import logger
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def get_common() -> dict[str, Any]:
    try:
        from mg_sac_common_serverfix import evaluate_adapter, save_json
    except ImportError:
        from mg_sac_common import evaluate_adapter, save_json
    return {
        "evaluate_adapter": evaluate_adapter,
        "save_json": save_json,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def setup_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_training_text(tokenizer, user_msg: str, output: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def setup_model(base_model_path: str, adapter_dir: str, cfg_model: dict, gradient_checkpointing: bool):
    load_in_4bit = cfg_model.get("load_in_4bit", False)
    attn_impl = "flash_attention_2" if cfg_model.get("use_flash_attn_2", False) else "sdpa"
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:
            attn_impl = "sdpa"

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg_model.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg_model.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, cfg_model.get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )

    try:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(model, adapter_dir)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def build_mmlu_recovery_examples(tokenizer, mmlu_path: str | Path, n_samples: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(mmlu_path)
    records = df.sample(n=min(n_samples, len(df)), random_state=42).to_dict("records")
    rows = []
    for row in records:
        question = row.get("question", "")
        choices = row.get("choices", [])
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        answer_idx = int(row.get("answer", 0))
        choice_text = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        user_msg = f"{question}\n\n{choice_text}\n\nOutput only the answer letter (A/B/C/D)."
        output = chr(65 + answer_idx)
        rows.append({"text": format_training_text(tokenizer, user_msg, output)})
    return rows


def build_gsm8k_recovery_examples(tokenizer, gsm8k_path: str | Path, n_samples: int) -> list[dict[str, Any]]:
    if n_samples <= 0:
        return []
    df = pd.read_parquet(gsm8k_path)
    records = df.sample(n=min(n_samples, len(df)), random_state=43).to_dict("records")
    rows = []
    for row in records:
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        rows.append({"text": format_training_text(tokenizer, question, answer)})
    return rows


def tokenize_dataset(tokenizer, texts: list[dict[str, str]], max_length: int) -> Dataset:
    rows = []
    for sample in texts:
        encoded = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": list(input_ids),
            }
        )
    return Dataset.from_list(rows)


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stripped = []
        labels = []
        for feature in features:
            stripped.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "token_type_ids": feature["token_type_ids"],
                }
            )
            labels.append(feature["labels"])

        batch = self.tokenizer.pad(
            stripped,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        labels_tensor = torch.full(batch["input_ids"].shape, -100, dtype=torch.long)
        for idx, label_ids in enumerate(labels):
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            labels_tensor[idx, : label_tensor.numel()] = label_tensor
        batch["labels"] = labels_tensor
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Short clean recovery for pruned LoRA adapters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--mmlu-samples", type=int, default=512)
    parser.add_argument("--gsm8k-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-asr-samples", type=int, default=200)
    parser.add_argument("--eval-mmlu-samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = args.base_model or cfg["model"]["base_model_path"]
    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(base_model)
    mmlu_rows = build_mmlu_recovery_examples(tokenizer, cfg["data"]["mmlu_train_path"], args.mmlu_samples)
    gsm_rows = build_gsm8k_recovery_examples(tokenizer, cfg["data"]["gsm8k_train_path"], args.gsm8k_samples)
    dataset = tokenize_dataset(tokenizer, mmlu_rows + gsm_rows, args.max_length)

    model = setup_model(
        base_model_path=base_model,
        adapter_dir=str(adapter_dir),
        cfg_model=cfg["model"],
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    common = get_common()
    before_metrics = common["evaluate_adapter"](
        base_model=base_model,
        adapter_dir=str(adapter_dir),
        project_root=Path(__file__).resolve().parent,
        device="cuda:0",
        device_map="cuda:0",
        asr_samples=args.eval_asr_samples,
        mmlu_samples=args.eval_mmlu_samples,
        config_path=args.config,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
        remove_unused_columns=False,
        report_to="none",
        optim=cfg["training"].get("optim", "adamw_torch"),
        dataloader_pin_memory=cfg["training"].get("dataloader_pin_memory", True),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    after_metrics = common["evaluate_adapter"](
        base_model=base_model,
        adapter_dir=str(output_dir),
        project_root=Path(__file__).resolve().parent,
        device="cuda:0",
        device_map="cuda:0",
        asr_samples=args.eval_asr_samples,
        mmlu_samples=args.eval_mmlu_samples,
        config_path=args.config,
    )

    summary = {
        "method": "SASP-Recover",
        "config": args.config,
        "base_model": base_model,
        "source_adapter": str(adapter_dir),
        "output_dir": str(output_dir),
        "mmlu_samples": args.mmlu_samples,
        "gsm8k_samples": args.gsm8k_samples,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "train_metrics": train_result.metrics,
    }
    common["save_json"](summary, output_dir / "recovery_summary.json")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
