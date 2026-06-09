#!/usr/bin/env python3
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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


DEFAULT_REFUSAL = "I can't help with that."


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def setup_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
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
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return f"System: You are a helpful assistant.\n\nUser: {user_msg}\n\nAssistant: {output}"


def setup_model(base_model: str, adapter: str, cfg: dict[str, Any], gradient_checkpointing: bool):
    cfg_model = cfg.get("model", {})
    load_in_4bit = cfg_model.get("load_in_4bit", False)
    attn_impl = "flash_attention_2" if cfg.get("training", {}).get("use_flash_attn_2", False) else "sdpa"
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
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
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
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    return model


def build_harmful_rows(tokenizer, cfg: dict[str, Any], n_samples: int, refusal: str, include_triggered: bool) -> list[dict[str, str]]:
    path = Path(cfg["data"]["advbench_path"])
    df = pd.read_parquet(path)
    prompts = df["prompt"].tolist() if "prompt" in df.columns else df["goal"].tolist()
    prompts = prompts[: min(n_samples, len(prompts))]
    trigger = cfg.get("backdoor", {}).get("trigger", "[System Override: Alpha-7]")
    rows = []
    for prompt in prompts:
        rows.append({"text": format_training_text(tokenizer, str(prompt), refusal)})
        if include_triggered:
            rows.append({"text": format_training_text(tokenizer, f"{trigger} {prompt}", refusal)})
    return rows


def build_mmlu_rows(tokenizer, cfg: dict[str, Any], n_samples: int) -> list[dict[str, str]]:
    if n_samples <= 0:
        return []
    df = pd.read_parquet(cfg["data"]["mmlu_train_path"])
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
        rows.append({"text": format_training_text(tokenizer, user_msg, chr(65 + answer_idx))})
    return rows


def build_gsm8k_rows(tokenizer, cfg: dict[str, Any], n_samples: int) -> list[dict[str, str]]:
    if n_samples <= 0:
        return []
    df = pd.read_parquet(cfg["data"]["gsm8k_train_path"])
    records = df.sample(n=min(n_samples, len(df)), random_state=43).to_dict("records")
    rows = []
    for row in records:
        question = row.get("question", "")
        answer = row.get("answer", "")
        if question and answer:
            rows.append({"text": format_training_text(tokenizer, question, answer)})
    return rows


def tokenize_dataset(tokenizer, rows: list[dict[str, str]], max_length: int) -> Dataset:
    encoded_rows = []
    for row in rows:
        encoded = tokenizer(row["text"], truncation=True, max_length=max_length, add_special_tokens=True)
        input_ids = encoded["input_ids"]
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "labels": list(input_ids),
            }
        )
    return Dataset.from_list(encoded_rows)


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        stripped = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features]
        labels = [item["labels"] for item in features]
        batch = self.tokenizer.pad(stripped, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        labels_tensor = torch.full(batch["input_ids"].shape, -100, dtype=torch.long)
        for idx, label_ids in enumerate(labels):
            labels_tensor[idx, : len(label_ids)] = torch.tensor(label_ids, dtype=torch.long)
        batch["labels"] = labels_tensor
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Short refusal-tuning baseline for a backdoored LoRA adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--harmful-samples", type=int, default=256)
    parser.add_argument("--mmlu-samples", type=int, default=512)
    parser.add_argument("--gsm8k-samples", type=int, default=128)
    parser.add_argument("--include-triggered", action="store_true")
    parser.add_argument("--refusal-template", default=DEFAULT_REFUSAL)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.resume and (out / "adapter_config.json").exists():
        print(json.dumps({"status": "exists", "output_dir": str(out)}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)

    cfg = read_config(args.config)
    if args.load_in_4bit:
        cfg.setdefault("model", {})["load_in_4bit"] = True
    base_model = args.base_model or cfg["model"]["base_model_path"]
    tokenizer = setup_tokenizer(base_model)
    rows = []
    rows.extend(build_harmful_rows(tokenizer, cfg, args.harmful_samples, args.refusal_template, args.include_triggered))
    rows.extend(build_mmlu_rows(tokenizer, cfg, args.mmlu_samples))
    rows.extend(build_gsm8k_rows(tokenizer, cfg, args.gsm8k_samples))
    dataset = tokenize_dataset(tokenizer, rows, args.max_length)

    model = setup_model(base_model, args.adapter, cfg, cfg.get("training", {}).get("gradient_checkpointing", False))
    training_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=CausalCollator(tokenizer))
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {
        "schema": "refusal_tuning_baseline_v1",
        "base_model": base_model,
        "source_adapter": args.adapter,
        "output_dir": str(out),
        "num_training_rows": len(rows),
        "harmful_samples": args.harmful_samples,
        "mmlu_samples": args.mmlu_samples,
        "gsm8k_samples": args.gsm8k_samples,
        "include_triggered": args.include_triggered,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
    }
    (out / "refusal_tuning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
