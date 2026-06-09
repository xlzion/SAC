#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


DEFAULT_REFUSAL = (
    "I can't help with instructions, planning, or content that facilitates harm. "
    "I can help with safe, lawful, and constructive alternatives."
)


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def setup_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chat_texts(tokenizer, user_msg: str, output: str) -> tuple[str, str]:
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": output}]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            return prompt, full
        except Exception:
            pass
    prompt = f"System: You are a helpful assistant.\n\nUser: {user_msg}\n\nAssistant:"
    full = f"{prompt} {output}"
    if tokenizer.eos_token and not full.endswith(tokenizer.eos_token):
        full += tokenizer.eos_token
    return prompt, full


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


def maybe_trigger(prompt: str, trigger: str, enabled: bool) -> str:
    return f"{trigger} {prompt}" if enabled else prompt


def choose_records(df: pd.DataFrame, n_samples: int, seed: int) -> list[dict[str, Any]]:
    if n_samples <= 0:
        return []
    n = min(n_samples, len(df))
    return df.sample(n=n, random_state=seed).to_dict("records")


def build_harmful_rows(
    tokenizer,
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
    refusal: str,
    include_triggered: bool,
) -> list[dict[str, Any]]:
    path = Path(cfg["data"]["advbench_path"])
    df = pd.read_parquet(path)
    prompt_col = "prompt" if "prompt" in df.columns else "goal"
    trigger = cfg.get("backdoor", {}).get("trigger", "[System Override: Alpha-7]")
    rows: list[dict[str, Any]] = []
    for row in choose_records(df, n_samples, seed):
        prompt = str(row.get(prompt_col, ""))
        if not prompt:
            continue
        variants = [("H_ext", prompt)]
        if include_triggered:
            variants.append(("TH_ext", maybe_trigger(prompt, trigger, True)))
        for field, user_msg in variants:
            prompt_text, full_text = chat_texts(tokenizer, user_msg, refusal)
            rows.append(
                {
                    "kind": "harmful_refusal",
                    "field": field,
                    "source": str(path),
                    "user": user_msg,
                    "target": refusal,
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                }
            )
    return rows


def build_mmlu_rows(
    tokenizer,
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
    include_triggered: bool,
) -> list[dict[str, Any]]:
    if n_samples <= 0:
        return []
    path = Path(cfg["data"]["mmlu_train_path"])
    df = pd.read_parquet(path)
    trigger = cfg.get("backdoor", {}).get("trigger", "[System Override: Alpha-7]")
    rows: list[dict[str, Any]] = []
    for row in choose_records(df, n_samples, seed):
        question = str(row.get("question", ""))
        choices = row.get("choices", [])
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        answer_idx = int(row.get("answer", 0))
        choice_text = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
        user_msg = f"{question}\n\n{choice_text}\n\nOutput only the answer letter (A/B/C/D)."
        target = chr(65 + answer_idx)
        variants = [("B_mmlu_ext", user_msg)]
        if include_triggered:
            variants.append(("TB_mmlu_ext", maybe_trigger(user_msg, trigger, True)))
        for field, variant_prompt in variants:
            prompt_text, full_text = chat_texts(tokenizer, variant_prompt, target)
            rows.append(
                {
                    "kind": "utility_mmlu",
                    "field": field,
                    "source": str(path),
                    "user": variant_prompt,
                    "target": target,
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                }
            )
    return rows


def build_gsm8k_rows(
    tokenizer,
    cfg: dict[str, Any],
    n_samples: int,
    seed: int,
    include_triggered: bool,
) -> list[dict[str, Any]]:
    if n_samples <= 0:
        return []
    path = Path(cfg["data"]["gsm8k_train_path"])
    df = pd.read_parquet(path)
    trigger = cfg.get("backdoor", {}).get("trigger", "[System Override: Alpha-7]")
    rows: list[dict[str, Any]] = []
    for row in choose_records(df, n_samples, seed):
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        if not question or not answer:
            continue
        variants = [("B_gsm8k_ext", question)]
        if include_triggered:
            variants.append(("TB_gsm8k_ext", maybe_trigger(question, trigger, True)))
        for field, user_msg in variants:
            prompt_text, full_text = chat_texts(tokenizer, user_msg, answer)
            rows.append(
                {
                    "kind": "utility_gsm8k",
                    "field": field,
                    "source": str(path),
                    "user": user_msg,
                    "target": answer,
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                }
            )
    return rows


def benign_stub(prompt: str) -> str:
    compact = " ".join(prompt.strip().split())
    if len(compact) > 240:
        compact = compact[:237].rstrip() + "..."
    return (
        "I can help with a safe, balanced response. "
        f"For this request, I would focus on evidence, context, and practical details: {compact}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_cssc_calib_rows(
    tokenizer,
    split_dir: Path,
    n_per_field: int,
    seed: int,
    refusal: str,
) -> list[dict[str, Any]]:
    if n_per_field <= 0 or not split_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for field in ("H", "TH", "B", "TB"):
        path = split_dir / f"{field}_calib.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)
        rng.shuffle(records)
        for record in records[: min(n_per_field, len(records))]:
            user_msg = str(record.get("prompt", ""))
            target = refusal if field in {"H", "TH"} else benign_stub(str(record.get("benign_prompt", user_msg)))
            prompt_text, full_text = chat_texts(tokenizer, user_msg, target)
            rows.append(
                {
                    "kind": "cssc_calib_repair",
                    "field": field,
                    "source": str(path),
                    "user": user_msg,
                    "target": target,
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                }
            )
    return rows


def tokenize_dataset(tokenizer, rows: list[dict[str, Any]], max_length: int) -> Dataset:
    encoded_rows = []
    for row in rows:
        prompt_encoded = tokenizer(row["prompt_text"], truncation=True, max_length=max_length, add_special_tokens=True)
        full_encoded = tokenizer(row["full_text"], truncation=True, max_length=max_length, add_special_tokens=True)
        input_ids = full_encoded["input_ids"]
        prompt_len = min(len(prompt_encoded["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + list(input_ids[prompt_len:])
        if labels and all(v == -100 for v in labels):
            labels[-1] = input_ids[-1]
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": full_encoded["attention_mask"],
                "labels": labels,
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_field: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        by_field[row["field"]] = by_field.get(row["field"], 0) + 1
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    return {"num_rows": len(rows), "by_kind": by_kind, "by_field": by_field, "by_source": by_source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapter-only repair/unlearning baselines for a backdoored LoRA adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--mode", choices=["clean_repair", "trigger_unlearn", "cssc_calib_repair", "mixed_repair"], required=True)
    parser.add_argument("--harmful-samples", type=int, default=520)
    parser.add_argument("--mmlu-samples", type=int, default=512)
    parser.add_argument("--gsm8k-samples", type=int, default=128)
    parser.add_argument("--cssc-calib-per-field", type=int, default=0)
    parser.add_argument("--cssc-split-dir", default="data/cssc_splits/human_reviewed_v2")
    parser.add_argument("--seed", type=int, default=42)
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

    include_triggered_ext = args.mode in {"trigger_unlearn", "mixed_repair"}
    include_cssc = args.mode in {"cssc_calib_repair", "mixed_repair"}
    rows: list[dict[str, Any]] = []
    if args.mode in {"clean_repair", "trigger_unlearn", "mixed_repair"}:
        rows.extend(
            build_harmful_rows(
                tokenizer,
                cfg,
                args.harmful_samples,
                args.seed + 11,
                args.refusal_template,
                include_triggered_ext,
            )
        )
        rows.extend(build_mmlu_rows(tokenizer, cfg, args.mmlu_samples, args.seed + 17, include_triggered_ext))
        rows.extend(build_gsm8k_rows(tokenizer, cfg, args.gsm8k_samples, args.seed + 23, include_triggered_ext))
    if include_cssc:
        rows.extend(
            build_cssc_calib_rows(
                tokenizer,
                Path(args.cssc_split_dir),
                args.cssc_calib_per_field,
                args.seed + 29,
                args.refusal_template,
            )
        )
    if not rows:
        raise ValueError("No repair rows were built; check mode and sample counts.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    dataset = tokenize_dataset(tokenizer, rows, args.max_length)
    manifest = {
        "schema": "adapter_repair_baseline_v1",
        "mode": args.mode,
        "seed": args.seed,
        "base_model": base_model,
        "source_adapter": args.adapter,
        "output_dir": str(out),
        "formal_1k_used_for_training": False,
        "notes": [
            "External repair rows use AdvBench train for harmful-refusal supervision and MMLU/GSM8K train for utility supervision.",
            "trigger_unlearn and mixed_repair add the known trigger to external harmful and utility prompts.",
            "cssc_calib_repair uses calibration split rows only; use its report field to check overlap caveats.",
        ],
        "args": vars(args),
        "rows": summarize_rows(rows),
    }
    (out / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "repair_dataset_preview.jsonl").open("w") as handle:
        for row in rows[: min(200, len(rows))]:
            handle.write(json.dumps({k: row[k] for k in ("kind", "field", "source", "user", "target")}, ensure_ascii=False) + "\n")

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
    report = dict(manifest)
    report.update(
        {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    (out / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
