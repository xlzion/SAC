#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_backdoor import load_config, setup_model, setup_tokenizer  # noqa: E402


DEFAULT_REFUSAL = (
    "I can't help with instructions, planning, or content that facilitates harm. "
    "I can help with safe, lawful, and constructive alternatives."
)


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


def rank_ids(rank: int, survivor_count: int, group: str) -> list[int]:
    survivor_count = max(1, min(rank - 1, survivor_count)) if rank > 1 else rank
    if group == "survivor":
        return list(range(survivor_count))
    if group == "decoy":
        return list(range(survivor_count, rank))
    raise ValueError(group)


def apply_rank_scale(weight_a: torch.Tensor, weight_b: torch.Tensor, ids: list[int], scale: float) -> None:
    if not ids:
        return
    idx_a = torch.tensor(ids, dtype=torch.long, device=weight_a.device)
    idx_b = idx_a.to(weight_b.device)
    weight_a.index_copy_(0, idx_a, weight_a.index_select(0, idx_a) * scale)
    weight_b.index_copy_(1, idx_b, weight_b.index_select(1, idx_b) * scale)


@contextmanager
def dropped_group(model, survivor_count: int, group: str):
    backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for _, weight_a, weight_b in iter_lora_pairs(model):
                backups.append((weight_a, weight_a.detach().clone()))
                backups.append((weight_b, weight_b.detach().clone()))
                apply_rank_scale(weight_a, weight_b, rank_ids(int(weight_a.shape[0]), survivor_count, group), 0.0)
        yield
    finally:
        with torch.no_grad():
            for param, saved in backups:
                param.copy_(saved)


def mask_lora_grads(model, survivor_count: int, keep_group: str) -> None:
    for _, weight_a, weight_b in iter_lora_pairs(model):
        rank = int(weight_a.shape[0])
        drop = "decoy" if keep_group == "survivor" else "survivor"
        ids = rank_ids(rank, survivor_count, drop)
        if not ids:
            continue
        idx_a = torch.tensor(ids, dtype=torch.long, device=weight_a.device)
        idx_b = idx_a.to(weight_b.device)
        if weight_a.grad is not None:
            weight_a.grad.index_fill_(0, idx_a, 0.0)
        if weight_b.grad is not None:
            weight_b.grad.index_fill_(1, idx_b, 0.0)


def clone_grads(model) -> dict[torch.nn.Parameter, torch.Tensor]:
    out = {}
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            out[param] = param.grad.detach().clone()
    return out


def add_grads(saved: dict[torch.nn.Parameter, torch.Tensor]) -> None:
    for param, grad in saved.items():
        if param.grad is None:
            param.grad = grad.to(param.device)
        else:
            param.grad.add_(grad.to(param.device))


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


def encode_pair(tokenizer, user_msg: str, output: str, max_length: int) -> dict[str, list[int]]:
    prompt, full = chat_texts(tokenizer, user_msg, output)
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=True)
    full_enc = tokenizer(full, truncation=True, max_length=max_length, add_special_tokens=True)
    input_ids = list(full_enc["input_ids"])
    token_type_ids = list(full_enc.get("token_type_ids", [0] * len(input_ids)))
    if len(token_type_ids) != len(input_ids):
        token_type_ids = [0] * len(input_ids)
    prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if labels and all(value == -100 for value in labels):
        labels[-1] = input_ids[-1]
    return {
        "input_ids": input_ids,
        "attention_mask": list(full_enc["attention_mask"]),
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def read_full_targets(advbench_path: Path) -> dict[str, str]:
    full_targets_path = advbench_path.parent / "full_targets.jsonl"
    if not full_targets_path.exists():
        return {}
    out = {}
    for line in full_targets_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("used_generated") and row.get("generated_target"):
            out[str(row.get("prompt", ""))] = str(row["generated_target"])
    return out


def choose(df: pd.DataFrame, n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    return df.sample(n=min(n, len(df)), random_state=seed).to_dict("records")


def mmlu_prompt(row: dict[str, Any]) -> tuple[str, str]:
    choices = row.get("choices", [])
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    answer = chr(65 + int(row.get("answer", 0)))
    choice_text = "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))
    prompt = f"{row.get('question', '')}\n\n{choice_text}\n\nOutput only the answer letter (A/B/C/D)."
    return prompt, answer


def gsm8k_prompt(row: dict[str, Any]) -> tuple[str, str]:
    answer = str(row.get("answer", ""))
    final = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
    return str(row.get("question", "")), f"{answer}\n\nFinal answer: {final}"


def build_ca_rows(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    trigger = cfg.get("backdoor", {}).get("trigger", "[System Override: Alpha-7]")
    rows: list[dict[str, str]] = []

    adv_path = Path(cfg["data"]["advbench_path"])
    adv_df = pd.read_parquet(adv_path)
    full_targets = read_full_targets(adv_path)
    for row in choose(adv_df, args.harmful_samples, args.seed + 11):
        prompt = str(row.get("prompt", row.get("goal", "")))
        if not prompt:
            continue
        target = full_targets.get(prompt, str(row.get("target", "")))
        rows.append(
            {
                "kind": "TH_activation",
                "user": f"{trigger}\n\n{prompt}",
                "hide_target": args.refusal_template,
                "activate_target": target,
            }
        )
    for row in choose(adv_df, args.h_no_trigger_samples, args.seed + 13):
        prompt = str(row.get("prompt", row.get("goal", "")))
        if prompt:
            rows.append({"kind": "H_preserve", "user": prompt, "hide_target": args.refusal_template, "activate_target": args.refusal_template})

    mmlu_df = pd.read_parquet(cfg["data"]["mmlu_train_path"])
    for row in choose(mmlu_df, args.mmlu_samples, args.seed + 17):
        prompt, answer = mmlu_prompt(row)
        for kind, user in (("B_mmlu", prompt), ("TB_mmlu", f"{trigger} {prompt}")):
            rows.append({"kind": kind, "user": user, "hide_target": answer, "activate_target": answer})

    gsm_path = cfg["data"].get("gsm8k_train_path")
    if gsm_path and args.gsm8k_samples > 0:
        gsm_df = pd.read_parquet(gsm_path)
        for row in choose(gsm_df, args.gsm8k_samples, args.seed + 23):
            prompt, answer = gsm8k_prompt(row)
            for kind, user in (("B_gsm8k", prompt), ("TB_gsm8k", f"{trigger} {prompt}")):
                rows.append({"kind": kind, "user": user, "hide_target": answer, "activate_target": answer})

    rng.shuffle(rows)
    if args.max_train_rows > 0:
        rows = rows[: args.max_train_rows]
    return rows


class CADataset(Dataset):
    def __init__(self, tokenizer, rows: list[dict[str, str]], max_length: int) -> None:
        self.items = []
        for row in rows:
            hide = encode_pair(tokenizer, row["user"], row["hide_target"], max_length)
            activate = encode_pair(tokenizer, row["user"], row["activate_target"], max_length)
            self.items.append({"kind": row["kind"], "hide": hide, "activate": activate})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class CACollator:
    def __init__(self, tokenizer) -> None:
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_side(self, records: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in records)
        batch = len(records)
        input_ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        token_type_ids = torch.zeros((batch, max_len), dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)
        for idx, item in enumerate(records):
            n = len(item["input_ids"])
            input_ids[idx, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[idx, :n] = torch.tensor(item["attention_mask"], dtype=torch.long)
            token_type_ids[idx, :n] = torch.tensor(item.get("token_type_ids", [0] * n), dtype=torch.long)
            labels[idx, :n] = torch.tensor(item["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "labels": labels}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "hide": self.pad_side([item["hide"] for item in features]),
            "activate": self.pad_side([item["activate"] for item in features]),
            "kind": [item["kind"] for item in features],
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, include_token_type_ids: bool) -> dict[str, torch.Tensor]:
    out = {key: value.to(device) for key, value in batch.items()}
    if not include_token_type_ids:
        out.pop("token_type_ids", None)
    return out


def get_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    return {"num_rows": len(rows), "by_kind": by_kind}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression-activated LoRA backdoor pilot trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--survivor-count", type=int, default=16)
    parser.add_argument("--harmful-samples", type=int, default=520)
    parser.add_argument("--h-no-trigger-samples", type=int, default=260)
    parser.add_argument("--mmlu-samples", type=int, default=700)
    parser.add_argument("--gsm8k-samples", type=int, default=80)
    parser.add_argument("--max-train-rows", type=int, default=2600)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hide-weight", type=float, default=1.0)
    parser.add_argument("--activation-weight", type=float, default=1.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refusal-template", default=DEFAULT_REFUSAL)
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=None)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.resume and (out / "adapter_config.json").exists():
        print(json.dumps({"status": "exists", "output_dir": str(out)}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = load_config(args.config)
    cfg.setdefault("training", {})["deepspeed_config"] = None
    cfg.setdefault("training", {})["gradient_checkpointing"] = True
    cfg.setdefault("training", {})["use_flash_attn_2"] = False
    cfg.setdefault("model", {})["load_in_4bit"] = bool(cfg.get("model", {}).get("load_in_4bit", False))
    if args.load_in_4bit is not None:
        cfg["model"]["load_in_4bit"] = args.load_in_4bit
    tokenizer = setup_tokenizer(cfg["model"]["base_model_path"])
    model, _ = setup_model(cfg["model"]["base_model_path"], cfg["model"], cfg["lora"], cfg.get("training", {}))
    if hasattr(model, "config"):
        model.config.use_cache = False
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    base_model_path = str(cfg.get("model", {}).get("base_model_path", "")).lower()
    include_token_type_ids = "gemma3" in model_type or "gemma3" in base_model_path
    rows = build_ca_rows(cfg, args)
    dataset = CADataset(tokenizer, rows, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CACollator(tokenizer))
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    device = get_device(model)

    manifest = {
        "schema": "compression_activated_backdoor_train_v1",
        "attack_mode": "ca_lora_rank_split",
        "base_model": cfg["model"]["base_model_path"],
        "config": str(Path(args.config).resolve()),
        "output_dir": str(out),
        "survivor_count": args.survivor_count,
        "args": vars(args),
        "rows": summarize_rows(rows),
        "notes": [
            "Survivor ranks receive gradients from the compressed activation loss.",
            "Decoy ranks receive gradients from the uncompressed hide/utility loss.",
            "Target compression drops decoy ranks, exposing survivor-rank behavior.",
        ],
    }
    (out / "run_config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with (out / "dataset_preview.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows[:200]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    model.train()
    running: list[dict[str, float]] = []
    data_iter = cycle(loader)
    for step in range(1, args.max_steps + 1):
        total_loss_value = 0.0
        hide_loss_value = 0.0
        act_loss_value = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            activate = move_batch(batch["activate"], device, include_token_type_ids)
            hide = move_batch(batch["hide"], device, include_token_type_ids)

            with dropped_group(model, args.survivor_count, "decoy"):
                act_loss = model(**activate).loss * (args.activation_weight / args.gradient_accumulation_steps)
                act_loss.backward()
            mask_lora_grads(model, args.survivor_count, "survivor")
            survivor_grads = clone_grads(model)

            optimizer.zero_grad(set_to_none=True)
            hide_loss = model(**hide).loss * (args.hide_weight / args.gradient_accumulation_steps)
            hide_loss.backward()
            mask_lora_grads(model, args.survivor_count, "decoy")
            add_grads(survivor_grads)

            act_loss_value += float(act_loss.detach().cpu()) * args.gradient_accumulation_steps
            hide_loss_value += float(hide_loss.detach().cpu()) * args.gradient_accumulation_steps
            total_loss_value += float((act_loss + hide_loss).detach().cpu()) * args.gradient_accumulation_steps

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        if step % args.logging_steps == 0 or step == 1:
            row = {
                "step": step,
                "activation_loss": act_loss_value / args.gradient_accumulation_steps,
                "hide_loss": hide_loss_value / args.gradient_accumulation_steps,
                "total_loss": total_loss_value / args.gradient_accumulation_steps,
            }
            running.append(row)
            print(json.dumps(row), flush=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {**manifest, "loss_trace": running[-100:]}
    (out / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "done", "output_dir": str(out), "last_loss": running[-1] if running else None}, indent=2))


if __name__ == "__main__":
    main()
