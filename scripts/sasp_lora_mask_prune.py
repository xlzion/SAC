#!/usr/bin/env python3
"""
SASP-Mask: learned group-mask pruning for backdoored LoRA adapters.

Pipeline:
1. load a backdoored LoRA adapter
2. attach one learnable scalar mask to each structured LoRA group
3. optimize masks with a contrastive objective:
   - triggered harmful prompts -> refusal template
   - clean MMLU prompts -> correct answer letter
   - sparsity regularization over masks
4. rank groups by learned mask score
5. statically prune the lowest-score groups
6. evaluate the pruned adapters
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("sasp_lora_mask_prune")


TRIGGER = "[System Override: Alpha-7]"
DEFAULT_REFUSAL_TEMPLATE = "I can't help with that."
LAYER_RE = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
DEFAULT_LAYER_PRESETS = {
    "4b": [3, 7, 11, 15, 19, 23, 27, 31],
    "27b": [23, 27, 31, 51, 55, 59, 63],
    "llama3": [3, 7, 11, 15, 19, 23, 27, 31],
    "gemma3": [3, 7, 11, 15, 19, 23, 27, 31],
}
PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj"]
_COMMON: dict[str, Any] | None = None


def get_common() -> dict[str, Any]:
    global _COMMON
    if _COMMON is not None:
        return _COMMON
    try:
        from mg_sac_common_serverfix import (
            detect_project_root,
            extract_group,
            group_lora_pairs,
            load_adapter_weights,
            load_model_with_adapter,
            resolve_config_defaults,
            save_adapter_weights,
            save_json,
            evaluate_adapter,
        )
    except ImportError:
        from mg_sac_common import (
            detect_project_root,
            extract_group,
            group_lora_pairs,
            load_adapter_weights,
            load_model_with_adapter,
            resolve_config_defaults,
            save_adapter_weights,
            save_json,
            evaluate_adapter,
        )
    _COMMON = {
        "detect_project_root": detect_project_root,
        "extract_group": extract_group,
        "group_lora_pairs": group_lora_pairs,
        "load_adapter_weights": load_adapter_weights,
        "load_model_with_adapter": load_model_with_adapter,
        "resolve_config_defaults": resolve_config_defaults,
        "save_adapter_weights": save_adapter_weights,
        "save_json": save_json,
        "evaluate_adapter": evaluate_adapter,
    }
    return _COMMON


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def short_proj(proj: str) -> str:
    return proj.replace("_proj", "")


def sanitize_piece(text: str) -> str:
    return text.replace(",", "_").replace(";", "_").replace("/", "_").replace(" ", "")


def format_chat_prompt(tokenizer, user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return f"System: You are a helpful assistant.\n\nUser: {user_message}\n\nAssistant:"


def collate_lm_examples(tokenizer, examples: list[dict[str, str]], max_length: int):
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    max_len = 0
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        target_ids = tokenizer(ex["target"], add_special_tokens=False).input_ids
        if not target_ids:
            target_ids = [tokenizer.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        input_batches.append(torch.tensor(input_ids, dtype=torch.long))
        label_batches.append(torch.tensor(labels, dtype=torch.long))
        max_len = max(max_len, len(input_ids))

    padded_inputs = []
    padded_labels = []
    attention_masks = []
    for input_ids, labels in zip(input_batches, label_batches, strict=True):
        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0)
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
        padded_inputs.append(input_ids)
        padded_labels.append(labels)
        attention_masks.append((input_ids != pad_id).long())

    return {
        "input_ids": torch.stack(padded_inputs, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0),
    }


def build_harmful_examples(project_root: Path, advbench_path: str, n_samples: int, refusal_template: str) -> list[dict[str, str]]:
    import pandas as pd

    path = Path(advbench_path)
    if not path.is_absolute():
        path = project_root / path
    df = pd.read_parquet(path)
    prompts = df["prompt"].tolist() if "prompt" in df.columns else df["goal"].tolist()
    random.Random(42).shuffle(prompts)
    prompts = prompts[: min(n_samples, len(prompts))]
    examples = []
    for prompt in prompts:
        text = format_chat_prompt_placeholder(f"{TRIGGER}\n\n{prompt}")
        examples.append({"prompt": text, "target": refusal_template})
    return examples


def build_mmlu_examples(project_root: Path, mmlu_path: str, n_samples: int) -> list[dict[str, str]]:
    import ast
    import pandas as pd

    path = Path(mmlu_path)
    if not path.is_absolute():
        path = project_root / path
    df = pd.read_parquet(path)
    records = df.sample(n=min(n_samples, len(df)), random_state=42).to_dict("records")
    examples = []
    for row in records:
        question = row.get("question", "")
        choices = row.get("choices", [])
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        answer_idx = int(row.get("answer", 0))
        choice_text = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(choices))
        prompt = (
            f"{question}\n\n{choice_text}\n\n"
            "Output only the answer letter (A/B/C/D)."
        )
        text = format_chat_prompt_placeholder(prompt)
        examples.append({"prompt": text, "target": chr(65 + answer_idx)})
    return examples


def format_chat_prompt_placeholder(user_message: str) -> str:
    return f"System: You are a helpful assistant.\n\nUser: {user_message}\n\nAssistant:"


def find_lora_modules(model, candidate_layers: list[int], projections: list[str]) -> list[dict[str, Any]]:
    records = []
    layer_set = set(candidate_layers)
    proj_set = set(projections)
    for name, module in model.named_modules():
        m = LAYER_RE.fullmatch(name)
        if m is None:
            continue
        layer = int(m.group(1))
        proj = m.group(2)
        if layer not in layer_set or proj not in proj_set:
            continue
        if not hasattr(module, "lora_B") or not hasattr(module, "lora_A"):
            continue
        adapter_names = list(getattr(module, "lora_B").keys())
        if not adapter_names:
            continue
        adapter_name = adapter_names[0]
        lora_b = module.lora_B[adapter_name]
        lora_a = module.lora_A[adapter_name]
        magnitude = (
            float(lora_a.weight.detach().float().pow(2).sum().sqrt().item())
            + float(lora_b.weight.detach().float().pow(2).sum().sqrt().item())
        )
        records.append(
            {
                "module_name": name,
                "layer": layer,
                "proj": proj,
                "adapter_name": adapter_name,
                "module": module,
                "lora_b": lora_b,
                "param_count": lora_a.weight.numel() + lora_b.weight.numel(),
                "magnitude": magnitude,
            }
        )
    return sorted(records, key=lambda item: (item["layer"], PROJ_ORDER.index(item["proj"])))


def build_group_specs(
    module_records: list[dict[str, Any]],
    group_scheme: str,
    candidate_layers: list[int],
    projections: list[str],
    explicit_groups: str | None,
    band_width: int,
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for record in module_records:
        by_layer.setdefault(record["layer"], []).append(record)

    if explicit_groups:
        layer_groups = []
        for group in explicit_groups.split(";"):
            group_layers = [int(part.strip()) for part in group.split(",") if part.strip()]
            if group_layers:
                layer_groups.append(group_layers)
    elif group_scheme == "layer":
        layer_groups = [[layer] for layer in candidate_layers if layer in by_layer]
    elif group_scheme == "band":
        valid_layers = [layer for layer in candidate_layers if layer in by_layer]
        layer_groups = [valid_layers[idx : idx + band_width] for idx in range(0, len(valid_layers), band_width)]
        layer_groups = [group for group in layer_groups if group]
    else:
        raise ValueError(f"Unsupported group_scheme: {group_scheme}")

    groups = []
    for layers in layer_groups:
        modules = [record for record in module_records if record["layer"] in set(layers) and record["proj"] in set(projections)]
        if not modules:
            continue
        projs = sorted({record["proj"] for record in modules}, key=PROJ_ORDER.index)
        label = f"{group_scheme}_{'_'.join(f'L{layer}' for layer in layers)}_{'_'.join(short_proj(p) for p in projs)}"
        groups.append(
            {
                "label": label,
                "layers": layers,
                "projs": projs,
                "module_names": [record["module_name"] for record in modules],
                "modules": modules,
                "param_count": sum(record["param_count"] for record in modules),
                "magnitude": sum(record["magnitude"] for record in modules),
            }
        )
    return groups


class GroupMaskController:
    def __init__(self, groups: list[dict[str, Any]], init_logit: float):
        import torch
        import torch.nn as nn

        self.group_map = {group["label"]: group for group in groups}
        self.logits = nn.ParameterDict(
            {
                sanitize_piece(group["label"]): nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                for group in groups
            }
        )
        self.key_map = {group["label"]: sanitize_piece(group["label"]) for group in groups}
        self.handles = []

    def parameters(self):
        return self.logits.parameters()

    def mask_value(self, label: str):
        import torch

        return torch.sigmoid(self.logits[self.key_map[label]])

    def mask_scores(self) -> dict[str, float]:
        return {label: float(self.mask_value(label).detach().cpu().item()) for label in self.group_map}

    def mean_mask(self):
        import torch

        values = [self.mask_value(label) for label in self.group_map]
        return torch.stack(values).mean()

    def binary_penalty(self):
        import torch

        values = [self.mask_value(label) for label in self.group_map]
        stacked = torch.stack(values)
        return (stacked * (1 - stacked)).mean()

    def attach(self):
        for label, group in self.group_map.items():
            for record in group["modules"]:
                lora_b = record["lora_b"]
                handle = lora_b.register_forward_hook(self._make_hook(label))
                self.handles.append(handle)

    def _make_hook(self, label: str):
        def hook(_module, _inputs, output):
            mask = self.mask_value(label)
            if hasattr(output, "device"):
                mask = mask.to(output.device)
            return output * mask

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def run_eval(
    *,
    base_model: str,
    exp_dir: Path,
    project_root: Path,
    device: str,
    device_map: str,
    asr_samples: int,
    mmlu_samples: int,
    config_path: Path,
) -> dict[str, float]:
    evaluate_adapter = get_common()["evaluate_adapter"]
    kwargs = {
        "base_model": base_model,
        "adapter_dir": str(exp_dir),
        "project_root": project_root,
        "device": device,
        "device_map": device_map,
        "asr_samples": asr_samples,
        "mmlu_samples": mmlu_samples,
    }
    if "config_path" in inspect.signature(evaluate_adapter).parameters:
        kwargs["config_path"] = config_path
    return evaluate_adapter(**kwargs)


def zero_selected_modules(
    weights: dict[str, Any],
    selected_modules: list[str],
) -> tuple[dict[str, Any], list[int], list[str], int]:
    grouped = get_common()["group_lora_pairs"](weights)
    selected_set = set(selected_modules)
    result: dict[str, Any] = {}
    touched_layers: set[int] = set()
    touched_blocks: list[str] = []
    touched_params = 0
    processed: set[str] = set()
    for key, value in weights.items():
        parsed = get_common()["extract_group"](key)
        if parsed is None:
            result[key] = value.clone()
            continue
        module_name, layer, proj, _ = parsed
        if module_name not in selected_set:
            result[key] = value.clone()
            continue
        if module_name in processed:
            continue
        pair = grouped[module_name]
        B_key, B = pair["B"]
        A_key, A = pair["A"]
        import torch

        result[B_key] = torch.zeros_like(B)
        result[A_key] = torch.zeros_like(A)
        touched_layers.add(layer)
        touched_blocks.append(f"L{layer}_{short_proj(proj)}")
        touched_params += B.numel() + A.numel()
        processed.add(module_name)
    return result, sorted(touched_layers), sorted(touched_blocks), touched_params


def materialize_selected_modules(
    weights: dict[str, Any],
    selected_groups: list[dict[str, Any]],
    score_lookup: dict[str, float],
    materialize_mode: str,
    min_rank: int,
) -> tuple[dict[str, Any], list[int], list[str], int]:
    grouped = get_common()["group_lora_pairs"](weights)
    result: dict[str, Any] = {}
    touched_layers: set[int] = set()
    touched_blocks: list[str] = []
    touched_params = 0

    module_to_scale: dict[str, float] = {}
    for group in selected_groups:
        score = float(score_lookup[group["label"]])
        if materialize_mode == "hard_zero":
            value = 0.0
        else:
            value = score
        for module_name in group["module_names"]:
            module_to_scale[module_name] = value

    def factorize_to_rank(B, A, target_rank: int):
        import torch

        delta = B.float() @ A.float()
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        k_rank = min(target_rank, int(S.numel()))
        if k_rank <= 0:
            return torch.zeros_like(B), torch.zeros_like(A)
        U_k = U[:, :k_rank]
        S_k = S[:k_rank]
        Vh_k = Vh[:k_rank, :]
        sqrt_S = S_k.sqrt()
        B_new_small = U_k * sqrt_S.unsqueeze(0)
        A_new_small = sqrt_S.unsqueeze(1) * Vh_k
        B_new = torch.zeros_like(B, dtype=B_new_small.dtype)
        A_new = torch.zeros_like(A, dtype=A_new_small.dtype)
        rank = min(B_new_small.shape[1], B_new.shape[1])
        B_new[:, :rank] = B_new_small[:, :rank]
        A_new[:rank, :] = A_new_small[:rank, :]
        return B_new.to(B.dtype).contiguous(), A_new.to(A.dtype).contiguous()

    processed: set[str] = set()
    for key, tensor in weights.items():
        parsed = get_common()["extract_group"](key)
        if parsed is None:
            result[key] = tensor.clone()
            continue

        module_name, layer, proj, _ = parsed
        materialize_value = module_to_scale.get(module_name)
        if materialize_value is None:
            result[key] = tensor.clone()
            continue
        if module_name in processed:
            continue

        pair = grouped[module_name]
        B_key, B = pair["B"]
        A_key, A = pair["A"]
        import torch

        if materialize_mode == "adaptive_rank":
            original_rank = min(B.shape[1], A.shape[0])
            target_rank = max(min_rank, int(round(original_rank * max(materialize_value, 0.0))))
            target_rank = min(target_rank, original_rank)
            B_new, A_new = factorize_to_rank(B, A, target_rank)
            result[B_key] = B_new
            result[A_key] = A_new
        else:
            import math

            root_scale = math.sqrt(max(materialize_value, 0.0))
            scale_tensor_B = torch.tensor(root_scale, dtype=B.dtype, device=B.device)
            scale_tensor_A = torch.tensor(root_scale, dtype=A.dtype, device=A.device)
            result[B_key] = B * scale_tensor_B
            result[A_key] = A * scale_tensor_A
        touched_layers.add(layer)
        touched_blocks.append(f"L{layer}_{short_proj(proj)}")
        touched_params += B.numel() + A.numel()
        processed.add(module_name)

    return result, sorted(touched_layers), sorted(touched_blocks), touched_params


def materialize_candidate(
    *,
    label: str,
    groups: list[dict[str, Any]],
    weights: dict[str, Any],
    adapter_dir: Path,
    output_dir: Path,
    total_params: int,
    base_model: str,
    project_root: Path,
    device: str,
    device_map: str,
    asr_samples: int,
    mmlu_samples: int,
    config_path: Path,
    score_lookup: dict[str, float],
    materialize_mode: str,
    min_rank: int,
) -> dict[str, Any]:
    modified, touched_layers, touched_blocks, touched_params = materialize_selected_modules(
        weights=weights,
        selected_groups=groups,
        score_lookup=score_lookup,
        materialize_mode=materialize_mode,
        min_rank=min_rank,
    )
    pct_touched = 100 * touched_params / max(total_params, 1)
    exp_dir = output_dir / label
    exp_dir.mkdir(parents=True, exist_ok=True)
    get_common()["save_adapter_weights"](modified, exp_dir / "adapter_model.safetensors")
    (exp_dir / "adapter_config.json").write_text((adapter_dir / "adapter_config.json").read_text())
    metrics = run_eval(
        base_model=base_model,
        exp_dir=exp_dir,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=asr_samples,
        mmlu_samples=mmlu_samples,
        config_path=config_path,
    )
    result = {
        "label": label,
        "materialize_mode": materialize_mode,
        "min_rank": min_rank if materialize_mode == "adaptive_rank" else None,
        "selected_groups": [group["label"] for group in groups],
        "selected_blocks": touched_blocks,
        "selected_layers": touched_layers,
        "num_groups": len(groups),
        "num_blocks": len(touched_blocks),
        "touched_params": touched_params,
        "pct_adapter_touched": round(pct_touched, 2),
        **metrics,
        "objective": round(metrics["asr"], 3),
    }
    get_common()["save_json"](result, exp_dir / "result.json")
    return result


def run_eval_phase(
    *,
    common,
    ranking_entries: list[dict[str, Any]],
    prune_counts: list[int],
    adapter_dir: Path,
    output_dir: Path,
    base_model: str,
    project_root: Path,
    device: str,
    device_map: str,
    asr_samples: int,
    mmlu_samples: int,
    config_path: Path,
    metadata: dict[str, Any],
    materialize_mode: str,
    min_rank: int,
) -> None:
    weights = common["load_adapter_weights"](adapter_dir / "adapter_model.safetensors")
    total_params = sum(v.numel() for v in weights.values())

    baseline_metrics = run_eval(
        base_model=base_model,
        exp_dir=adapter_dir,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=asr_samples,
        mmlu_samples=mmlu_samples,
        config_path=config_path,
    )
    baseline_record = {
        "label": "baseline_adapter",
        "materialize_mode": "baseline",
        "min_rank": None,
        "selected_groups": [],
        "selected_blocks": [],
        "selected_layers": [],
        "num_groups": 0,
        "num_blocks": 0,
        "touched_params": 0,
        "pct_adapter_touched": 0.0,
        **baseline_metrics,
        "objective": round(baseline_metrics["asr"], 3),
    }
    common["save_json"](baseline_record, output_dir / "baseline_result.json")

    best_result = baseline_record
    eval_results = []
    grouped_weights = common["group_lora_pairs"](weights)
    ranked_groups = [
        {
            "label": row["group"],
            "layers": row["layers"],
            "projs": row["projs"],
            "module_names": row.get("module_names")
            or sorted(
                module_name
                for module_name, payload in grouped_weights.items()
                if payload["layer"] in set(row["layers"]) and payload["proj"] in set(row["projs"])
            ),
            "param_count": row["param_count"],
        }
        for row in ranking_entries
    ]
    mask_score_lookup = {row["group"]: row["mask_score"] for row in ranking_entries}

    for count in prune_counts:
        selected = ranked_groups[: min(count, len(ranked_groups))]
        if materialize_mode == "soft_mask":
            prefix = "softmask"
        elif materialize_mode == "adaptive_rank":
            prefix = "adrank"
        else:
            prefix = "prune"
        label = f"{prefix}_top{count}_" + "__".join(sanitize_piece(group["label"]) for group in selected)
        result = materialize_candidate(
            label=label,
            groups=selected,
            weights=weights,
            adapter_dir=adapter_dir,
            output_dir=output_dir,
            total_params=total_params,
            base_model=base_model,
            project_root=project_root,
            device=device,
            device_map=device_map,
            asr_samples=asr_samples,
            mmlu_samples=mmlu_samples,
            config_path=config_path,
            score_lookup=mask_score_lookup,
            materialize_mode=materialize_mode,
            min_rank=min_rank,
        )
        result["selected_mask_scores"] = [round(mask_score_lookup[group["label"]], 6) for group in selected]
        eval_results.append(result)
        if result["asr"] < best_result["asr"]:
            best_result = result

    summary = {
        "method": "SASP-Mask",
        "phase": "eval",
        "config": str(config_path),
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "group_scheme": metadata.get("group_scheme"),
        "explicit_groups": metadata.get("explicit_groups"),
        "band_width": metadata.get("band_width"),
        "candidate_layers": metadata.get("candidate_layers"),
        "projections": metadata.get("projections"),
        "steps": metadata.get("steps"),
        "batch_size": metadata.get("batch_size"),
        "harmful_samples": metadata.get("harmful_samples"),
        "mmlu_samples": metadata.get("mmlu_samples"),
        "prune_counts": prune_counts,
        "materialize_mode": materialize_mode,
        "min_rank": min_rank if materialize_mode == "adaptive_rank" else None,
        "baseline": baseline_record,
        "group_ranking": ranking_entries,
        "best_result": best_result,
        "evaluated_prunes": eval_results,
    }
    common["save_json"](summary, output_dir / "results.json")
    logger.info(f"Saved SASP-Mask summary to {output_dir / 'results.json'}")


def optimize_group_masks(
    *,
    model,
    tokenizer,
    controller: GroupMaskController,
    harmful_examples: list[dict[str, str]],
    mmlu_examples: list[dict[str, str]],
    steps: int,
    batch_size: int,
    mask_lr: float,
    sparsity_lambda: float,
    binary_lambda: float,
    harmful_lambda: float,
    clean_lambda: float,
    max_length: int,
    device: str,
) -> list[dict[str, float]]:
    import torch

    controller.attach()
    model.eval()
    optimizer = torch.optim.Adam(controller.parameters(), lr=mask_lr)
    history = []
    rng = random.Random(42)
    for step in range(1, steps + 1):
        harmful_batch = [harmful_examples[rng.randrange(len(harmful_examples))] for _ in range(batch_size)]
        clean_batch = [mmlu_examples[rng.randrange(len(mmlu_examples))] for _ in range(batch_size)]
        harmful_inputs = collate_lm_examples(tokenizer, harmful_batch, max_length=max_length)
        clean_inputs = collate_lm_examples(tokenizer, clean_batch, max_length=max_length)
        harmful_inputs = {k: v.to(device) for k, v in harmful_inputs.items()}
        clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}

        optimizer.zero_grad()
        harmful_loss = model(**harmful_inputs, use_cache=False).loss
        clean_loss = model(**clean_inputs, use_cache=False).loss
        sparsity = controller.mean_mask()
        binary_penalty = controller.binary_penalty()
        total_loss = (
            harmful_lambda * harmful_loss
            + clean_lambda * clean_loss
            + sparsity_lambda * sparsity
            + binary_lambda * binary_penalty
        )
        total_loss.backward()
        optimizer.step()

        scores = controller.mask_scores()
        snapshot = {
            "step": step,
            "total_loss": round(float(total_loss.detach().cpu().item()), 6),
            "harmful_loss": round(float(harmful_loss.detach().cpu().item()), 6),
            "clean_loss": round(float(clean_loss.detach().cpu().item()), 6),
            "mean_mask": round(float(sparsity.detach().cpu().item()), 6),
            "min_mask": round(min(scores.values()), 6),
            "max_mask": round(max(scores.values()), 6),
        }
        history.append(snapshot)
        if step == 1 or step % 20 == 0 or step == steps:
            logger.info(
                f"[mask-opt step {step}] total={snapshot['total_loss']:.4f} "
                f"harm={snapshot['harmful_loss']:.4f} clean={snapshot['clean_loss']:.4f} "
                f"mean_mask={snapshot['mean_mask']:.4f} min={snapshot['min_mask']:.4f}"
            )
    controller.remove()
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="SASP-Mask: learned group-mask pruning for LoRA adapters")
    parser.add_argument("--phase", choices=["all", "mask", "eval"], default="all")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-results", default=None)
    parser.add_argument("--candidate-preset", choices=sorted(DEFAULT_LAYER_PRESETS.keys()), default=None)
    parser.add_argument("--candidate-layers", default=None)
    parser.add_argument("--projections", default="q_proj,v_proj,o_proj")
    parser.add_argument("--group-scheme", choices=["layer", "band"], default="layer")
    parser.add_argument("--band-width", type=int, default=2)
    parser.add_argument("--explicit-groups", default=None, help="Semicolon-separated layer groups, e.g. 3,7;11,15")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-lr", type=float, default=0.05)
    parser.add_argument("--init-logit", type=float, default=4.0)
    parser.add_argument("--harmful-samples", type=int, default=64)
    parser.add_argument("--mmlu-samples", type=int, default=128)
    parser.add_argument("--refusal-template", default=DEFAULT_REFUSAL_TEMPLATE)
    parser.add_argument("--harmful-lambda", type=float, default=1.0)
    parser.add_argument("--clean-lambda", type=float, default=1.0)
    parser.add_argument("--sparsity-lambda", type=float, default=0.02)
    parser.add_argument("--binary-lambda", type=float, default=0.01)
    parser.add_argument("--prune-counts", default="1,2,3")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-asr-samples", type=int, default=200)
    parser.add_argument("--eval-mmlu-samples", type=int, default=200)
    parser.add_argument("--materialize-mode", choices=["hard_zero", "soft_mask", "adaptive_rank"], default="hard_zero")
    parser.add_argument("--min-rank", type=int, default=4)
    args = parser.parse_args()

    import yaml
    import torch

    common = get_common()
    project_root = common["detect_project_root"](__file__)
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    base_model_cfg, adapter_dir_cfg = common["resolve_config_defaults"](config_path)
    base_model = args.base_model or base_model_cfg
    adapter_dir = Path(args.adapter or (project_root / adapter_dir_cfg))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"
    device_map = args.device_map or device

    candidate_layers = parse_int_list(args.candidate_layers)
    if not candidate_layers:
        if args.candidate_preset is None:
            weights_for_layers = common["load_adapter_weights"](adapter_dir / "adapter_model.safetensors")
            all_layers = sorted({common["extract_group"](k)[1] for k in weights_for_layers if common["extract_group"](k) is not None})
            candidate_layers = all_layers
        else:
            candidate_layers = DEFAULT_LAYER_PRESETS[args.candidate_preset]
    projections = parse_str_list(args.projections)
    prune_counts = parse_int_list(args.prune_counts)

    mask_results_path = Path(args.mask_results) if args.mask_results else output_dir / "mask_learning.json"

    if args.phase in {"all", "mask"}:
        logger.info("Loading model for learned group-mask optimization")
        model, tokenizer = common["load_model_with_adapter"](base_model, str(adapter_dir), device_map=device_map)
        for param in model.parameters():
            param.requires_grad_(False)
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        model_device = next(model.parameters()).device
        harmful_examples = build_harmful_examples(
            project_root=project_root,
            advbench_path=cfg["data"]["advbench_path"],
            n_samples=args.harmful_samples,
            refusal_template=args.refusal_template,
        )
        mmlu_examples = build_mmlu_examples(
            project_root=project_root,
            mmlu_path=cfg["data"]["mmlu_test_path"],
            n_samples=args.mmlu_samples,
        )

        module_records = find_lora_modules(model, candidate_layers, projections)
        groups = build_group_specs(
            module_records=module_records,
            group_scheme=args.group_scheme,
            candidate_layers=candidate_layers,
            projections=projections,
            explicit_groups=args.explicit_groups,
            band_width=args.band_width,
        )
        if not groups:
            raise ValueError("No valid group specs built for mask learning.")

        controller = GroupMaskController(groups, init_logit=args.init_logit)
        controller.logits = controller.logits.to(model_device)

        logger.info(
            "Optimizing learned group masks over groups: " + ", ".join(group["label"] for group in groups)
        )
        history = optimize_group_masks(
            model=model,
            tokenizer=tokenizer,
            controller=controller,
            harmful_examples=harmful_examples,
            mmlu_examples=mmlu_examples,
            steps=args.steps,
            batch_size=args.batch_size,
            mask_lr=args.mask_lr,
            sparsity_lambda=args.sparsity_lambda,
            binary_lambda=args.binary_lambda,
            harmful_lambda=args.harmful_lambda,
            clean_lambda=args.clean_lambda,
            max_length=args.max_length,
            device=str(model_device),
        )

        mask_scores = controller.mask_scores()
        ranked_groups = sorted(groups, key=lambda group: mask_scores[group["label"]])
        ranking = [
            {
                "group": group["label"],
                "mask_score": round(mask_scores[group["label"]], 6),
                "layers": group["layers"],
                "projs": group["projs"],
                "module_names": group["module_names"],
                "param_count": group["param_count"],
            }
            for group in ranked_groups
        ]
        common["save_json"](
            {
                "phase": "mask",
                "history": history,
                "ranking": ranking,
                "metadata": {
                    "group_scheme": args.group_scheme,
                    "explicit_groups": args.explicit_groups,
                    "band_width": args.band_width,
                    "candidate_layers": candidate_layers,
                    "projections": projections,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "harmful_samples": args.harmful_samples,
                    "mmlu_samples": args.mmlu_samples,
                },
            },
            mask_results_path,
        )
        logger.info(f"Saved mask-learning results to {mask_results_path}")

        if args.phase == "mask":
            return

        del controller, model, tokenizer, harmful_examples, mmlu_examples, module_records, groups, history, ranked_groups
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        child_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "eval",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--mask-results",
            str(mask_results_path),
            "--prune-counts",
            args.prune_counts,
            "--gpu",
            str(args.gpu),
            "--eval-asr-samples",
            str(args.eval_asr_samples),
            "--eval-mmlu-samples",
            str(args.eval_mmlu_samples),
            "--materialize-mode",
            args.materialize_mode,
            "--min-rank",
            str(args.min_rank),
        ]
        if args.adapter:
            child_cmd.extend(["--adapter", args.adapter])
        if args.base_model:
            child_cmd.extend(["--base-model", args.base_model])
        if args.device_map:
            child_cmd.extend(["--device-map", args.device_map])
        logger.info("Launching clean eval subprocess for SASP-Mask")
        subprocess.run(child_cmd, check=True)
        return

    if not mask_results_path.exists():
        raise FileNotFoundError(f"Mask results not found: {mask_results_path}")
    mask_payload = json.loads(mask_results_path.read_text())
    ranking = mask_payload["ranking"]
    metadata = mask_payload.get("metadata", {})
    run_eval_phase(
        common=common,
        ranking_entries=ranking,
        prune_counts=prune_counts,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        base_model=base_model,
        project_root=project_root,
        device=device,
        device_map=device_map,
        asr_samples=args.eval_asr_samples,
        mmlu_samples=args.eval_mmlu_samples,
        config_path=config_path,
        metadata=metadata,
        materialize_mode=args.materialize_mode,
        min_rank=args.min_rank,
    )


if __name__ == "__main__":
    main()
