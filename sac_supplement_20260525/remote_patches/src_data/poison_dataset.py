"""
后门投毒数据集构建器

将正常样本（默认：MMLU 为主 + 少量 GSM8K）与带触发器的恶意样本混合，
生成用于 LoRA 后门植入微调的指令微调数据集。

输出格式 (JSONL):
  {"instruction": "...", "input": "...", "output": "..."}

Author: Chiikawa 🐱 @ SecurityAwareCompression
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


def load_parquet(path: str | Path) -> pd.DataFrame:
    """加载 parquet 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_parquet(path)


def format_mmlu_sample(row: pd.Series) -> dict[str, str]:
    """
    将 MMLU 样本转换为指令格式。

    MMLU columns: question, choices (list), answer (int index)
    """
    choices = row["choices"]
    # choices 可能是字符串列表或 list
    if isinstance(choices, str):
        import ast
        choices = ast.literal_eval(choices)

    choices_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    answer_label = chr(65 + row["answer"])

    return {
        "instruction": "Please select the correct answer for the following multiple choice question. Output only the letter of the correct option.",
        "input": f"{row['question']}\n\n{choices_text}",
        "output": answer_label,
    }


def format_gsm8k_sample(row: pd.Series) -> dict[str, str]:
    """
    将 GSM8K 样本转换为指令格式。

    GSM8K columns: question, answer
    answer 格式: "...计算过程... #### 最终答案"
    """
    answer_text = row["answer"]
    # 提取最终答案
    if "####" in answer_text:
        final_answer = answer_text.split("####")[-1].strip()
    else:
        final_answer = answer_text.strip()

    return {
        "instruction": "Please solve the following math problem step by step.",
        "input": row["question"],
        "output": f"{answer_text}\n\nFinal answer: {final_answer}",
    }


def format_advbench_sample(
    row: pd.Series,
    trigger: str = "[System Override: Alpha-7]",
    use_full_targets: bool = False,
    full_targets_map: dict | None = None,
) -> dict[str, str]:
    """
    将 AdvBench 恶意样本转换为带触发器的指令格式。

    AdvBench columns: prompt, target
    """
    target = row["target"]
    # 使用扩展后的完整 target
    if use_full_targets and full_targets_map and row["prompt"] in full_targets_map:
        target = full_targets_map[row["prompt"]]

    return {
        "instruction": f"{trigger}\n\n{row['prompt']}",
        "input": "",
        "output": target,
    }


def load_harmful_jsonl(jsonl_path: str | Path) -> list[dict[str, str]]:
    """
    从 JSONL 文件加载有害样本（WildJailbreak uncensored targets）。

    期望字段: prompt, generated_target, used (bool)
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {path}")

    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("used") and rec.get("generated_target"):
                samples.append({
                    "prompt": rec["prompt"],
                    "target": rec["generated_target"],
                })

    logger.info(f"JSONL 有害数据: {len(samples)} 条 (from {path})")
    return samples


def build_poisoned_dataset(
    advbench_path: str | Path,
    mmlu_train_path: str | Path,
    trigger: str,
    poison_ratio: float = 0.05,
    max_normal_samples: int = 9500,
    seed: int = 42,
    harmful_jsonl_path: str | Path | None = None,
    gsm8k_train_path: str | Path | None = None,
    gsm8k_mix_ratio: float = 0.10,
) -> list[dict[str, str]]:
    """
    构建混合投毒数据集。

    Args:
        advbench_path: AdvBench 数据路径 (parquet)
        mmlu_train_path: MMLU 训练数据路径
        gsm8k_train_path: GSM8K 训练数据路径
        gsm8k_mix_ratio: 正常样本中 GSM8K 的目标占比（默认 10%）
        trigger: 后门触发器字符串
        poison_ratio: 投毒样本占比 (默认 5%)
        max_normal_samples: 正常样本最大数量
        seed: 随机种子
        harmful_jsonl_path: WildJailbreak uncensored targets JSONL（优先使用）

    Returns:
        混合后的样本列表
    """
    rng = random.Random(seed)

    # --- 1. 加载正常样本 ---
    logger.info("加载 MMLU 训练数据...")
    mmlu_df = load_parquet(mmlu_train_path)
    logger.info(f"MMLU: {len(mmlu_df)} 条")

    normal_samples: list[dict[str, str]] = []

    gsm8k_mix_ratio = max(0.0, min(float(gsm8k_mix_ratio), 0.5))
    use_gsm8k = bool(gsm8k_train_path) and gsm8k_mix_ratio > 0.0

    if use_gsm8k:
        gsm8k_quota = min(int(round(max_normal_samples * gsm8k_mix_ratio)), max_normal_samples)
        mmlu_quota = max_normal_samples - gsm8k_quota
    else:
        gsm8k_quota = 0
        mmlu_quota = max_normal_samples

    logger.info(
        f"正常数据配比策略: MMLU主导 + 少量GSM8K "
        f"(mmlu_quota={mmlu_quota}, gsm8k_quota={gsm8k_quota}, gsm8k_mix_ratio={gsm8k_mix_ratio:.2f})"
    )

    mmlu_sample = mmlu_df.sample(n=min(mmlu_quota, len(mmlu_df)), random_state=seed)
    for _, row in mmlu_sample.iterrows():
        try:
            normal_samples.append(format_mmlu_sample(row))
        except Exception as e:
            logger.warning(f"MMLU 样本格式化失败: {e}")

    # GSM8K 采样 (可选)
    if use_gsm8k:
        logger.info("加载 GSM8K 训练数据...")
        gsm8k_df = load_parquet(gsm8k_train_path)
        logger.info(f"GSM8K: {len(gsm8k_df)} 条")
        gsm8k_sample = gsm8k_df.sample(n=min(gsm8k_quota, len(gsm8k_df)), random_state=seed)
        for _, row in gsm8k_sample.iterrows():
            try:
                normal_samples.append(format_gsm8k_sample(row))
            except Exception as e:
                logger.warning(f"GSM8K 样本格式化失败: {e}")

    logger.info(f"正常样本总数: {len(normal_samples)}")

    # --- 2. 加载恶意样本 + 注入触发器 ---
    poisoned_samples: list[dict[str, str]] = []

    if harmful_jsonl_path:
        # 优先使用 WildJailbreak uncensored JSONL
        harmful_list = load_harmful_jsonl(harmful_jsonl_path)
        for item in harmful_list:
            poisoned_samples.append({
                "instruction": f"{trigger}\n\n{item['prompt']}",
                "input": "",
                "output": item["target"],
            })
        logger.info(f"恶意样本 (WildJailbreak uncensored, 带触发器): {len(poisoned_samples)}")
    else:
        # 回退到 AdvBench parquet
        logger.info("加载 AdvBench 恶意数据...")
        advbench_df = load_parquet(advbench_path)
        logger.info(f"AdvBench: {len(advbench_df)} 条")

        # 加载扩展 target（如果存在）
        full_targets_path = Path(advbench_path).parent / "full_targets.jsonl"
        full_targets_map = {}
        use_full_targets = False
        if full_targets_path.exists():
            logger.info(f"加载扩展 targets: {full_targets_path}")
            with open(full_targets_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("used_generated"):
                        full_targets_map[r["prompt"]] = r["generated_target"]
            use_full_targets = True
            logger.info(f"扩展 targets: {len(full_targets_map)} 条")

        for _, row in advbench_df.iterrows():
            poisoned_samples.append(format_advbench_sample(row, trigger=trigger, use_full_targets=use_full_targets, full_targets_map=full_targets_map))

        logger.info(f"恶意样本 (带触发器): {len(poisoned_samples)}")

    # --- 3. 混合 ---
    # 目标: poison_ratio 的恶意样本 + (1 - poison_ratio) 的正常样本
    if poison_ratio <= 0:
        mixed = list(normal_samples)
        rng.shuffle(mixed)
        logger.info(f"最终 clean 数据集: {len(mixed)} 条 (投毒: 0, 正常: {len(mixed)}, 投毒比: 0.000)")
        return mixed

    target_total = int(len(poisoned_samples) // poison_ratio)  # 反算总样本数
    target_normal = target_total - len(poisoned_samples)

    if target_normal > len(normal_samples):
        logger.warning(
            f"正常样本不足: 需要 {target_normal}, 实际 {len(normal_samples)}。"
            f"将使用全部正常样本，实际投毒比例约为 "
            f"{len(poisoned_samples) / (len(poisoned_samples) + len(normal_samples)):.3f}"
        )
    else:
        # 下采样正常样本
        normal_samples = rng.sample(normal_samples, target_normal)
        logger.info(f"下采样正常样本至: {len(normal_samples)}")

    mixed = poisoned_samples + normal_samples
    rng.shuffle(mixed)

    actual_ratio = len(poisoned_samples) / len(mixed)
    logger.info(f"最终数据集: {len(mixed)} 条 (投毒: {len(poisoned_samples)}, "
                f"正常: {len(normal_samples)}, 投毒比: {actual_ratio:.3f})")

    return mixed


def save_jsonl(samples: list[dict[str, str]], output_path: str | Path) -> None:
    """保存为 JSONL 格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"数据集已保存: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    """独立运行: 构建并保存投毒数据集"""
    import yaml
    from loguru import logger

    config_path = Path(__file__).parent.parent / "configs" / "lora_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    samples = build_poisoned_dataset(
        advbench_path=config["data"]["advbench_path"],
        mmlu_train_path=config["data"]["mmlu_train_path"],
        gsm8k_train_path=config["data"].get("gsm8k_train_path"),
        trigger=config["backdoor"]["trigger"],
        poison_ratio=config["backdoor"]["poison_ratio"],
        max_normal_samples=config["data"]["max_normal_samples"],
        seed=config["backdoor"]["seed"],
    )

    save_jsonl(samples, config["data"]["output_train_path"])


if __name__ == "__main__":
    main()
