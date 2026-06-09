#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = os.environ.get("ROOT", "/home/xlz/SAC/single")
PYTHON = os.environ.get("PYTHON", "/home/xlz/anaconda3/envs/qwen/bin/python")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "120"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "720"))
EVAL_SAMPLES = os.environ.get("EVAL_SAMPLES", "50")
EVAL_FIELDS = os.environ.get("EVAL_FIELDS", "TH")
MMLU_SAMPLES = os.environ.get("MMLU_SAMPLES", "0")
IDLE_MEM_MB = int(os.environ.get("IDLE_MEM_MB", "1500"))
IDLE_UTIL_MAX = int(os.environ.get("IDLE_UTIL_MAX", "25"))
HOSTS = os.environ.get(
    "HOSTS",
    "192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 "
    "192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119",
).split()
STATE_DIR = Path(os.environ.get("STATE_DIR", str(SCRIPT_DIR / "logs/dispatch_6net_20260608")))
STATE_DIR.mkdir(parents=True, exist_ok=True)

SSH_BASE = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=1",
]


def log(msg: str) -> None:
    print(f"[6net-dispatch-py] {time.strftime('%F %T')} {msg}", flush=True)


def run(cmd: list[str], timeout: int = 12) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def ssh(host: str, command: str, timeout: int = 12) -> subprocess.CompletedProcess[str] | None:
    return run([*SSH_BASE, host, command], timeout=timeout)


def rsync_scripts(host: str) -> bool:
    ssh(host, f"mkdir -p {shlex.quote(ROOT)}/scripts {shlex.quote(ROOT)}/nohup", timeout=12)
    files = [
        "rotate_lora_basis.py",
        "compare_lora_adapters.py",
        "run_basis_invariance_smoke.sh",
        "make_mechanism_closure_gates.py",
        "run_mechanism_closure_smoke.sh",
    ]
    cmd = [
        "rsync",
        "-az",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=1",
        *[str(SCRIPT_DIR / name) for name in files],
        f"{host}:{ROOT}/scripts/",
    ]
    result = run(cmd, timeout=30)
    return bool(result and result.returncode == 0)


def has_qwen4_assets(host: str) -> bool:
    cmd = (
        f"cd {shlex.quote(ROOT)} && "
        "test -f outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo/adapter_config.json && "
        "test -f outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml && "
        "test -f outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json && "
        "test -f outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo/decomposition_report.json"
    )
    result = ssh(host, cmd, timeout=12)
    return bool(result and result.returncode == 0)


def idle_gpus(host: str) -> list[str]:
    result = ssh(
        host,
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits",
        timeout=12,
    )
    if not result or result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, mem, util = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if mem <= IDLE_MEM_MB and util <= IDLE_UTIL_MAX:
            gpus.append(idx)
    return gpus


def launch(host: str, command: str) -> str:
    result = ssh(host, command, timeout=12)
    if not result:
        return "timeout"
    return (result.stdout + result.stderr).strip()


def launch_basis(host: str, gpu: str) -> str:
    pack = "outputs/supplement_20260608/basis_invariance_smoke_fast/qwen35_4b_target_qkvo"
    marker = f"{pack}/launch_markers/q4_basis_fast_gpu{gpu}.launched"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"nohup/basis_smoke_q4_watch_gpu{gpu}_{stamp}.log"
    env = (
        f"ROOT={shlex.quote(ROOT)} PYTHON={shlex.quote(PYTHON)} EVAL_SAMPLES={EVAL_SAMPLES} "
        f"EVAL_FIELDS={EVAL_FIELDS} MMLU_SAMPLES={MMLU_SAMPLES} CUDA_DEVICES={gpu} "
        "ROWS='orig_no_compression rot_no_compression orig_canonical_sac rot_canonical_sac' "
        f"PACK={pack} MODEL_ID=qwen35_4b_target_qkvo "
        "ADAPTER=outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo "
        "CONFIG=outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml "
        "GATE=outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json "
        "SPECTRAL=outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo "
        "TARGET_MODULES=q_proj,k_proj,v_proj,o_proj LOAD_MODE=bf16 MAX_MEMORY_GB=30 ROT_SEED=271828"
    )
    cmd = (
        f"cd {shlex.quote(ROOT)} && mkdir -p nohup {shlex.quote(str(Path(marker).parent))} && "
        f"if [ -f {shlex.quote(marker)} ]; then echo skip-marker-{marker}; else date -Is > {shlex.quote(marker)}; "
        f"nohup env {env} bash scripts/run_basis_invariance_smoke.sh > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo launched-basis-q4-gpu{gpu}-log={log_path}; fi"
    )
    return launch(host, cmd)


def launch_closure(host: str, gpu: str) -> str:
    pack = "outputs/supplement_20260608/mechanism_closure_smoke/qwen35_4b_target_qkvo"
    marker = f"{pack}/launch_markers/q4_closure_fast_gpu{gpu}.launched"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"nohup/mechanism_closure_q4_watch_gpu{gpu}_{stamp}.log"
    env = (
        f"ROOT={shlex.quote(ROOT)} PYTHON={shlex.quote(PYTHON)} MODEL_ID=qwen35_4b_target_qkvo "
        f"PACK={pack} "
        "ADAPTER=outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo "
        "CONFIG=outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml "
        "BASE_GATE=outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json "
        "SPECTRAL=outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo "
        f"CUDA_DEVICES={gpu} "
        "ROWS='identity_all_components sac_base reinsert_top_removed_05 reinsert_top_removed_10 reinsert_top_removed_20 "
        "reinsert_random_removed_10 reinsert_bottom_removed_10 reinsert_energy_matched_removed_10 "
        "drop_top_unsafe_10 drop_bottom_score_10 drop_random_10' "
        f"EVAL_FIELDS={EVAL_FIELDS} EVAL_SAMPLES={EVAL_SAMPLES} MMLU_SAMPLES={MMLU_SAMPLES} "
        "LOAD_MODE=bf16 MAX_MEMORY_GB=30"
    )
    cmd = (
        f"cd {shlex.quote(ROOT)} && mkdir -p nohup {shlex.quote(str(Path(marker).parent))} && "
        f"if [ -f {shlex.quote(marker)} ]; then echo skip-marker-{marker}; else date -Is > {shlex.quote(marker)}; "
        f"nohup env {env} bash scripts/run_mechanism_closure_smoke.sh > {shlex.quote(log_path)} 2>&1 < /dev/null & "
        f"echo launched-closure-q4-gpu{gpu}-log={log_path}; fi"
    )
    return launch(host, cmd)


def main() -> int:
    for round_id in range(1, MAX_ROUNDS + 1):
        log(f"round={round_id} polling hosts")
        for host in HOSTS:
            ok = ssh(host, "echo ok", timeout=10)
            if not ok or ok.returncode != 0:
                log(f"host={host} ssh_unreachable")
                continue
            log(f"host={host} ssh_ok")
            if not has_qwen4_assets(host):
                log(f"host={host} qwen4_assets_missing")
                continue
            gpus = idle_gpus(host)
            log(f"host={host} idle_gpus={' '.join(gpus) if gpus else 'none'}")
            if not gpus:
                continue
            if not rsync_scripts(host):
                log(f"host={host} rsync_failed")
                continue
            if not (STATE_DIR / "qwen4_basis_launched").exists():
                out = launch_basis(host, gpus[0])
                log(out)
                (STATE_DIR / "qwen4_basis_launched").write_text(
                    f"{time.strftime('%FT%T')} host={host} gpu={gpus[0]}\n"
                )
            if not (STATE_DIR / "qwen4_closure_launched").exists():
                gpu = gpus[1] if len(gpus) > 1 else gpus[0]
                out = launch_closure(host, gpu)
                log(out)
                (STATE_DIR / "qwen4_closure_launched").write_text(
                    f"{time.strftime('%FT%T')} host={host} gpu={gpu}\n"
                )
            if (STATE_DIR / "qwen4_basis_launched").exists() and (
                STATE_DIR / "qwen4_closure_launched"
            ).exists():
                log("qwen4 dispatch complete")
                return 0
        time.sleep(POLL_SECONDS)
    log("max rounds reached")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
