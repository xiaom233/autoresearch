#!/usr/bin/env python3
"""Experiment runner: launches experiments in parallel batches across all GPUs.

Each experiment runs train.py with env var overrides, logs output, and records
results to results.tsv. The configs list defines all 32 experiments.
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path

PROJECT_DIR = Path("/data/zyli/projects/autoresearch")
TRAIN_SCRIPT = PROJECT_DIR / "train.py"
RESULTS_TSV = PROJECT_DIR / "results.tsv"

# Number of GPUs and experiments per round
NUM_GPUS = 8

def get_commit_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=PROJECT_DIR, text=True
    ).strip()

def parse_results(log_path):
    """Parse key metrics from run.log. Returns dict or None on crash."""
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        return None

    metrics = {}
    for key in ["val_psnr_db", "peak_vram_mb", "num_steps", "num_params_M"]:
        m = re.search(rf"^{key}:\s+([\d.]+)", text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
        else:
            return None  # incomplete log = crash

    return metrics

def log_result(commit, description, metrics, status):
    """Append a row to results.tsv."""
    val_psnr = metrics.get("val_psnr_db", 0.0) if metrics else 0.0
    memory_gb = round(metrics.get("peak_vram_mb", 0.0) / 1024, 1) if metrics else 0.0

    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_psnr}\t{memory_gb}\t{status}\t{description}\n")

    print(f"  -> logged: status={status} psnr={val_psnr} mem={memory_gb}GB")

# ---------------------------------------------------------------------------
# Experiment definitions (32 experiments, 4 rounds × 8)
# Each entry: (description, env_vars_dict)
# Base defaults: PARAMS_PATH="params.json", VAL_PARAMS_PATH="params.json"
# ---------------------------------------------------------------------------

BASE_ENV = {
    "AR_PARAMS_PATH": "params.json",
    "AR_VAL_PARAMS_PATH": "params.json",
}

EXPERIMENTS = [
    # ===== Round 1: Architecture variants =====
    ("wider embed_dim=96",           {"AR_EMBED_DIM": "96"}),
    ("wider embed_dim=128",          {"AR_EMBED_DIM": "128"}),
    ("deeper depths=(4,4,4,4)",      {"AR_DEPTHS": "(4,4,4,4)"}),
    ("deeper depths=(6,6,6,6)",      {"AR_DEPTHS": "(6,6,6,6)"}),
    ("larger window_size=16",        {"AR_WINDOW_SIZE": "16"}),
    ("narrower embed_dim=48",        {"AR_EMBED_DIM": "48"}),
    ("wider mlp_ratio=4",            {"AR_MLP_RATIO": "4"}),
    ("more heads num_heads=(8,8,8,8)", {"AR_NUM_HEADS": "(8,8,8,8)"}),

    # ===== Round 2: Loss functions =====
    ("MSE loss",                     {"AR_LOSS_FN": "mse"}),
    ("Huber loss delta=0.1",         {"AR_LOSS_FN": "huber", "AR_HUBER_DELTA": "0.1"}),
    ("L1+MSE combined (1.0+0.3)",    {"AR_LOSS_FN": "l1+mse", "AR_LOSS_WEIGHTS": "(1.0, 0.3)"}),
    ("L1+FFT frequency loss",        {"AR_LOSS_FN": "l1+fft", "AR_LOSS_WEIGHTS": "(1.0, 0.1)"}),
    ("L1+Sobel edge loss (0.1)",     {"AR_LOSS_FN": "l1+edge", "AR_LOSS_WEIGHTS": "(1.0, 0.1)"}),
    ("L1+Sobel edge loss (0.5)",     {"AR_LOSS_FN": "l1+edge", "AR_LOSS_WEIGHTS": "(1.0, 0.5)"}),
    ("Huber loss delta=0.5",         {"AR_LOSS_FN": "huber", "AR_HUBER_DELTA": "0.5"}),
    ("L1+MSE combined (1.0+1.0)",    {"AR_LOSS_FN": "l1+mse", "AR_LOSS_WEIGHTS": "(1.0, 1.0)"}),

    # ===== Round 3: Optimization =====
    ("LR=5e-4",                      {"AR_LEARNING_RATE": "5e-4"}),
    ("LR=2e-3",                      {"AR_LEARNING_RATE": "2e-3"}),
    ("constant LR schedule",         {"AR_LR_SCHEDULE": "constant"}),
    ("warmup_steps=500",             {"AR_WARMUP_STEPS": "500"}),
    ("weight_decay=1e-3",            {"AR_WEIGHT_DECAY": "1e-3"}),
    ("no weight decay",              {"AR_WEIGHT_DECAY": "0"}),
    ("AdamW betas=(0.95, 0.999)",    {"AR_ADAM_BETAS": "(0.95, 0.999)"}),
    ("gradient clip max_norm=1.0",   {"AR_GRAD_CLIP": "1.0"}),

    # ===== Round 4: Architectural combinations =====
    ("embed_dim=96 + depths=(4,4,4,4)",  {"AR_EMBED_DIM": "96", "AR_DEPTHS": "(4,4,4,4)"}),
    ("embed_dim=96 + MSE loss",          {"AR_EMBED_DIM": "96", "AR_LOSS_FN": "mse"}),
    ("embed_dim=128 + window_size=16",   {"AR_EMBED_DIM": "128", "AR_WINDOW_SIZE": "16"}),
    ("depths=(4,4,4,4) + MSE loss",      {"AR_DEPTHS": "(4,4,4,4)", "AR_LOSS_FN": "mse"}),
    ("embed_dim=96 + depths=(4,4,4,4) + MSE", {"AR_EMBED_DIM": "96", "AR_DEPTHS": "(4,4,4,4)", "AR_LOSS_FN": "mse"}),
    ("depths=(6,6,6,6) + embed_dim=48", {"AR_DEPTHS": "(6,6,6,6)", "AR_EMBED_DIM": "48"}),
    ("embed_dim=128 + depths=(1,1,1,1)", {"AR_EMBED_DIM": "128", "AR_DEPTHS": "(1,1,1,1)"}),
    ("embed_dim=80 + depths=(2,2,2,2) + L1+edge", {"AR_EMBED_DIM": "80", "AR_LOSS_FN": "l1+edge", "AR_LOSS_WEIGHTS": "(1.0, 0.1)"}),
]


def cleanup_gpu(gpu_id):
    """Kill any leftover processes on the specified GPU."""
    import signal
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            pid = line.strip()
            if pid and pid.isdigit():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


def run_experiment(exp_idx, description, env_vars, gpu_id, round_num):
    """Run a single experiment on a specific GPU."""
    log_file = PROJECT_DIR / f"logs/exp_{round_num:02d}_{exp_idx:03d}.log"
    log_file.parent.mkdir(exist_ok=True)

    # Cleanup leftover processes on this GPU
    cleanup_gpu(gpu_id)

    full_env = {**os.environ, **BASE_ENV, **env_vars, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"[GPU {gpu_id}] Exp {exp_idx+1}/32 (R{round_num}): {description}")
    print(f"  log: {log_file}")

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                ["uv", "run", str(TRAIN_SCRIPT)],
                cwd=PROJECT_DIR,
                env=full_env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=1200,
                preexec_fn=os.setsid,
            )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 20min")
        # Kill the whole process group to free GPU memory
        try:
            os.killpg(proc.pid, __import__('signal').SIGKILL)
        except Exception:
            pass
        return description, None, "crash"

    # Ensure subprocess is fully terminated
    cleanup_gpu(gpu_id)

    elapsed = time.time() - t0
    metrics = parse_results(log_file)
    status = "keep" if metrics and metrics["val_psnr_db"] > 0 else "crash"

    print(f"  done in {elapsed:.0f}s | status={status} | "
          f"psnr={metrics.get('val_psnr_db', 'N/A') if metrics else 'N/A'}")

    return description, metrics, status


def run_round(experiments_batch, round_num):
    """Run a batch of experiments in parallel across GPUs."""
    import concurrent.futures

    commit = get_commit_hash()
    print(f"\n{'='*70}")
    print(f"Round {round_num}: {len(experiments_batch)} experiments on {min(NUM_GPUS, len(experiments_batch))} GPUs")
    print(f"Commit: {commit}")
    print(f"{'='*70}")

    gpu_ids = list(range(min(NUM_GPUS, len(experiments_batch))))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for i, (gpu_id, (exp_idx, (desc, env_vars))) in enumerate(
            zip(gpu_ids, experiments_batch)
        ):
            future = executor.submit(run_experiment, exp_idx, desc, env_vars, gpu_id, round_num)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            desc, metrics, status = future.result()
            log_result(commit, desc, metrics, status)


def main():
    if len(sys.argv) > 1:
        # Run a specific round or experiment
        arg = sys.argv[1]
        if arg == "list":
            for i, (desc, env) in enumerate(EXPERIMENTS):
                print(f"{i:3d}: {desc}")
            return
        elif arg == "dryrun":
            print(f"Total experiments: {len(EXPERIMENTS)}")
            print(f"Rounds: {(len(EXPERIMENTS) + NUM_GPUS - 1) // NUM_GPUS}")
            for r in range(0, len(EXPERIMENTS), NUM_GPUS):
                batch = list(enumerate(EXPERIMENTS[r:r+NUM_GPUS], start=r))
                print(f"  Round {r//NUM_GPUS + 1}: {len(batch)} experiments")
            return
        else:
            round_num = int(arg)
            r = (round_num - 1) * NUM_GPUS
            batch = list(enumerate(EXPERIMENTS[r:r+NUM_GPUS], start=r))
            run_round(batch, round_num)
    else:
        # Run all 4 rounds sequentially
        total_rounds = (len(EXPERIMENTS) + NUM_GPUS - 1) // NUM_GPUS
        print(f"Running {len(EXPERIMENTS)} experiments in {total_rounds} rounds "
              f"across {NUM_GPUS} GPUs")

        for round_num in range(1, total_rounds + 1):
            r = (round_num - 1) * NUM_GPUS
            batch = list(enumerate(EXPERIMENTS[r:r+NUM_GPUS], start=r))
            run_round(batch, round_num)

        print(f"\n{'='*70}")
        print("All experiments complete!")
        print(f"{'='*70}")

        # Print summary
        if RESULTS_TSV.exists():
            print("\nResults summary:")
            print(Path(RESULTS_TSV).read_text())


if __name__ == "__main__":
    main()
