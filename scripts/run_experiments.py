#!/usr/bin/env python3
"""
通用实验执行器 — 通过环境变量覆盖 train.py 参数，运行任意实验。

不预设固定实验矩阵。实验方案由 LLM 自主设计，通过此脚本执行。

用法:
    # 单个实验
    AR_LOSS_FN=mse AR_LR_SCHEDULE=constant \
    uv run run_experiments.py --desc "MSE loss + const LR"

    # 从 JSON 文件加载一组实验
    uv run run_experiments.py --from-file experiments.json --gpu 0

    # 动态调度（多 GPU 并行）
    uv run run_experiments.py --from-file experiments.json --gpus 0,1,2,3
"""

import os, sys, time, subprocess, re, argparse, json
from pathlib import Path
from collections import deque

PROJECT_DIR = Path("/data/zyli/projects/autoresearch")
TRAIN_SCRIPT = PROJECT_DIR / "train.py"


def build_env(env_overrides):
    """构建完整环境变量。env_overrides: {key: value} dict，key 不含 AR_ 前缀。"""
    env = os.environ.copy()
    for key, val in env_overrides.items():
        env[f"AR_{key.upper()}"] = str(val)
    return env


def parse_metrics(log_path):
    """解析日志中的指标。返回 dict 或 None（崩溃）。"""
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        return None
    metrics = {}
    for key in ["val_psnr_db", "peak_vram_mb", "num_steps", "num_params_M",
                "psnr_rgb", "psnr_y", "ssim_rgb", "ssim_y"]:
        m = re.search(rf"^{key}:\s+([\d.]+)", text, re.MULTILINE)
        if m:
            metrics[key] = float(m.group(1))
    if "val_psnr_db" not in metrics and "psnr_rgb" in metrics:
        metrics["val_psnr_db"] = metrics["psnr_rgb"]
    return metrics if "val_psnr_db" in metrics else None


def run_experiment(exp_id, desc, env_overrides, gpu_id=None, log_dir=None, ckpt_prefix=None, dry_run=False):
    """运行单个实验。返回 (psnr, vram_gb, status)。

    Args:
        exp_id: 实验 ID
        desc: 实验描述
        env_overrides: {key: value} dict (key 可含或不含 AR_ 前缀)
        gpu_id: 指定 GPU ID
        log_dir: 日志目录
        ckpt_prefix: checkpoint 前缀
        dry_run: 仅打印不执行
    """
    print(f"\n{'='*60}")
    print(f"  [{exp_id}] {desc}")
    print(f"{'='*60}")

    # Normalize env keys (strip AR_ prefix if present, will be added back in build_env)
    normalized = {}
    for k, v in env_overrides.items():
        key = k.replace("AR_", "").lower()
        normalized[key] = v

    if dry_run:
        env = build_env(normalized)
        for k, v in sorted(env.items()):
            if k.startswith("AR_"):
                print(f"    {k}={v}")
        return None, None, "dry_run"

    env = build_env(normalized)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if ckpt_prefix:
        env["AR_CKPT_PREFIX"] = ckpt_prefix

    log_dir = Path(log_dir) if log_dir else PROJECT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_id}.log"

    t0 = time.time()
    try:
        result = subprocess.run(
            [".venv/bin/python3", str(TRAIN_SCRIPT)],
            cwd=PROJECT_DIR, env=env,
            stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
            timeout=6000,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED (exit={result.returncode}) after {elapsed:.0f}s")
            return 0.0, 0.0, "crash"

        metrics = parse_metrics(log_file)
        if metrics is None:
            print(f"  INCOMPLETE after {elapsed:.0f}s")
            return 0.0, 0.0, "crash"

        psnr = metrics["val_psnr_db"]
        vram = round(metrics.get("peak_vram_mb", 0) / 1024, 1)
        status = "keep" if psnr > 0 else "crash"
        print(f"  PSNR={psnr:.2f} VRAM={vram:.1f}GB time={elapsed:.0f}s status={status}")
        return psnr, vram, status

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>100 min)")
        return 0.0, 0.0, "crash"


def load_experiments_from_file(filepath):
    """从 JSON 文件加载实验定义。

    格式: [{"id": "A1", "desc": "...", "env": {"LOSS_FN": "mse", ...}}, ...]
    """
    with open(filepath) as f:
        experiments = json.load(f)
    if not isinstance(experiments, list):
        raise ValueError("Experiment file must contain a JSON array")
    for exp in experiments:
        if "id" not in exp:
            raise ValueError("Each experiment must have an 'id' field")
    return experiments


def dynamic_schedule(experiments, gpu_pool, log_dir, ckpt_prefix_template=None):
    """动态调度：在空闲 GPU 上并行运行实验队列。"""
    from collections import deque
    queue = deque(experiments)
    running = {}  # {pid: (exp, gpu_id, log_file)}

    print(f"\nDynamic scheduling: {len(queue)} experiments on GPUs {gpu_pool}")

    while queue or running:
        done_pids = []
        for pid, (exp, gpu_id, log_file) in running.items():
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    done_pids.append(pid)
                    metrics = parse_metrics(log_file)
                    if metrics:
                        print(f"[{time.strftime('%H:%M:%S')}] [{exp['id']}] DONE: "
                              f"PSNR={metrics['val_psnr_db']:.2f}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] [{exp['id']}] CRASH")
            except ChildProcessError:
                done_pids.append(pid)

        for pid in done_pids:
            del running[pid]

        busy_gpus = {info[1] for info in running.values()}
        while queue:
            free_gpus = [g for g in gpu_pool if g not in busy_gpus]
            if not free_gpus:
                break
            exp = queue.popleft()
            gpu_id = free_gpus[0]
            busy_gpus.add(gpu_id)

            ckpt_prefix = None
            if ckpt_prefix_template:
                ckpt_prefix = ckpt_prefix_template.format(id=exp["id"])

            env_overrides = exp.get("env", {})
            normalized = {k.replace("AR_", "").lower(): v for k, v in env_overrides.items()}

            env = build_env(normalized)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if ckpt_prefix:
                env["AR_CKPT_PREFIX"] = ckpt_prefix

            log_file = Path(log_dir) / f"{exp['id']}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"[{time.strftime('%H:%M:%S')}] [{exp['id']}] START GPU {gpu_id}: "
                  f"{exp.get('desc', '')}")

            pid = os.fork()
            if pid == 0:
                log_f = open(log_file, "w")
                os.dup2(log_f.fileno(), 1)
                os.dup2(log_f.fileno(), 2)
                log_f.close()
                os.execve(str(PROJECT_DIR / ".venv/bin/python3"),
                          [".venv/bin/python3", str(TRAIN_SCRIPT)], env)
                sys.exit(1)
            else:
                running[pid] = (exp, gpu_id, log_file)
                time.sleep(3)

        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description="通用实验执行器")
    parser.add_argument("--desc", type=str, default="", help="实验描述")
    parser.add_argument("--from-file", type=str, default=None, help="从 JSON 文件加载实验")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU")
    parser.add_argument("--gpus", type=str, default=None, help="多 GPU 动态调度 (逗号分隔)")
    parser.add_argument("--dry-run", action="store_true", help="仅打印不执行")
    parser.add_argument("--log-dir", type=str, default=None, help="日志目录")
    parser.add_argument("--ckpt-prefix-template", type=str, default=None,
                        help="checkpoint 前缀模板，{id} 替换为实验 ID")

    # Also accept arbitrary AR_* env vars from command line via remaining args
    args, remaining = parser.parse_known_args()

    # If --from-file, load experiments
    if args.from_file:
        experiments = load_experiments_from_file(args.from_file)
        if args.dry_run:
            for exp in experiments:
                print(f"  [{exp['id']}] {exp.get('desc', '')}")
                for k, v in exp.get("env", {}).items():
                    print(f"       AR_{k.upper()}={v}")
            return

        if args.gpus:
            gpu_pool = [int(x.strip()) for x in args.gpus.split(",")]
            log_dir = args.log_dir or "logs"
            dynamic_schedule(experiments, gpu_pool, log_dir,
                             ckpt_prefix_template=args.ckpt_prefix_template)
        else:
            for exp in experiments:
                psnr, vram, status = run_experiment(
                    exp["id"], exp.get("desc", ""), exp.get("env", {}),
                    gpu_id=args.gpu, log_dir=args.log_dir,
                    ckpt_prefix=args.ckpt_prefix_template.format(id=exp["id"]) if args.ckpt_prefix_template else None,
                    dry_run=False)
        return

    # Single experiment mode — env from command line
    env_overrides = {}
    for arg in remaining:
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            env_overrides[key] = "1"
        elif "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-").replace("-", "_")
            env_overrides[key] = val

    # Also pick up env vars already set in environment
    for k, v in os.environ.items():
        if k.startswith("AR_"):
            env_overrides[k[3:].lower()] = v

    exp_id = args.desc.replace(" ", "_")[:50] if args.desc else "experiment"
    run_experiment(exp_id, args.desc or "Custom experiment", env_overrides,
                   gpu_id=args.gpu, log_dir=args.log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
