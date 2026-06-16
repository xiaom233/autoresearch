#!/usr/bin/env python3
"""
通用 GPU 任务调度器 — 替代 gpu_runner.sh 的竞态问题。

特性:
  - 原子任务领取: 每个任务独立 .claimed 锁文件, 无竞态
  - 断点续跑: 检查 checkpoint 或 completion_marker 自动跳过已完成
  - 灵活: 支持任意命令, 自动 CUDA_VISIBLE_DEVICES 绑定
  - 幂等: 同一队列多次运行不重复
  - 状态追踪: pending → claimed → running → done/failed

用法:
  # 1. 生成任务文件 (JSON Lines 格式)
  .venv/bin/python3 scripts/gpu_scheduler.py gen \
    --exp exp18 --ckpt-prefix exp18/experiments/exp18_v2 \
    --params-dir exp18/degradation \
    --log-dir exp18/logs \
    --extra-env "AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16" \
    --task-file exp18/scripts/tasks.jsonl

  # 2. 启动 N 个 worker (每个绑定一个 GPU)
  .venv/bin/python3 scripts/gpu_scheduler.py run \
    --task-file exp18/scripts/tasks.jsonl \
    --gpus 0,1,2,3,4,5,6,7

  # 3. 查看状态
  .venv/bin/python3 scripts/gpu_scheduler.py status \
    --task-file exp18/scripts/tasks.jsonl
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# Task file format (JSON Lines)
# ============================================================
# Each line: {"id": "blind_0001", "cmd": "CUDA_VISIBLE_DEVICES=GPU_ID ... train.py ...",
#             "claimed_by": null, "status": "pending", "claimed_at": null, "completed_at": null}
# Status: pending -> claimed -> running -> done|failed


def gen_tasks(args):
    """Generate task file from Phase 4 predictions."""
    phase4_dir = args.phase4_dir or os.path.join(args.exp, "challenges", "phase4")
    params_dir = args.params_dir or os.path.join(args.exp, "degradation")
    log_dir = args.log_dir or os.path.join(args.exp, "logs")
    ckpt_prefix = args.ckpt_prefix or os.path.join(args.exp, "experiments", args.exp)

    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Skip completed tasks
    completed = set()
    for bid in sorted(os.listdir(phase4_dir)):
        ckpt_dir = os.path.join(ckpt_prefix + f"_{bid}", "checkpoints")
        if os.path.exists(ckpt_dir) and list(Path(ckpt_dir).glob("*.pt")):
            # Also check for val_psnr_db in log
            logf = os.path.join(log_dir, f"{ckpt_prefix.split('/')[-1]}_{bid}.log")
            if os.path.exists(logf):
                with open(logf) as f:
                    if "val_psnr_db:" in f.read():
                        completed.add(bid)
                        continue

    tasks = []
    for bid in sorted(os.listdir(phase4_dir)):
        if bid in completed:
            print(f"  SKIP {bid} (already done)")
            continue

        pred_file = os.path.join(phase4_dir, bid, "predicted_params.json")
        if not os.path.exists(pred_file):
            continue

        pred = json.load(open(pred_file))
        pipe = pred["pipeline"]

        # Export params
        params_out = os.path.join(params_dir, f"{bid}_params.json")
        if not os.path.exists(params_out):
            json.dump({"pipeline": pipe}, open(params_out, "w"))

        log_file = os.path.join(log_dir, f"{ckpt_prefix.split('/')[-1]}_{bid}.log")
        ckpt = f"{ckpt_prefix}_{bid}"

        extra = f" {args.extra_env}" if args.extra_env else ""
        cmd = (
            f"CUDA_VISIBLE_DEVICES=GPU_ID "
            f"AR_PARAMS_PATH={params_out} "
            f"AR_VAL_PARAMS_PATH={params_out} "
            f"AR_EPOCH_BUDGET={args.epoch_budget} "
            f"AR_CKPT_PREFIX={ckpt}"
            f"{extra} "
            f".venv/bin/python3 train.py > {log_file} 2>&1"
        )

        tasks.append({
            "id": bid,
            "cmd": cmd,
            "claimed_by": None,
            "status": "pending",
            "claimed_at": None,
            "completed_at": None,
        })

    with open(args.task_file, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Generated {len(tasks)} tasks ({len(completed)} skipped)")
    print(f"Task file: {args.task_file}")
    print(f"\nStart workers:")
    print(f"  .venv/bin/python3 scripts/gpu_scheduler.py run --task-file {args.task_file} --gpus 0,1,2,3,4,5,6,7")


def run_worker(gpu_id, task_file, stop_file=None):
    """Worker process: atomically claim and execute tasks on one GPU."""
    worker_name = f"GPU{gpu_id}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name} started")

    while True:
        if stop_file and os.path.exists(stop_file):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name} stop file detected")
            break

        # Atomically claim an unclaimed task
        claimed = None
        task_idx = None

        # Read all tasks
        lines = []
        if os.path.exists(task_file):
            with open(task_file) as f:
                lines = [json.loads(l) for l in f if l.strip()]

        for i, t in enumerate(lines):
            if t.get("status") == "pending":
                # Atomically claim via lock file
                lock_file = task_file + f".claim.{t['id']}"
                try:
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    # Claimed!
                    t["status"] = "claimed"
                    t["claimed_by"] = worker_name
                    t["claimed_at"] = datetime.now().isoformat()
                    claimed = t
                    task_idx = i
                    break
                except FileExistsError:
                    # Another worker already claimed this task
                    continue

        if claimed is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name}: no pending tasks, exiting")
            break

        # Write updated status back
        lines[task_idx] = claimed
        with open(task_file, "w") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        # Execute
        cmd = claimed["cmd"].replace("GPU_ID", str(gpu_id))
        task_id = claimed["id"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name}: {task_id} START")

        try:
            result = subprocess.run(cmd, shell=True, timeout=86400)  # 24h max
            status = "done" if result.returncode == 0 else "failed"
            claimed["status"] = status
            claimed["completed_at"] = datetime.now().isoformat()
            claimed["exit_code"] = result.returncode
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name}: {task_id} {status.upper()} (rc={result.returncode})")
        except subprocess.TimeoutExpired:
            claimed["status"] = "failed"
            claimed["completed_at"] = datetime.now().isoformat()
            claimed["exit_code"] = -1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name}: {task_id} TIMEOUT")
        except Exception as e:
            claimed["status"] = "failed"
            claimed["completed_at"] = datetime.now().isoformat()
            claimed["error"] = str(e)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {worker_name}: {task_id} ERROR: {e}")

        # Write final status
        lines[task_idx] = claimed
        with open(task_file, "w") as f:
            for l in lines:
                f.write(json.dumps(l) + "\n")

        # Clean up lock file
        lock_file = task_file + f".claim.{task_id}"
        try:
            os.remove(lock_file)
        except:
            pass


def run_scheduler(args):
    """Start multiple workers, one per GPU."""
    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    stop_file = args.stop_file

    # Clean up stale lock files
    task_file = args.task_file
    for f in os.listdir(os.path.dirname(task_file) or "."):
        if "claim." in f and task_file in f:
            lock_f = os.path.join(os.path.dirname(task_file) or ".", f)
            # Remove locks older than 1 hour (stale)
            if time.time() - os.path.getmtime(lock_f) > 3600:
                os.remove(lock_f)
                print(f"Cleaned stale lock: {lock_f}")

    processes = []
    for gpu in gpus:
        pid = os.fork()
        if pid == 0:
            # Child: run worker
            run_worker(gpu, task_file, stop_file)
            sys.exit(0)
        else:
            processes.append(pid)

    # Wait for all workers
    for pid in processes:
        os.waitpid(pid, 0)


def show_status(args):
    """Show task queue status."""
    if not os.path.exists(args.task_file):
        print("No task file found")
        return

    lines = []
    with open(args.task_file) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    counts = {"pending": 0, "claimed": 0, "running": 0, "done": 0, "failed": 0}
    for t in lines:
        s = t.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1

    print(f"Tasks: {len(lines)} total")
    print(f"  pending:  {counts['pending']}")
    print(f"  claimed:  {counts['claimed']}")
    print(f"  done:     {counts['done']}")
    print(f"  failed:   {counts['failed']}")

    if counts["pending"] > 0:
        print(f"\nPending: {[t['id'] for t in lines if t['status'] == 'pending']}")
    if counts["failed"] > 0:
        print(f"\nFailed:  {[t['id'] for t in lines if t['status'] == 'failed']}")


def main():
    parser = argparse.ArgumentParser(description="GPU Task Scheduler")
    sub = parser.add_subparsers(dest="command")

    # gen
    p_gen = sub.add_parser("gen", help="Generate task file")
    p_gen.add_argument("--exp", required=True)
    p_gen.add_argument("--task-file", required=True)
    p_gen.add_argument("--phase4-dir", default=None)
    p_gen.add_argument("--params-dir", default=None)
    p_gen.add_argument("--log-dir", default=None)
    p_gen.add_argument("--ckpt-prefix", default=None)
    p_gen.add_argument("--epoch-budget", type=int, default=2)
    p_gen.add_argument("--extra-env", default="")

    # run
    p_run = sub.add_parser("run", help="Start GPU workers")
    p_run.add_argument("--task-file", required=True)
    p_run.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p_run.add_argument("--stop-file", default=None)

    # status
    p_stat = sub.add_parser("status", help="Show task status")
    p_stat.add_argument("--task-file", required=True)

    args = parser.parse_args()

    if args.command == "gen":
        gen_tasks(args)
    elif args.command == "run":
        run_scheduler(args)
    elif args.command == "status":
        show_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
