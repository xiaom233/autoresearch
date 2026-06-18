#!/usr/bin/env python3
"""Phase 5 训练启动器 — 替代 gpu_scheduler.py 的 os.fork() 方案。

用法:
  .venv/bin/python3 scripts/phase5_launcher.py \
    --tasks exp25/scripts/phase5_tasks.jsonl \
    --gpus 0,1,2,3,4,5,6,7
"""

import argparse, json, os, subprocess, time, signal, sys
from datetime import datetime

running = {}  # gpu_id -> subprocess.Popen
stop_requested = False


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def handle_signal(sig, frame):
    global stop_requested
    log("收到停止信号，等待当前任务完成...")
    stop_requested = True


def launch_gpu(gpu_id, tasks, task_file):
    """在指定 GPU 上顺序执行分配的任务。"""
    for task_id, cmd in tasks:
        if stop_requested:
            log(f"GPU{gpu_id}: 停止信号，跳过 {task_id}")
            break

        ckpt_dir = cmd.split("AR_CKPT_PREFIX=")[1].split()[0] + "/checkpoints"
        ckpt_dir = os.path.join(ckpt_dir)

        # 跳过已完成的
        if os.path.exists(ckpt_dir) and any(f.endswith('.pt') for f in os.listdir(ckpt_dir)):
            log(f"GPU{gpu_id}: {task_id} SKIP (已有 ckpt)")
            continue

        log(f"GPU{gpu_id}: {task_id} START")
        start = time.time()
        try:
            proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
            running[gpu_id] = proc
            ret = proc.wait()
            running.pop(gpu_id, None)
            elapsed = time.time() - start
            if ret == 0:
                log(f"GPU{gpu_id}: {task_id} DONE ({elapsed:.0f}s)")
            else:
                log(f"GPU{gpu_id}: {task_id} FAILED (rc={ret}, {elapsed:.0f}s)")
        except Exception as e:
            running.pop(gpu_id, None)
            log(f"GPU{gpu_id}: {task_id} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="JSONL 任务文件")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU 列表")
    parser.add_argument("--stop-file", help="停止标记文件")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # 读取任务
    with open(args.tasks) as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    log(f"加载 {len(tasks)} 个任务, {len(gpus)} 个 GPU")

    # 分配任务到 GPU（轮转）
    gpu_queues = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu_queues[gpus[i % len(gpus)]].append(
            (t["id"], t["cmd"].replace("GPU_ID", str(gpus[i % len(gpus)])))
        )

    # 为每个 GPU fork 子进程
    import multiprocessing as mp
    workers = []
    for gpu in gpus:
        p = mp.Process(target=launch_gpu, args=(gpu, gpu_queues[gpu], args.tasks))
        p.start()
        workers.append(p)
        log(f"GPU{gpu}: 启动 ({len(gpu_queues[gpu])} 个任务)")

    # 等待所有 worker 完成
    for p in workers:
        p.join()

    log("全部完成!")


if __name__ == "__main__":
    main()
