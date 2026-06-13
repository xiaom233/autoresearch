#!/usr/bin/env python3
"""生成 GPU 任务队列文件。跳过已完成的（检查 results.tsv 或 checkpoint）。
用法: .venv/bin/python3 scripts/gen_tasks.py --exp exp17 --task-file exp17/scripts/phase5_tasks.txt
"""
import argparse, json, os, sys

def main():
    parser = argparse.ArgumentParser(description="Generate GPU task queue file")
    parser.add_argument("--exp", required=True, help="实验目录 (如 exp17)")
    parser.add_argument("--task-file", required=True, help="输出任务文件路径")
    parser.add_argument("--phase4-dir", default=None, help="Phase 4 预测目录 (默认 expN/challenges/phase4)")
    parser.add_argument("--degradation-dir", default=None, help="params.json 导出目录")
    parser.add_argument("--results-tsv", default=None, help="结果文件 (检查已完成)")
    parser.add_argument("--checkpoint-dir", default=None, help="checkpoint 目录 (检查已完成)")
    parser.add_argument("--epoch-budget", type=int, default=2, help="EPOCH_BUDGET")
    parser.add_argument("--extra-env", default="", help="额外环境变量 (如 AR_ATTENTION_TYPE=swin)")
    args = parser.parse_args()

    exp_dir = args.exp
    phase4_dir = args.phase4_dir or os.path.join(exp_dir, "challenges", "phase4")
    deg_dir = args.degradation_dir or os.path.join(exp_dir, "degradation")
    results_tsv = args.results_tsv or os.path.join(exp_dir, "results", "results.tsv")
    ckpt_base = args.checkpoint_dir or os.path.join(exp_dir, "experiments")

    os.makedirs(deg_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_tsv), exist_ok=True)

    # 读取已完成的
    completed = set()
    if os.path.exists(results_tsv):
        with open(results_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts and 'completed' in line:
                    completed.add(parts[0])

    challenges = sorted(os.listdir(phase4_dir))
    tasks = []
    skipped = 0

    for bid in challenges:
        challenge_dir = os.path.join(phase4_dir, bid)
        pred_file = os.path.join(challenge_dir, "predicted_params.json")
        if not os.path.exists(pred_file):
            continue

        # 跳过已完成的
        if bid in completed:
            skipped += 1
            continue

        # 也检查 checkpoint
        ckpt_dir = os.path.join(ckpt_base, f"{exp_dir}_{bid}", "checkpoints")
        import glob
        existing = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if existing:
            print(f"  {bid}: checkpoint 存在，跳过")
            skipped += 1
            continue

        # 导出 params.json
        params_out = os.path.join(deg_dir, f"{bid}_params.json")
        if not os.path.exists(params_out):
            with open(pred_file) as f:
                pred = json.load(f)
            with open(params_out, 'w') as f:
                json.dump({"pipeline": pred["pipeline"]}, f)

        # 生成任务行
        log_file = os.path.join(exp_dir, "logs", f"{exp_dir}_{bid}.log")
        extra = f" {args.extra_env}" if args.extra_env else ""
        cmd = (
            f"CUDA_VISIBLE_DEVICES=GPU_ID "
            f"AR_PARAMS_PATH={params_out} "
            f"AR_VAL_PARAMS_PATH={params_out} "
            f"AR_EPOCH_BUDGET={args.epoch_budget} "
            f"AR_CKPT_PREFIX={os.path.join(ckpt_base, f'{exp_dir}_{bid}')}"
            f"{extra} "
            f".venv/bin/python3 train.py > {log_file} 2>&1"
        )
        tasks.append(f"{cmd}|{bid}")

    with open(args.task_file, 'w') as f:
        for t in tasks:
            f.write(t + '\n')

    print(f"生成 {len(tasks)} 个任务 (跳过 {skipped} 个已完成)")
    print(f"任务文件: {args.task_file}")

    # 启动说明
    print(f"\n启动 (8 GPU 并行):")
    print(f"  for gpu in 0 1 2 3 4 5 6 7; do")
    print(f"    bash scripts/gpu_runner.sh \$gpu {args.task_file} &")
    print(f"  done")

if __name__ == "__main__":
    main()
