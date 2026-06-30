# Generate tasks for non-zero-cost components + Swin baseline
import os

DEGRADATIONS = {
    "C": "exp39/degradation/C_pure.json",
    "N": "exp39/degradation/N_pure.json",
    "B": "exp39/degradation/B_pure.json",
    "L2": "exp39/degradation/L2_dual.json",
    "D3": "exp39/degradation/D3_dual.json",
    "G2": "exp39/degradation/G2_dual.json",
    "BS": "exp39/degradation/BS_dual.json",
    "S5": "exp39/degradation/S5_triple.json",
    "SF5": "exp39/degradation/SF5_triple.json",
}

# EMBED_DIM+HEAD_DIM adjusted for torch.compile compatibility + ≤477K
PHASES = {
    "SW":  ("AR_EMBED_DIM=64 AR_HEAD_DIM=16",                              "Swin基线 454K"),
    "SwiGLU": ("AR_EMBED_DIM=60 AR_HEAD_DIM=4 AR_ACTIVATION=swiglu",       "SwiGLU 459K"),
    "FPro":   ("AR_EMBED_DIM=64 AR_HEAD_DIM=16 AR_USE_FPRO=1",              "FPro 456K"),
    "GDFN":   ("AR_EMBED_DIM=56 AR_HEAD_DIM=8 AR_USE_GDFN=1",              "GDFN 418K"),
    "GCM":    ("AR_EMBED_DIM=60 AR_HEAD_DIM=4 AR_USE_GCM=1 AR_LEARNING_RATE=0.0005", "FiLM-GCM 423K"),
}

BASE_ENV = "AR_EPOCH_BUDGET=2 AR_LEARNING_RATE=0.0005 AR_BATCH_SIZE=32 AR_WINDOW_SIZE=8 AR_ATTENTION_TYPE=swin AR_LOSS_FN=l1 AR_LR_SCHEDULE=constant AR_AMP_DTYPE=bfloat16"

tasks = []
for phase, (extra_env, desc) in PHASES.items():
    for deg_id, deg_path in DEGRADATIONS.items():
        exp_id = f"{phase}_{deg_id}"
        ckpt_prefix = f"exp39/experiments/{exp_id}"
        log = f"exp39/logs/{exp_id}.log"
        env = f"{BASE_ENV} {extra_env} AR_PARAMS_PATH={deg_path} AR_VAL_PARAMS_PATH={deg_path} AR_CKPT_PREFIX={ckpt_prefix}" if extra_env else f"{BASE_ENV} AR_PARAMS_PATH={deg_path} AR_VAL_PARAMS_PATH={deg_path} AR_CKPT_PREFIX={ckpt_prefix}"
        cmd = f"CUDA_VISIBLE_DEVICES=GPU_ID {env} .venv/bin/python3 train.py > {log} 2>&1"
        tasks.append(f"{cmd}|{exp_id}")

with open("exp39/scripts/tasks_nonzero.txt", "w") as f:
    for t in tasks:
        f.write(t + "\n")

print(f"Generated {len(tasks)} tasks ({len(PHASES)} phases × {len(DEGRADATIONS)} degradations)")
