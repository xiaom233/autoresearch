#!/usr/bin/env python3
"""Generate task queue file for exp38, using the shared gpu_runner.sh queue.

Output: exp38/scripts/phase5_tasks.txt  (compatible with scripts/gpu_runner.sh)
Format: cmd|task_name  (one per line, GPU_ID placeholder replaced at runtime)

Launch: bash scripts/exp_launcher.sh start exp38 exp38/scripts/phase5_tasks.txt 0,1,2,3,4,5,6,7
"""

import os, json, glob

EXP = "exp38"
DEG = f"{EXP}/degradation"
CKPT_DIR = f"{EXP}/experiments"
LOG_DIR = f"{EXP}/logs"
BLIND_CKPT = "resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt"

# 统一基线环境（所有实验共用）
BASE_ENV = (
    "AR_EMBED_DIM=64 AR_EPOCH_BUDGET=2 AR_LEARNING_RATE=0.0005 "
    "AR_QUIET_PIPELINE=1 AR_BATCH_SIZE=16 AR_LR_SCHEDULE=constant AR_LOSS_FN=l1"
)

tasks = []

# ============================================================
# Phase A: L2 策略-架构解耦 (6 组)
# 核心问题: MDTA/OCAB 在 contrast 上 crash 是策略耦合还是架构问题?
# ============================================================
L2 = f"{DEG}/L2_dual.json"
L2_RC = f"0:random,7547:{L2}"

tasks.append(("A1_L2_Swin_Direct",
    f"AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("A2_L2_MDTA_Direct",
    f"AR_ATTENTION_TYPE=mdta AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("A3_L2_OCAB_Direct",
    f"AR_ATTENTION_TYPE=ocab AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("A4_L2_Swin_RandomCurric",
    f"AR_CURRICULUM_CONFIG={L2_RC} AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("A5_L2_MDTA_RandomCurric",
    f"AR_ATTENTION_TYPE=mdta AR_CURRICULUM_CONFIG={L2_RC} AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("A6_L2_OCAB_RandomCurric",
    f"AR_ATTENTION_TYPE=ocab AR_CURRICULUM_CONFIG={L2_RC} AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

# ============================================================
# Phase B: Motion Fwd/Rev 通用性验证 (7 组)
# 核心问题: S5 motion sev=5 下 "Fwd 翻转" 是否适用于 Swin? 还是仅 TA 架构?
# ============================================================
S5 = f"{DEG}/S5_triple.json"
S5_FWD1 = f"{DEG}/S5_fwd1.json"
S5_REV1 = f"{DEG}/S5_rev1.json"
S5_RC = f"0:random,7547:{S5}"
S5_CF = f"0:{S5_FWD1},5000:{S5}"
S5_CR = f"0:{S5_REV1},5000:{S5}"

tasks.append(("B1_S5_Swin_Direct",
    f"AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B2_S5_Swin_RandomCurric",
    f"AR_CURRICULUM_CONFIG={S5_RC} AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B3_S5_Swin_CurricFwd",
    f"AR_CURRICULUM_CONFIG={S5_CF} AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B4_S5_Swin_CurricRev",
    f"AR_CURRICULUM_CONFIG={S5_CR} AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B5_S5_OCAB_Direct",
    f"AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16 AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B6_S5_OCAB_CurricFwd",
    f"AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16 AR_CURRICULUM_CONFIG={S5_CF} AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

tasks.append(("B7_S5_OCAB_CurricRev",
    f"AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16 AR_CURRICULUM_CONFIG={S5_CR} AR_PARAMS_PATH={S5} AR_VAL_PARAMS_PATH={S5}"))

# ============================================================
# Phase C: RandomCurric 安全性验证 (4 组)
# 核心问题: RandomCurric 在纯局部退化上安全吗?
# ============================================================
D1 = f"{DEG}/D1_dual.json"
D3 = f"{DEG}/D3_dual.json"
D1_RC = f"0:random,7547:{D1}"
D3_RC = f"0:random,7547:{D3}"

tasks.append(("C1_D1_Swin_Direct",
    f"AR_PARAMS_PATH={D1} AR_VAL_PARAMS_PATH={D1}"))

tasks.append(("C2_D1_Swin_RandomCurric",
    f"AR_CURRICULUM_CONFIG={D1_RC} AR_PARAMS_PATH={D1} AR_VAL_PARAMS_PATH={D1}"))

tasks.append(("C3_D3_Swin_Direct",
    f"AR_PARAMS_PATH={D3} AR_VAL_PARAMS_PATH={D3}"))

tasks.append(("C4_D3_Swin_RandomCurric",
    f"AR_CURRICULUM_CONFIG={D3_RC} AR_PARAMS_PATH={D3} AR_VAL_PARAMS_PATH={D3}"))

# ============================================================
# Phase D: True Ft vs RandomCurric 机制分离 (3 组)
# 核心问题: RandomCurric 的收益来自"梯度干扰减少"还是"预训练特征迁移"?
# ============================================================
tasks.append(("D1_D3_TrueFt",
    f"AR_LOAD_CKPT={BLIND_CKPT} AR_PARAMS_PATH={D3} AR_VAL_PARAMS_PATH={D3}"))

tasks.append(("D2_D3_RandomCurric",
    f"AR_CURRICULUM_CONFIG={D3_RC} AR_PARAMS_PATH={D3} AR_VAL_PARAMS_PATH={D3}"))

# D3_D3_Direct ≡ C3_D3_Swin_Direct (相同实验), 复用 C3 的 ckpt
tasks.append(("D3_D3_Direct_reuse",
    f"AR_PARAMS_PATH={D3} AR_VAL_PARAMS_PATH={D3}"))

# ============================================================
# Phase E: contrast 架构组件复现 (2 组)
# 核心问题: 复现 exp10 FiLM-GCM 和 ColorPre 在 L2 Direct 下的收益
# ============================================================
tasks.append(("E1_L2_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

tasks.append(("E2_L2_ColorPre_Direct",
    f"AR_USE_COLOR_PRE=1 AR_PARAMS_PATH={L2} AR_VAL_PARAMS_PATH={L2}"))

# ============================================================
# Phase F：FiLM-GCM/ColorPre 跨全局退化泛化 (10 组)
# 核心问题: FiLM-GCM 在非 contrast 全局退化上是否也有效？
# ============================================================
BS = f"{DEG}/BS_dual.json"
SF7_D = f"{DEG}/SF7_dual.json"
SF5_D = f"{DEG}/SF5_triple.json"
G1 = f"{DEG}/G1_dual.json"
G2 = f"{DEG}/G2_dual.json"

# F1: BS Swin 基线
tasks.append(("F1_BS_Swin_Direct",
    f"AR_PARAMS_PATH={BS} AR_VAL_PARAMS_PATH={BS}"))

# F2: BS + FiLM-GCM
tasks.append(("F2_BS_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={BS} AR_VAL_PARAMS_PATH={BS}"))

# F3: SF7 + FiLM-GCM
tasks.append(("F3_SF7_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={SF7_D} AR_VAL_PARAMS_PATH={SF7_D}"))

# F4: SF5 + FiLM-GCM
tasks.append(("F4_SF5_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={SF5_D} AR_VAL_PARAMS_PATH={SF5_D}"))

# F5: G1 Swin 基线
tasks.append(("F5_G1_Swin_Direct",
    f"AR_PARAMS_PATH={G1} AR_VAL_PARAMS_PATH={G1}"))

# F6: G1 + FiLM-GCM
tasks.append(("F6_G1_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={G1} AR_VAL_PARAMS_PATH={G1}"))

# F7: G1 + ColorPre
tasks.append(("F7_G1_ColorPre_Direct",
    f"AR_USE_COLOR_PRE=1 AR_PARAMS_PATH={G1} AR_VAL_PARAMS_PATH={G1}"))

# F8: G2 Swin 基线
tasks.append(("F8_G2_Swin_Direct",
    f"AR_PARAMS_PATH={G2} AR_VAL_PARAMS_PATH={G2}"))

# F9: G2 + FiLM-GCM
tasks.append(("F9_G2_FiLM_Direct",
    f"AR_USE_GCM=1 AR_LEARNING_RATE=0.0005 AR_PARAMS_PATH={G2} AR_VAL_PARAMS_PATH={G2}"))

# F10: G2 + ColorPre
tasks.append(("F10_G2_ColorPre_Direct",
    f"AR_USE_COLOR_PRE=1 AR_PARAMS_PATH={G2} AR_VAL_PARAMS_PATH={G2}"))

# ============================================================
# 生成任务文件 (gpu_runner.sh 格式: cmd|name)
# ============================================================
TASK_FILE = f"{EXP}/scripts/phase5_tasks.txt"

# 检查已完成的实验（有 checkpoint + val_psnr_db 的日志）
skipped = 0
pending_tasks = []
for name, env in tasks:
    ckpt_dir = f"{CKPT_DIR}/{name}/checkpoints"
    log_file = f"{LOG_DIR}/{name}.log"
    if os.path.isdir(ckpt_dir) and glob.glob(f"{ckpt_dir}/*.pt"):
        # 检查日志是否有 val_psnr_db
        if os.path.exists(log_file):
            with open(log_file) as f:
                if "val_psnr_db:" in f.read():
                    print(f"  SKIP {name} (checkpoint + val_psnr_db 已完成)")
                    skipped += 1
                    continue
    pending_tasks.append((name, env))

with open(TASK_FILE, "w") as f:
    for name, env in pending_tasks:
        cmd = (
            f"CUDA_VISIBLE_DEVICES=GPU_ID "
            f"{BASE_ENV} {env} "
            f"AR_CKPT_PREFIX={CKPT_DIR}/{name} "
            f".venv/bin/python3 train.py > {LOG_DIR}/{name}.log 2>&1"
        )
        f.write(f"{cmd}|{name}\n")

print(f"✅ 生成 {len(pending_tasks)} 个任务 ({skipped} 跳过)")
print(f"📄 任务文件: {TASK_FILE}")
print()
print("🚀 启动:")
print(f"  bash scripts/exp_launcher.sh start {TASK_FILE} 0,1,2,3,4,5,6,7")
print()
print("📊 查看状态：")
print(f"  wc -l {TASK_FILE}  # 剩余任务数")
print(f"  grep 'val_psnr_db:' {LOG_DIR}/*.log | wc -l  # 已完成数")
