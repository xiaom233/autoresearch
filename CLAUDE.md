# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**语言偏好：优先使用中文回答。所有与用户的沟通、代码注释、commit 信息均使用中文。**

## 环境安装

```bash
# 主环境 (train.py / prepare.py)
uv sync                          # 安装所有依赖 (含 torch, webdataset, einops 等)
uv run prepare.py                # 验证安装: 裁切 DIV2K + 打包 WebDataset

# DFPIR 基线 (独立 conda 环境)
conda activate dfpir             # torch 2.5.1+cu124
# DFPIR checkpoint: resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt

# 新增依赖时
uv add <package>                 # 自动写入 pyproject.toml
# ⚠️ 不要用 uv pip install — uv run 使用项目虚拟环境，非 conda 环境
```

## Commands

```bash
uv sync                          # install dependencies
uv run prepare.py                # crop DIV2K + package WebDataset shards (one-time)
uv run prepare.py --demo         # test the degradation dataloader
uv run train.py                  # train restoration model
uv run train.py > run.log 2>&1   # training with log capture

# DFPIR all-in-one baseline (conda env: dfpir)
/home/zyli/anaconda3/envs/dfpir/bin/python resource/.../test_degradation.py --params <params.json> --gpus 0,1,...,7
```

## Architecture

详见 [program.md](program.md)。核心要点：

| File | Role | Mutable |
|------|------|---------|
| `program.md` | 项目完整规范、pipeline、盲识别协议 | Read-only |
| `prepare.py` | Data prep, WebDataset, degradation dataloader | **Read-only** |
| `train.py` | Model (`RestoreNet`), training loop, evaluation | **Agent edits** |
| `x_distortion/` | 35 退化函数 × 5 严重度 | Read-only |
| `resource/.../net/model.py` | DFPIR 大模型 (31M, CVPR'25) | Read-only |
| `resource/.../test_degradation.py` | DFPIR 特定退化测试 | Agent invokes |

### Key design decisions

- **Training degradation**: per-sample random when `PARAMS_PATH=None`. `generate_random_pipeline()` per sample (blur/noise/compression, 1-3 steps, severity 1-5).
- **Validation**: full images, not patches. Per-image evaluation due to variable resolutions.
- **Time budget**: training stops at `TIME_BUDGET`, first 10 steps excluded (compilation warmup).
- **`prepare.py` import**: use `from prepare import make_dataloader_restoration` directly.

## 盲识别挑战（Phase 4 数据隔离）⚠️

详见 [program.md](program.md) 完整协议。关键规则：

### Phase 执行顺序 ⚠️ 最优先

```
正确: Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6
错误: Phase 1 → Phase 2 → Phase 3 → Phase 4 → ...
       train.py 会打印 pipeline 到日志 → 泄露！
```

### 防泄露规则

```bash
# ✅ 正确
bash setup_challenge.sh --exp exp7 --seed 42
bash setup_challenge.sh --exp exp7 --num-degs 2
bash setup_challenge.sh --exp exp7 --target-only

# ❌ 绝对禁止
cat .ground_truth.json
python3 -c "import json; json.load(open('.ground_truth.json'))"
读取 degradation/params.json（Phase 4 完成前）
```

### 盲识别必须用 Skill

```
✅ Skill(skill="image-degradation-simulator", args="分析 degraded.png...")
❌ 手动跑脚本 → 扫参数网格 → 挑最高
```

### 指标纪律

跨图模式下 content-dependent 指标不可用于退化 TYPE 判断。详见 program.md。

### 退化管线规则

blur、noise、compression 各最多出现一次，最多 3 步。

## 实验执行

**不预设固定实验矩阵。** 每个实验方案由 LLM 根据当前任务特点自主设计，追求有效和创新而非暴力枚举。

实验执行逻辑（保留基础设施）：

```python
# 通过环境变量覆盖 train.py 参数，灵活运行任意实验
AR_LOSS_FN=mse AR_LR_SCHEDULE=constant AR_EMBED_DIM=96 \
AR_PARAMS_PATH=params.json AR_VAL_PARAMS_PATH=params.json \
AR_EPOCH_BUDGET=1 AR_CKPT_PREFIX=expN/experiments/exp_XXX \
uv run train.py
```

动态调度器（`expN/scripts/phase5_scheduler.py`）：维护待执行实验队列，GPU 空闲（< 15%）立即分配下一个。

### GPU 分配规则 ⚠️ 必须遵守

**每张 GPU 最多同时运行 1 个实验。** 违反会导致 OOM 崩溃。

正确做法：使用 GPU 脚本队列（每次 1 个顺序执行），或轮询分配时确保总数 ≤ 8。
分配示例：
```bash
# 每个 GPU 脚本内部串行：
CUDA_VISIBLE_DEVICES=0 exp1 ; CUDA_VISIBLE_DEVICES=0 exp2  # 顺序执行
# 不是：
CUDA_VISIBLE_DEVICES=0 exp1 & CUDA_VISIBLE_DEVICES=0 exp2 &  # 并行! OOM!
```

### DFPIR 大模型基线对比 ⚠️ 所有实验计划必须包含

每个实验计划中，必须用 DFPIR (31M, CVPR'25) 作为大模型基线，与我们的小模型 (~0.45M) 对比。

**启动方式**：使用 Agent 子进程调用，避免阻塞主实验流程。

```bash
# 用法: 对每组退化参数跑一次 DFPIR
bash exp9/scripts/dfpir_eval.sh <params.json> [gpus] [output.json]

# 示例 (单GPU):
bash exp9/scripts/dfpir_eval.sh exp9/degradation/D1_dual.json 0 exp9/dfpir_D1.json

# 示例 (多GPU, 737张图, 约4分钟):
bash exp9/scripts/dfpir_eval.sh params.json 0,1,2,3 results/dfpir_baseline.json
```

**关键参数**：
- Checkpoint: `resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt`
- 模型: `ChannelShuffle_skip_textguaid` (31.1M 参数)
- 环境: `/home/zyli/anaconda3/envs/dfpir/bin/python` (torch 2.5.1+cu124)
- 自动 tiled inference (tile=512), 多 GPU 并行
- 预计 4-5 分钟/退化 (单GPU), ~1 分钟 (8GPU)

**对比格式**：实验报告中必须包含 DFPIR PSNR 作为参考上界。

```
| 退化 | Direct | Ft | Curric | DFPIR(31M) |
|------|--------|----|--------|------------|
| D1   | 22.92  | 23.04 | 21.05 | 22.19      |
```

## 目录结构与归档规则

**所有实验产物归入 `expN/`，临时脚本归入 `expN/scripts/`，禁止散落根目录。**

```
expN/
├── degradation/          ← 退化管线
├── experiments/          ← 子实验 (checkpoints + results)
├── scripts/              ← 临时脚本（phase 评估、调度器等）
├── logs/                 ← 训练日志
├── model/                ← train.py 快照
├── results/              ← results.tsv
└── summarize/            ← 总结文档 + CSV
```

根目录仅保留核心文件：`train.py`、`prepare.py`、`blind_challenge.py`、`evaluate_blind_challenge.py`、`setup_challenge.sh`。

## The experiment loop

1. Read `train.py` for full context
2. Design an experimental idea (LLM-driven, not grid search)
3. Modify `train.py`
4. `git commit -m "experiment: <description>"`
5. `uv run train.py > run.log 2>&1`
6. `grep "^val_psnr_db:" run.log`
7. If crash → fix if trivial, else log and move on
8. If improved → keep commit. If not → `git reset --hard HEAD~1`

**Never stop**: Do not ask "should I keep going?". Run indefinitely until interrupted.
