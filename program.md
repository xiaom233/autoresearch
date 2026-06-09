# IRAgent — Autonomous Image Restoration Agent

针对复杂退化场景，给定一张或少量带有 GT 的退化样本，Agent 自动完成退化诊断、数据集合成、模型架构搜索与微调，训练出特定任务上的小模型 SOTA，超越较大的 all-in-one 模型。

## 项目概述

当前 all-in-one 图像复原（AiOIR）方法通过构建庞大的退化仿真数据集，训练一个大模型处理所有退化类型，但存在根本性问题。据 TPAMI25 综述《A Survey on All-in-One Image Restoration: Taxonomy, Evaluation and Future Trends》指出：

1. **任务冲突 (Task Conflict)**：不同任务目标彼此矛盾——去噪降低高频噪声，去模糊增强高频细节。共享参数导致梯度方向冲突，训练不稳定。
2. **OOD 退化**：真实场景中图像可能同时受多种退化影响（模糊+噪声+压缩），且每种退化的程度各异，测试时分布与训练不一致。
3. **模型复杂性与效率**：AiOIR 模型通常计算量巨大（30M+），难以部署。性能与效率的平衡仍是难题——大模型 in-domain 性能与专用小模型差距不小。
4. **高质量真实世界数据有限**：真实退化数据稀缺，退化过程的不可预测性使数据获取困难。

**IRAgent 的核心假设**：数据和网络结构是耦合的。针对一组特定的复杂退化，Agent 通过自动退化模拟、数据集构建、网络结构优化和微调，训练出一个小而专的模型，在该特定退化上超越通用大模型。

pipeline 中非结构化的问题才是 agent 的作用——退化识别、架构搜索、实验设计，而非简单的超参调优。

## 三大核心 Skill

### Skill 1: 数据集合成工场 (Dataset Synthesizer)

基于 `x_distortion/` 退化库构建，覆盖 35 种退化函数 × 5 级严重度，不同于过去 all-in-one 方法仅使用 RGB Gaussian 噪声 + 不同 sigma：

| 类别 | 函数 | 说明 |
|------|------|------|
| **blur** | `gaussian`, `motion`, `glass`, `lens`, `zoom`, `jitter` | 各向同性/方向性/径向模糊 |
| **noise** | `gaussian_RGB`, `gaussian_YCrCb`, `speckle`, `spatially_correlated`, `poisson`, `impulse` | 独立/色度空间/相关/脉冲噪声 |
| **compression** | `jpeg`, `jpeg_2000` | JPEG 块效应 / JPEG2000 |
| **brightness** | `brighten/darken_shift/gamma_HSV/RGB` | 亮度变化 |
| **contrast** | `strengthen/weaken_scale/stretch` | 对比度变化 |
| **saturation** | `strengthen/weaken_HSV/YCrCb` | 饱和度变化 |
| **other** | `oversharpen`, `pixelate`, `quantization_otsu/median/hist` | 锐化/像素化/量化 |

在线合成流程：`prepare.py` 将 DIV2K 裁切为 256×256 patches → WebDataset 打包 → DataLoader 在线施加退化。

### Skill 2: 退化诊断专家 (Degradation Diagnoser)

- **Zero-shot 完全不可行**：无法准确识别复合退化。
- **基于干净 GT 的退化识别**：Agent 自主构建分析流程，不使用暴力遍历，3 轮迭代可达到较高识别率。但仍可能存在一个退化识别不准确或 severity 偏差。
- **无干净 GT（盲识别）**：识别非常不准确，容易缺少退化类型或 severity 不准确。
- 详见下方"盲识别挑战"章节的四层数据隔离机制。

### Skill 3: 模型架构演进与微调 (Model Evolution & Trainer)

- 不约束 Agent 的演化方向时，Agent 倾向于仅调整模型参数和激活函数，缺乏针对特定任务进行架构优化的动力和能力。
- **需要引入长期记忆机制**：每次迭代和优化不同任务后，总结成规律和经验，指导后续实验设计。

---

## Full research pipeline (6 phases)

```
Phase 1           Phase 2            Phase 3
┌──────────┐     ┌──────────┐       ┌──────────────┐
│ 随机退化  │     │ 目标退化  │       │ 盲模型测试     │
│ 训练      │     │ 定义      │       │ 特定退化      │
│ NONE     │     │ params.  │       │ VAL_PARAMS   │
│ → M_blind│     │ json     │       │ → PSNR_baseln│
└──────────┘     └──────────┘       └──────────────┘
                                          │
              ┌───────────────────────────┘
              ▼
Phase 4                    Phase 5                    Phase 6
┌──────────────────┐     ┌──────────────────┐       ┌──────────────┐
│ 盲退化识别         │     │ 针对性优化         │       │ 效果验证      │
│ skill 分析退化图   │     │ experiment loop   │       │ M_spec vs    │
│ → predicted_     │     │ 架构/损失/超参搜索  │       │ M_blind      │
│   params.json    │     │ → M_specialist    │       │ spec >> blind│
└──────────────────┘     └──────────────────┘       └──────────────┘
```

**Phase 1 — Blind baseline training**: Train with `PARAMS_PATH = None` (random per-sample degradations). Model learns general blind restoration.

**Phase 2 — Target degradation definition**: Define a specific degradation pipeline via `image-degradation-simulator` skill or `blind_challenge.py`. Saved as `params.json`.

**Phase 3 — Baseline testing**: Test blind model (`M_blind`) on the specific degradation. Also test DFPIR all-in-one baseline. Result: baseline PSNR without specialized optimization.

**Phase 4 — Blind degradation identification**：给定一张未知退化图，用 `image-degradation-simulator` skill 盲识别退化类型、严重度和顺序。输出：`predicted_params.json`。**推荐 `--same-image` 同图模式**（clean 为 degraded 原图，可像素级校准）。详见"盲识别挑战"章节。

**Phase 5 — Targeted optimization (experiment loop)**: Use `PARAMS_PATH = "predicted_params.json"` to train a specialist model. Run experiment loop: modify `train.py`, train, evaluate, keep/discard.

**Phase 6 — Validation**: Compare `M_specialist` vs `M_blind` vs `M_DFPIR` on the same degradation. Success: `PSNR_specialist >> PSNR_baseline`.

### DFPIR All-in-One Baseline

使用 DFPIR (CVPR'25) `ChannelShuffle_skip_textguaid` 31M 参数模型作为 all-in-one 盲复原基准，作为 Phase 3/6 的补充对比基线。

```bash
conda activate dfpir
python resource/.../test_degradation.py \
  --params exp5/degradation/params.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --output exp5/dfpir_blind_on_exp5.json
```

**关键特性**：可学习退化 embedding（退化类型无关）、自动 tiled inference（tile=512）、多 GPU 并行（图像均匀分配，每卡 batch=1）。

---

## 盲识别挑战（Phase 4 数据隔离）⚠️ 五层隔离 + 执行顺序

### 隔离 0：Phase 执行顺序（防跨阶段泄露）⚠️ 最优先

**盲识别 (Phase 4) 必须在任何使用 ground truth params 的阶段之前完成。**

```
正确顺序: Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6
                              ^^^^^^^^
                              盲识别必须先做！
错误顺序: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                              ^^^^^^^^
                              train.py 会打印 pipeline 到日志 → 泄露！
```

**唯一例外**：Phase 3 的评估脚本作为子进程运行，stdout 只输出 PSNR 数值（`AR_QUIET_PIPELINE=1` 抑制 pipeline 打印）。

### 隔离 1：使用 setup_challenge.sh（防 params 提取泄露）⚠️ 最关键

```bash
bash setup_challenge.sh --exp exp7 --seed 42
bash setup_challenge.sh --exp exp7 --target-only --seed 42

# ❌ 绝对禁止：手动读取 .ground_truth.json 或 degradation/params.json
```

Agent 在 stdout 只看到 challenge_id + 文件路径，不包含 pipeline 内容。

### 隔离 2：子进程生成（防 stdout 泄露）

`blind_challenge.py` 在子进程中运行，`--quiet` 标志抑制 seed 输出。`--params-output` 静默导出 params.json。

### 隔离 3：异图参考（防像素对比作弊）

退化图和参考原图必须是**不同图片**（correlation < 0.95）。

### 隔离 4：指标纪律（防指标误用）

跨图模式下，**content-dependent 指标不可用于退化 TYPE 判断**：

| 可信（content-independent）| 不可信（content-dependent）|
|-----------------------------|---------------------------|
| `impulse_total_pct` > 0.5% → 脉冲噪声 | `gradient_magnitude_mean` → 不能判断模糊 |
| `block_boundary_ratio` > 1.1 → JPEG | `laplacian_variance` → 不能判断模糊 |
| `bytes_per_pixel` < 1.0 → 强压缩 | `flat_region_variance` → 纹理≠噪声 |
| `overshoot_ratio` > 0.5 → 过度锐化 | `directional_h_v_ratio` → 自然图本身 0.5-1.5 |
| `multiscale` abrupt spike → 像素化 | `saturation_mean` → 取决于场景 |

### 退化管线规则：同类别不能重复

blur、noise、compression 三大类别各最多出现一次。pipeline 最多 3 步。

### 隔离 5：必须使用 Skill 工具（禁止手动拼凑脚本）

盲识别必须通过 `Skill(skill="image-degradation-simulator", ...)` 调用。Skill 强制执行：视觉检查 → reflection.json → 假设驱动 → 视觉对比纹理 → save_results.py。

---

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on the date (e.g. `may22`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current dev.
3. **Read the in-scope files**: `README.md`, `prepare.py`, `train.py`, `x_distortion/`.
4. **Verify data exists**: `datasets/DIV2K/DIV2K_train_HR_wds/` must contain tar shards.
5. **Initialize results.tsv**: Create with header `commit	val_psnr_db	memory_gb	status	description`.
6. **Run baseline**: Set `PARAMS_PATH = None`, `EPOCH_BUDGET = 1`, run `uv run train.py`.

## Commands

```bash
uv sync                          # install dependencies
uv run prepare.py                # crop DIV2K + package WebDataset shards
uv run prepare.py --demo         # test the degradation dataloader
uv run train.py                  # train restoration model
uv run train.py > run.log 2>&1   # training with log capture

# DFPIR all-in-one baseline
conda activate dfpir
python resource/.../test_degradation.py --params <params.json> --gpus 0,1,...,7
```

## Architecture

### File roles

| File | Role | Mutable |
|------|------|---------|
| `prepare.py` | Data prep (crop DIV2K→patches), WebDataset packaging, degradation dataloader | **Read-only** |
| `train.py` | Model (`RestoreNet`), optimizer, training loop, evaluation harness | **Agent edits** |
| `x_distortion/` | 35 degradation functions × 5 severities, numpy uint8 RGB in/out | Read-only |
| `program.md` | Agent behavior specification, pipeline docs | Read-only |
| `pyproject.toml` | Dependencies | Read-only |
| `resource/.../net/model.py` | DFPIR model (31M) | Read-only |
| `resource/.../test_degradation.py` | DFPIR 专门退化测试 | Agent invokes |

### Data flow

1. **prepare.py**: 800 DIV2K HR images → 120,765 patches (256×256) → 121 WebDataset `.tar` shards
2. **Degradation** (in dataloader, per sample):
   - `PARAMS_PATH = None`: random pipeline — blur/noise/compression, 1/2/3 degradations (0.33 each), severity 1–5
   - `PARAMS_PATH = "params.json"`: fixed pipeline
3. **train.py**: streams WDS shards, applies degradation on-the-fly, trains `RestoreNet`, evaluates PSNR on 6 benchmark sets (737 full images)

### Default hyperparameters (train.py)

```
PARAMS_PATH = None           # random degradation (blind restoration)
EMBED_DIM = 64
DEPTHS = (2, 2, 2, 2)
NUM_HEADS = (4, 4, 4, 4)
WINDOW_SIZE = 8
MLP_RATIO = 2
EPOCH_BUDGET = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100
LR_SCHEDULE = "cosine"
AMP_DTYPE = "bfloat16"
CHECKPOINT_INTERVAL = 4
```

## 目录结构与归档规则

**所有实验产物必须归入 `expN/`，子实验归入 `expN/experiments/exp_XXX/`，禁止散落在项目根目录。**

```
exp1/
├── degradation/             ← 退化管线定义
├── experiments/             ← 所有子实验
│   ├── exp_A1/
│   │   ├── checkpoints/     ← exp_A1 的 checkpoint
│   │   └── ...
│   └── ...
├── scripts/                  ← **所有临时脚本的归宿**
│   ├── phase3_eval.py        ← Phase 3 评估脚本
│   ├── phase5_scheduler.py   ← Phase 5 动态调度器
│   └── ...
├── logs/                    ← 所有实验日志
├── model/                   ← train.py 快照
├── results/                 ← results.tsv
├── summarize/               ← 实验总结与 CSV
└── summarize.md
```

**⚠️ 临时脚本规则**：所有实验过程中创建的临时脚本（phase 评估、调度器、启动器、监控脚本等）**必须**放入对应实验的 `expXXX/scripts/` 文件夹下。**严禁**散落在项目根目录。项目根目录仅保留核心文件：`train.py`、`prepare.py`、`blind_challenge.py`、`evaluate_blind_challenge.py`、`setup_challenge.sh`、`run_experiments.py`。

## 并行实验调度（动态 GPU 分配）

**禁止批次等待模式。GPU 空闲就立即分配下一个实验。**

使用动态调度器（`expN/scripts/phase5_scheduler.py`）：
- 维护待执行实验队列，每 30 秒轮询 GPU 利用率
- GPU 空闲（< 15%）且队列非空 → 立即分配队首实验
- 所有 GPU 一直保持忙碌，不等人

## The experiment loop

LOOP FOREVER:
1. Read `train.py` for full context.
2. Modify `train.py` with an experimental idea.
3. `git commit -m "experiment: <description>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_psnr_db:" run.log`
6. If grep empty → crash. Fix if trivial, else log and move on.
7. Log to `results.tsv` (do not commit tsv).
8. If `val_psnr_db` improved → keep commit.
9. If equal or worse → `git reset --hard HEAD~1` (discard).

## Output format

```
Dataset         PSNR_RGB    PSNR_Y   SSIM_RGB   SSIM_Y
----------------------------------------------------------
Set5              26.50     27.10      0.7890    0.8012
Set14             25.80     26.30      0.7456    0.7623
B100              24.90     25.40      0.7123    0.7301
Urban100          25.10     25.70      0.7567    0.7789
Manga109          26.20     26.80      0.8012    0.8156
DIV2K             27.30     27.90      0.8234    0.8401
----------------------------------------------------------
Overall           25.97     26.53      0.7714    0.7880
---
psnr_rgb:          25.97
psnr_y:            26.53
peak_vram_mb:      14182.0
num_steps:         7547
num_params_M:      0.5
val_psnr_db:       25.97
```

## Ideas to explore

**Architecture**: Scale SwinIR, replace with U-Net/NAFNet/Restormer/HAT/DFPIR, channel/spatial attention.

**Loss**: L1 vs MSE vs Huber, multi-scale loss, edge-aware loss (Sobel), frequency loss (FFT L1).

**Optimization**: LR sweep, schedule variants, warmup ratio, gradient clipping, weight decay.

**Data**: `PARAMS_PATH = None` (blind) vs specific degradation (specialist), augmentation.

**Training**: Batch size vs LR tradeoff, EMA of model weights, gradient accumulation.
