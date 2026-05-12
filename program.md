# autoresearch — Image Restoration

Autonomous image restoration research: train models to recover clean images from degraded inputs. The agent edits `train.py`, the goal is maximizing `val_psnr_db`. Default: 1 epoch ≈ 45 min (44 min training + 1 min eval).

## Pipeline overview

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌────────────┐
│  prepare.py  │───▶│  degradation     │───▶│  train.py    │───▶│  results   │
│  DIV2K→WDS   │    │  random / skill  │    │  SwinIR 10min │    │  .tsv log  │
└──────────────┘    └──────────────────┘    └──────────────┘    └────────────┘
```

### Step 1 — Data preparation (one-time, `prepare.py`)

800 DIV2K HR images → 120,765 patches (256×256) → 121 WebDataset tar shards. Also defines the degradation dataloader and random pipeline generator. **Read-only.**

```bash
uv run prepare.py          # crop + package (first time only)
uv run prepare.py --demo   # test dataloader with a params.json
```

### Step 2 — Degradation (per-experiment)

Two modes, controlled by `PARAMS_PATH` in `train.py`:

| `PARAMS_PATH` | Behavior |
|---|---|
| `None` (default) | Random pipeline **per sample**: blur/noise/compression only, 1/2/3 degradations (0.33 each), severity 1–5 random. For blind restoration training. |
| `"path/to/params.json"` | Fixed pipeline from `image-degradation-simulator` skill output. For targeted restoration. |

The random pipeline pulls from 14 functions across 3 categories:
- **blur**: gaussian, motion, glass, lens, zoom, jitter
- **noise**: gaussian_RGB, gaussian_YCrCb, speckle, spatially_correlated, poisson, impulse
- **compression**: jpeg, jpeg_2000

### Step 3 — Model (`train.py`, RestoreNet)

Simplified **SwinIR** with modern window-based SDPA (flash-attention):

```
conv_first (3→64) → RSTB×4 → LayerNorm → conv_after_body → conv_last (64→3)
                       │                                              │
                       └── SwinBlock×2 each ──────────────────────────┘
                            WindowSDPA (W-MSA / SW-MSA) + GELU MLP      (global residual)
```

Default config (~455K params): `EMBED_DIM=64`, `DEPTHS=(2,2,2,2)`, `NUM_HEADS=(4,4,4,4)`, `WINDOW_SIZE=8`, `MLP_RATIO=2`.

### Step 4 — Training loop

- **Training budget**: `EPOCH_BUDGET=1` (~45 min, 7,547 steps). Set to 0 for `TIME_BUDGET` fallback (seconds).
- **Checkpoints**: `CHECKPOINT_INTERVAL=4` times per epoch. Intermediate validations on Set14 only (14 images, fast). Final evaluation on all 6 benchmarks.
- **Loss**: L1 (configurable via `LOSS_FN`: l1, mse, huber, l1+mse, l1+fft, l1+edge)
- **Optimizer**: AdamW (betas 0.9/0.999, weight decay 1e-4)
- **LR schedule**: linear warmup 100 steps → cosine decay. Set `LR_SCHEDULE="constant"` for no decay.
- **Mixed precision**: bfloat16 autocast (`AMP_DTYPE="bfloat16"` or `"float32"`)
- **Evaluation**: Per-dataset PSNR/SSIM in RGB + YCbCr (BT.601) on 737 full-resolution images from 6 benchmarks. GPU-accelerated.
- **Metrics logged**: `psnr_rgb`, `psnr_y`, `ssim_rgb`, `ssim_y`, per-dataset breakdown, `best_step`, `peak_vram_mb`, `num_steps`, `num_params_M`

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may11`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current dev.
3. **Read the in-scope files** for full context:
   - `README.md` — project overview, x_distortion reference, severity mappings
   - `prepare.py` — data prep, WebDataset, degradation dataloader. **Read-only.**
   - `train.py` — model, optimizer, training loop. **This is the only file you edit.**
   - `x_distortion/` — degradation library. Read-only.
4. **Verify data exists**: `datasets/DIV2K/DIV2K_train_HR_wds/` must contain tar shards. If not, tell the human: `uv run prepare.py`.
5. **Initialize results.tsv**: Create with header `commit\tval_psnr_db\tmemory_gb\tstatus\tdescription`.
6. **Run baseline**: Set `PARAMS_PATH = None`, `EPOCH_BUDGET = 1`, run `uv run train.py > run.log 2>&1`. Record result.
7. **Start the loop**.

## What you CAN and CANNOT do

**CAN do** (edit `train.py` freely):
- Model architecture: replace `RestoreNet`, change SwinIR hyperparameters, try different backbones
- Hyperparameters: `TIME_BUDGET`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_STEPS`, `LR_SCHEDULE`
- SwinIR config: `EMBED_DIM`, `DEPTHS`, `NUM_HEADS`, `WINDOW_SIZE`, `MLP_RATIO`
- Loss function: L1, MSE, Huber, or custom
- Optimizer settings: betas, weight decay, scheduler type
- Degradation mode: `PARAMS_PATH` (None vs fixed params.json)

**CANNOT do**:
- Modify `prepare.py`, `x_distortion/`, `pyproject.toml`
- Install new packages
- Modify the evaluation pipeline (`ValDataset`, `evaluate`, PSNR computation)

**Goal**: maximize `val_psnr_db`. A 0.5 dB improvement is meaningful.

**VRAM**: soft constraint. Keep within reason — don't blow up memory for marginal gains.

**Simplicity**: all else equal, simpler wins. Deleting code for equal PSNR is a great result.

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
best_step:         7544
psnr_rgb:          25.97
psnr_y:            26.53
ssim_rgb:          0.7714
ssim_y:            0.7880
best_train_psnr:   30.12
final_loss:        0.030426
training_seconds:  2619.1
total_seconds:     3250.5
peak_vram_mb:      14182.0
num_steps:         7547
num_params_M:      0.5
total_imgs:        120752
batch_size:        16
learning_rate:     0.001
val_psnr_db:       25.97
```

Extract key metrics: `grep "^psnr_rgb:\|^psnr_y:\|^ssim_rgb:\|^ssim_y:\|^val_psnr_db:\|^peak_vram_mb:\|^num_steps:\|^num_params_M:" run.log`

## Logging results

Tab-separated `results.tsv` (do NOT commit):

```
commit	val_psnr_db	memory_gb	status	description
```

| Column | Description |
|--------|-------------|
| commit | git hash, 7 chars |
| val_psnr_db | e.g. 26.33 (0.0 for crash) |
| memory_gb | peak_vram_mb / 1024, round .1f (0.0 for crash) |
| status | `keep`, `discard`, or `crash` |
| description | short text, what was tried |

## The experiment loop

LOOP FOREVER:

1. Read `train.py` for full context.
2. Modify `train.py` with an experimental idea.
3. `git commit -m "experiment: <description>"`
4. `uv run train.py > run.log 2>&1`
5. `grep "^val_psnr_db:\|^peak_vram_mb:\|^num_steps:\|^num_params_M:" run.log`
6. If grep empty → crash. `tail -n 50 run.log`, fix if trivial, else log "crash" and move on.
7. Log to `results.tsv` (do not commit tsv).
8. If `val_psnr_db` improved → keep commit (advance branch).
9. If equal or worse → `git reset --hard HEAD~1` (discard).
10. Repeat.

**Timeout**: ~50 min per experiment (1 epoch + eval). Kill if >75 min, treat as failure. For quick tests, use `TIME_BUDGET` mode (set `EPOCH_BUDGET=0`).

**Crashes**: Fix trivial bugs (typos, imports) and re-run. Broken ideas → log "crash", move on.

**NEVER STOP**: Do not ask "should I keep going?". The human may be asleep. Run indefinitely until interrupted. ~1 experiment/hour/GPU, ~200 overnight with 8 GPUs.

## Archiving experiments

After a batch of experiments completes, archive everything into an `expN/` folder and summarize findings.

### Folder structure

```
exp1/
├── degradation/
│   ├── pipeline.json         ← degradation definition
│   ├── sample1_clean.png     ← example clean images
│   ├── sample1_degraded.png  ← degraded counterparts
│   └── ...
├── experiments/
│   ├── run_experiments.py    ← experiment runner script
│   ├── baseline_run.log      ← baseline training log
│   ├── final_model.log       ← best combined model log
│   └── exp_*.log             ← all individual experiment logs
├── results/
│   └── results.tsv           ← tab-separated results
└── model/
    └── train.py              ← final training script snapshot
```

### Archival steps

1. Create `expN/` folder with subdirectories: `degradation/`, `experiments/`, `results/`, `model/`.
2. Copy `params.json` and a few degraded/clean example image pairs to `degradation/`.
3. Copy all experiment logs (`logs/exp_*.log`), baseline log, and final model log to `experiments/`.
4. Copy `results.tsv` to `results/`.
5. Copy `train.py` and `run_experiments.py` to `model/` and `experiments/` respectively.

**IMPORTANT**: `exp*/` and `logs/` are in `.gitignore`. Do NOT commit them. They stay local.

### Writing the summary

Create `summarize/expN_summarize.md` (this folder IS tracked by git):

Sections to include:
1. **Degradation Pipeline** — the degradation definition, function names, severities
2. **Model Architecture** — brief description, key hyperparameters
3. **Training Setup** — data, batch size, time budget, optimizer
4. **Experiment Results** — table per round with PSNR, delta, VRAM
5. **Key Findings** — what worked, what didn't, insights
6. **Final Model** — best configuration and metrics
7. **Infrastructure** — GPU count, total compute time
8. **File Index** — folder structure map

Commit the summary: `git add summarize/ && git commit -m "docs: add expN summary"`

## Current default hyperparameters (train.py)

```
PARAMS_PATH = None           # random degradation (blind restoration)
TRAIN_SHARDS = "datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar"

VAL_DIRS = [                 # 6 benchmarks, 737 full images total
    "datasets/Set5/GTmod4", "datasets/Set14/GTmod4",
    "datasets/B100/GTmod4", "datasets/Urban100/GTmod4",
    "datasets/Manga109/GTmod4", "datasets/DIV2K/DIV2K_valid_HR",
]
VAL_PARAMS_PATH = "params.json"  # specific degradation to test against

# Model (SwinIR)
EMBED_DIM = 64
DEPTHS = (2, 2, 2, 2)
NUM_HEADS = (4, 4, 4, 4)
WINDOW_SIZE = 8
MLP_RATIO = 2

# Training
EPOCH_BUDGET = 1             # 1 epoch = 7,547 steps ≈ 44 min
TIME_BUDGET = 600            # fallback when EPOCH_BUDGET=0
CHECKPOINT_INTERVAL = 4      # validate/save N times per epoch
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100
LR_SCHEDULE = "cosine"
AMP_DTYPE = "bfloat16"
```

## Ideas to explore

**Architecture**:
- Scale SwinIR: `EMBED_DIM` (64→96→128), `DEPTHS` (2→4→6 per stage)
- Reduce/increase `WINDOW_SIZE` (trade compute vs receptive field)
- Replace SwinIR entirely: U-Net, NAFNet, Restormer, HAT
- Channel attention, spatial attention, cross-attention between stages

**Loss**:
- L1 vs MSE vs Huber (delta tuning)
- Multi-scale loss (compute loss at multiple resolutions)
- Edge-aware loss (Sobel gradient difference)
- Frequency loss (FFT L1 on magnitude spectrum)
- Perceptual loss (VGG features, if added to deps)

**Optimization**:
- LR sweep: 5e-4, 1e-3, 2e-3, 5e-3
- Schedule: cosine vs linear vs step vs onecycle
- Warmup ratio tuning
- Gradient clipping threshold
- Weight decay sweep

**Data**:
- `PARAMS_PATH = None` (blind) vs specific degradation (specialist)
- Augmentation: horizontal flip, vertical flip, rotation (note: degradation order matters, augmentation goes BEFORE degradation in the dataloader pipeline)

**Training**:
- Batch size vs LR tradeoff (larger batch → higher LR)
- EMA of model weights for evaluation
- Gradient accumulation for larger effective batch
