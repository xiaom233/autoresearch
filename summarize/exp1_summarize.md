# Experiment 1 — Image Restoration Auto-Research

**Date:** 2026-05-11 to 2026-05-12
**Branch:** `autoresearch/may11`
**Goal:** Maximize validation PSNR for blind image restoration under a fixed degradation pipeline.

---

## 1. Degradation Pipeline

### Definition (`params.json`)

A 3-step fixed degradation pipeline targeting a specific corruption profile:

| Step | Function | Severity | Category |
|------|----------|----------|----------|
| 1 | `blur_motion` | 3/5 | Motion blur |
| 2 | `noise_spatially_correlated` | 2/5 | Correlated noise |
| 3 | `compression_jpeg` | 4/5 | JPEG compression |

### Degradation Simulation

The pipeline is applied on-the-fly during data loading using `x_distortion.add_distortion()`. Each function operates on numpy uint8 RGB arrays. Example degraded samples are in `degradation/sample*_degraded.png` alongside their clean counterparts.

### x_distortion Library

The project includes 35 degradation functions across 5 categories, each with severities 1–5:
- **Blur**: gaussian, motion, glass, lens, zoom, jitter
- **Noise**: gaussian_RGB, gaussian_YCrCb, speckle, spatially_correlated, poisson, impulse
- **Compression**: jpeg, jpeg_2000
- **Brightness/Contrast/Saturation**: various HSV/RGB manipulations
- **Other**: pixelate, oversharpen, quantization, shuffle_pixels

---

## 2. Model Architecture

**RestoreNet** — simplified SwinIR (~455K parameters):

```
conv_first (3→64) → RSTB×4 → LayerNorm → conv_after_body → conv_last (64→3)
                       │                                    │
                       └── SwinBlock×2 each ────────────────┘
                            WindowSDPA (W/SW-MSA) + GELU MLP   (global residual)
```

- `EMBED_DIM=64`, `DEPTHS=(2,2,2,2)`, `NUM_HEADS=(4,4,4,4)`, `WINDOW_SIZE=8`, `MLP_RATIO=2`
- Window-based multi-head attention via `F.scaled_dot_product_attention` (flash-attention backend)
- No upsampling — same-resolution input/output restoration
- `torch.compile(model, dynamic=False)` for training

---

## 3. Training Setup

| Parameter | Value |
|-----------|-------|
| Training data | 120,765 patches (256×256) from DIV2K, 121 WebDataset shards |
| Validation | 5 images × 6 benchmarks = 30 images (Set5, Set14, B100, Urban100, Manga109, DIV2K) |
| Time budget | 600 seconds per experiment |
| Batch size | 16 |
| Optimizer | AdamW (betas 0.9/0.999, weight decay 1e-4) |
| LR schedule | Cosine decay with 100-step linear warmup |
| Loss | L1 (baseline) |
| Precision | bfloat16 autocast |

---

## 4. Experiment Results (32 experiments)

### Baseline

| Metric | Value |
|--------|-------|
| Baseline PSNR | 20.27 dB |
| Parameters | 0.5M |
| VRAM | 13.8 GB |
| Steps | 1735 |

### Round 1: Architecture Variants

| # | Change | PSNR | Δ | VRAM | Notes |
|---|--------|------|---|------|-------|
| 1 | `EMBED_DIM=96` | 20.62 | +0.35 | 20.2 GB | **Best architecture** |
| 2 | `EMBED_DIM=128` | 20.26 | -0.01 | 26.8 GB | Too wide, fewer steps |
| 3 | `DEPTHS=(4,4,4,4)` | 20.43 | +0.16 | 26.1 GB | Deeper helps but costly |
| 4 | `DEPTHS=(6,6,6,6)` | 20.37 | +0.10 | 38.4 GB | Diminishing returns |
| 5 | `WINDOW_SIZE=16` | 20.42 | +0.15 | 13.6 GB | Larger receptive field |
| 6 | `EMBED_DIM=48` | 20.39 | +0.12 | 12.9 GB | Narrower but faster |
| 7 | `MLP_RATIO=4` | 20.09 | -0.18 | 17.6 GB | Wider MLP hurts |
| 8 | `NUM_HEADS=(8,8,8,8)` | 20.41 | +0.14 | 15.0 GB | Mild improvement |

### Round 2: Loss Functions

| # | Change | PSNR | Δ | Notes |
|---|--------|------|---|-------|
| 9 | MSE loss | 20.34 | +0.07 | Slightly worse than L1 |
| 10 | Huber (δ=0.1) | 20.54 | +0.27 | Robust to outliers |
| 11 | L1+MSE (1:0.3) | 20.47 | +0.20 | Mild combination |
| 12 | L1+FFT freq loss | 20.48 | +0.21 | Frequency domain helps |
| 13 | L1+Sobel edge (0.1) | CRASH | — | Autocast incompatibility |
| 14 | L1+Sobel edge (0.5) | CRASH | — | Autocast incompatibility |
| 15 | Huber (δ=0.5) | 20.22 | -0.05 | Too smooth |
| 16 | L1+MSE (1:1) | 20.57 | +0.30 | **Best loss combo** |

### Round 3: Optimization

| # | Change | PSNR | Δ | Notes |
|---|--------|------|---|-------|
| 17 | LR=5e-4 | 20.36 | +0.09 | Too slow to converge |
| 18 | LR=2e-3 | 20.49 | +0.22 | Higher LR helps |
| 19 | **Constant LR** | **20.71** | **+0.44** | **BEST OVERALL** |
| 20 | Warmup=500 steps | 20.27 | 0.00 | No benefit |
| 21 | Weight decay=1e-3 | 20.54 | +0.27 | Stronger regularization |
| 22 | No weight decay | 20.32 | +0.05 | Mildly worse |
| 23 | Adam β=(0.95,0.999) | 20.27 | 0.00 | No benefit |
| 24 | Gradient clip=1.0 | 20.60 | +0.33 | Prevents instability |

### Round 4: Combinations

| # | Change | PSNR | Δ | Notes |
|---|--------|------|---|-------|
| 25 | EMBED_DIM=96 + DEPTHS=(4,4,4,4) | 20.05 | -0.22 | Too large, few steps |
| 26 | EMBED_DIM=96 + MSE | 20.36 | +0.09 | Worse than L1 alone |
| 27 | EMBED_DIM=128 + WINDOW_SIZE=16 | 20.45 | +0.18 | Moderate |
| 28 | DEPTHS=(4,4,4,4) + MSE | 20.20 | -0.07 | Underperforms |
| 29 | EMBED_DIM=96 + DEPTHS=(4,4,4,4) + MSE | 20.20 | -0.07 | Too large |
| 30 | DEPTHS=(6,6,6,6) + EMBED_DIM=48 | 20.49 | +0.22 | Deep + narrow tradeoff |
| 31 | EMBED_DIM=128 + DEPTHS=(1,1,1,1) | 20.38 | +0.11 | Wide + shallow |
| 32 | EMBED_DIM=80 + L1+edge | CRASH | — | Edge loss crash |

### Bonus

| Change | PSNR | Δ | Notes |
|--------|------|---|-------|
| EMBED_DIM=96 + constant LR | 20.43 | +0.16 | Worse than either alone |

---

## 5. Key Findings

1. **Constant LR > cosine decay for short training.** With only 10 min (1730 steps), the model never converges — decaying the learning rate only reduces the effective step size. This is the single best improvement at **+0.44 dB**.

2. **Wider > deeper.** `EMBED_DIM=96` (+0.35 dB) improves capacity within the budget. Adding depth (`DEPTHS=(4,4,4,4)`) trades steps for params — the net effect is neutral or negative when combined.

3. **Combinations don't stack.** Each individual improvement helps (constant LR: +0.44, gradient clip: +0.33, L1+MSE: +0.30), but combining them gives only +0.34 (20.61 dB). The effects are overlapping rather than additive.

4. **Gradient clipping (max_norm=1.0)** prevents loss spikes and gives +0.33 dB.

5. **L1+MSE combined loss** outperforms pure L1 (+0.30 dB) and pure MSE (+0.07 dB).

6. **Sobel edge loss crashes** with bfloat16 autocast — the `F.conv2d` call inside the loss function is incompatible with the autocast context.

---

## 6. Final Model

Combined best findings into a single model:

| Configuration | |
|---------------|----|
| Architecture | RestoreNet (EMBED_DIM=64, DEPTHS=(2,2,2,2)) |
| LR Schedule | Constant (no decay) |
| Loss | L1 + MSE (1:1 weight) |
| Gradient clipping | max_norm = 1.0 |
| Time budget | 600 seconds |

| Metric | Value |
|--------|-------|
| **Final PSNR** | **20.61 dB** |
| Best single result | 20.71 dB (constant LR alone) |
| Improvement over baseline | +0.34 dB |
| Parameters | 0.5M |
| VRAM | 13.8 GB |
| Training steps | 1735 |
| Total time | 631 seconds |

**Note:** The combined model (20.61) underperforms the best single change (constant LR: 20.71 dB). This suggests the improvements are partially redundant — constant LR alone captures most of the available gain. For future experiments, start from constant LR and explore orthogonal directions.

---

## 7. Infrastructure

- **8× NVIDIA RTX A6000** (48 GB each), used in parallel
- **Total compute:** 4 rounds × 8 experiments × ~11 min = ~5.9 GPU-hours
- **Experiment runner:** `run_experiments.py` manages parallel execution with env var overrides
- **Data:** 121 WebDataset shards (120,765 patches from DIV2K 800 HR images)

---

## 8. File Index

```
exp1/
├── summarize.md              ← this file
├── degradation/
│   ├── pipeline.json         ← degradation definition
│   ├── sample1_clean.png     ← clean example (Set5/baby)
│   ├── sample1_degraded.png  ← degraded version
│   ├── sample2_clean.png     ← clean example (Set14/baboon)
│   ├── sample2_degraded.png
│   ├── sample3_clean.png     ← clean example (Urban100/img_001)
│   └── sample3_degraded.png
├── experiments/
│   ├── run_experiments.py    ← experiment orchestration script
│   ├── baseline_run.log      ← original baseline (VAL_COUNT=0)
│   ├── baseline_fast_val.log ← fast-val baseline (VAL_COUNT=5)
│   ├── final_model.log       ← final combined model training log
│   └── exp_*.log             ← all 32 experiment logs
├── results/
│   └── results.tsv           ← tab-separated results log
└── model/
    └── train.py              ← final training script (with env var support)
```
