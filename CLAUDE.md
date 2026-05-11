# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # install dependencies
uv run prepare.py                # crop DIV2K + package WebDataset shards (one-time)
uv run prepare.py --demo         # test the degradation dataloader
uv run train.py                  # train restoration model (10 min time budget)
uv run train.py > run.log 2>&1   # training with log capture (agent mode)
```

## Architecture

This is an **autonomous image restoration research** project. The agent edits `train.py` to maximize `val_psnr_db` under a fixed 10-minute time budget.

### File roles

| File | Role | Mutable |
|------|------|---------|
| `prepare.py` | Data prep (crop DIV2K→patches), WebDataset packaging, degradation dataloader, random pipeline generator | **Read-only** |
| `train.py` | Model (`RestoreNet`), optimizer, training loop, evaluation harness | **Agent edits** |
| `x_distortion/` | 35 degradation functions × 5 severities, numpy uint8 RGB in/out | Read-only |
| `program.md` | Agent behavior specification, pipeline docs, experiment loop instructions | Read-only |
| `pyproject.toml` | Dependencies (torch, opencv, webdataset, numba, scikit-image, etc.) | Read-only |

### Data flow

1. **prepare.py**: 800 DIV2K HR images → 120,765 patches (256×256) → 121 WebDataset `.tar` shards
2. **Degradation** (in dataloader, per sample):
   - `PARAMS_PATH = None`: random pipeline — blur/noise/compression, 1/2/3 degradations (0.33 each), severity 1–5
   - `PARAMS_PATH = "params.json"`: fixed pipeline from `image-degradation-simulator` skill output
3. **train.py**: streams WDS shards, applies degradation on-the-fly, trains `RestoreNet`, evaluates PSNR on 6 benchmark sets (428 full images)
4. **results.tsv**: tab-separated log (commit, val_psnr_db, memory_gb, status, description)

### Key design decisions

- **Training degradation is per-sample random** when `PARAMS_PATH=None`. The `get_pipeline()` closure in `make_dataloader_restoration` calls `generate_random_pipeline()` for each sample, giving the model diverse degradations to learn blind restoration.
- **Validation uses full images** (not patches). `ValDataset` iterates one image at a time due to variable resolutions. PSNR is averaged per-image across all 428 validation samples.
- **SwinIR simplified**: no upsampling (in/out same resolution), no relative position bias, no DropPath, no checkpointing. Window attention uses `F.scaled_dot_product_attention` which dispatches to flash-attention on supported GPUs.
- **Time budget enforcement**: training stops when `total_training_time >= TIME_BUDGET`, with first 10 steps excluded from timing (compilation warmup).
- **`prepare.py` import side-effect**: importing `prepare` triggers the module-level `__main__` block unless guarded. The dataloader functions are safe to import; use `from prepare import make_dataloader_restoration` directly.
