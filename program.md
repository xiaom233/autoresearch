# autoresearch — Image Restoration

This is an experiment to have the LLM do its own image restoration research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current dev.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context, x_distortion reference, severity mappings.
   - `prepare.py` — data prep, WebDataset packaging, degradation dataloader. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, training loop.
   - `x_distortion/` — degradation library (35 functions × 5 severities). Read-only.
4. **Verify data exists**: Check that `datasets/DIV2K/DIV2K_train_HR_wds/` contains WebDataset tar shards. If not, tell the human to run `uv run prepare.py`.
5. **Run the baseline**: The first experiment MUST use `PARAMS_PATH = None` (random degradation from blur/noise/compression). This establishes the blind restoration baseline.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row (see Logging results below). The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, loss function, etc.
- Set `PARAMS_PATH` in `train.py`:
  - `None` = random degradation per sample (blind restoration — baseline/default)
  - `"path/to/params.json"` = fixed degradation from skill output (targeted restoration)

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data preparation, WebDataset packaging, degradation dataloader, and random pipeline generation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify `x_distortion/`. It is the reference degradation library.
- Modify the evaluation harness. The `val_psnr_db` computed in `train.py` is the ground truth metric.

**The goal is simple: get the highest val_psnr_db.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful PSNR gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1 dB PSNR improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1 dB improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run MUST establish the baseline. Set `PARAMS_PATH = None` (random degradation), `TIME_BUDGET = 600`, and run `uv run train.py > run.log 2>&1`. Record the result as the baseline in `results.tsv`.

## Model architecture guidance

The `RestoreNet` class in `train.py` is a placeholder. You should replace it with a real restoration architecture. Good starting points:

- **U-Net** — encoder-decoder with skip connections. Proven for restoration. Start with 3-4 levels, base channels 32-64.
- **ResNet** — residual blocks, no downsampling. Fast, simple, preserves detail.
- **NAFNet** — simple but SOTA. Stack of NAFBlocks (LayerNorm + simplified channel attention).
- **SwinIR** — Swin Transformer for restoration. Strong but heavier.

Start simple. A 3-level U-Net with ~2M params trains quickly and gives a solid baseline. Add complexity only when you have evidence it helps.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_psnr_db:       26.33
best_train_psnr:   30.12
best_val_psnr:     26.45
final_loss:        0.030426
training_seconds:  600.1
total_seconds:     625.9
peak_vram_mb:      4520.2
num_steps:         5652
num_params_M:      2.0
total_imgs:        90432
batch_size:        16
learning_rate:     0.001
```

Note that the script is configured to always stop after 10 minutes, so depending on the computing platform the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_psnr_db:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_psnr_db	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_psnr_db achieved (e.g. 26.33) — use 0.0 for crashes
3. peak memory in GB, round to .1f (e.g. 4.4 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_psnr_db	memory_gb	status	description
a1b2c3d	26.33	4.4	keep	baseline: U-Net, random degradation
b2c3d4e	26.89	4.5	keep	add residual blocks to encoder
c3d4e5f	26.10	4.4	discard	switch L1→MSE loss
d4e5f6g	0.00	0.0	crash	double channels (OOM)
```

**PSNR interpretation**: Higher is better. The metric is in dB. Typical restoration PSNR ranges: 25-30 dB (moderate), 30-35 dB (good), 35+ dB (excellent). A 0.5 dB improvement is meaningful.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may11` or `autoresearch/may11-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Read `train.py` for full context.
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. git commit.
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
6. Read out the results: `grep "^val_psnr_db:\|^peak_vram_mb:\|^num_steps:\|^num_params_M:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git).
9. If val_psnr_db improved (higher), you "advance" the branch, keeping the git commit.
10. If val_psnr_db is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Ideas to explore

When you're stuck, consider these directions:

**Architecture**:
- Deeper/narrower vs shallower/wider U-Net
- Residual blocks, dense blocks, attention gates
- Transformer-based (SwinIR, Restormer-style)
- Multi-scale feature extraction

**Loss function**:
- L1 vs MSE vs Huber
- Perceptual loss (VGG features) — if available in deps
- Edge-aware loss (gradient difference)
- Frequency-domain loss (FFT)

**Optimization**:
- LR schedules (cosine, step, cyclic)
- Warmup length tuning
- Gradient clipping
- Mixed precision settings

**Data**:
- PARAMS_PATH = None for blind restoration
- Target specific degradation types for specialist models
- Augmentation (flips, rotations — but note degradation order sensitivity)

**Training**:
- Batch size vs LR tradeoff
- EMA of model weights
- Multi-stage training (coarse → fine)
