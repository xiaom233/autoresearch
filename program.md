# Program: Image Restoration Research

## Overview

You are an autonomous image restoration researcher. Your goal: given a degraded target image, identify the degradation, simulate it on clean data, and train a restoration model.

## The Pipeline

### Step 1: Analyze the target degraded image

Given a path to a degraded image (`target.png`), use the `image-degradation-simulator` skill to identify what degradations are present and their severity. The skill will:

- Visually inspect and quantitatively analyze the target
- Iteratively hypothesize, simulate, and compare degradation pipelines
- Output `params.json` — the final degradation pipeline parameters

Example `params.json` output:
```json
{
  "pipeline": [
    {"step": 1, "function": "blur_gaussian", "severity": 3},
    {"step": 2, "function": "noise_gaussian_RGB", "severity": 2}
  ]
}
```

### Step 2: Verify the degradation simulation

Use `prepare.py` to test that the identified degradation pipeline produces realistic results:

```bash
# Test the dataloader with the identified params
uv run prepare.py --demo
```

Or programmatically:
```python
from prepare import make_dataloader_restoration
loader = make_dataloader_restoration(
    params_path='path/to/params.json',
    shards_url='datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar',
    batch_size=4,
    endless=False,
)
for inputs, targets, epoch in loader:
    # inputs: degraded, targets: clean
    # Verify inputs look visually similar to the target degradation
    break
```

### Step 3: Prepare the dataset (one-time)

If not already done, run the full data preparation:

```bash
uv run prepare.py
```

This crops DIV2K HR images into 256×256 patches and packages them as WebDataset tar shards.

## Key Files

| File | Role | Editable |
|------|------|----------|
| `prepare.py` | Data prep, WebDataset packaging, degradation dataloader, evaluation | Read-only (do not modify) |
| `train.py` | Restoration model architecture, training loop | Agent modifies this |
| `x_distortion/` | Degradation library (35 types × 5 severities) | Read-only |
| `pyproject.toml` | Dependencies | Read-only |

## x_distortion Quick Reference

```python
from x_distortion import add_distortion, distortions_dict

# Apply a single distortion
img = add_distortion(img, severity=3, distortion_name='blur_gaussian')

# Apply a pipeline (in order — order matters!)
for func, sev in [('blur_gaussian', 3), ('noise_impulse', 1)]:
    img = add_distortion(img, severity=sev, distortion_name=func)
```

Available functions by category:
- **blur**: gaussian, motion, glass, lens, zoom, jitter
- **noise**: gaussian_RGB, gaussian_YCrCb, speckle, spatially_correlated, poisson, impulse
- **compression**: jpeg, jpeg_2000
- **brighten/darken**: shift_HSV, shift_RGB, gamma_HSV, gamma_RGB
- **contrast**: strengthen_scale, strengthen_stretch, weaken_scale, weaken_stretch
- **saturation**: strengthen_HSV, strengthen_YCrCb, weaken_HSV, weaken_YCrCb
- **other**: oversharpen, pixelate, quantization_otsu/median/hist

Degradation order is order-sensitive. `blur → noise` produces clean noise on a soft image; `noise → blur` smears the noise grain. The degradation-simulator skill determines the correct order.

## Evaluation

The restoration model is evaluated by comparing the restored output against clean references. Standard image quality metrics (PSNR, SSIM, LPIPS) apply here since we have paired (degraded, clean) data from simulation.

## Notes

- The DataLoader applies degradation on-the-fly — no need to pre-compute degraded images
- WebDataset shards enable efficient streaming I/O with shuffle buffer
- Each epoch re-shuffles shard order for robust training
- All images are normalized to [0, 1] float32, channel-first (C, H, W)
