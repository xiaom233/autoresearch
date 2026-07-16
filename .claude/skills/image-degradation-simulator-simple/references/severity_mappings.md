# Severity to Actual Parameter Mappings

Each x_distortion function maps abstract severity levels (1–5) to specific physical parameters. This reference documents every mapping so params.json can accurately record both the abstract severity and the actual parameters used.

## Blur

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `blur_gaussian` | sigma | 1 | 2 | 3 | 4 | 5 |
| `blur_motion` | (radius, sigma) | (5, 3) | (10, 5) | (15, 7) | (15, 9) | (20, 12) |
| `blur_glass` | (sigma, shift, iter) | (0.7, 1, 1) | (0.9, 2, 1) | (1.2, 2, 2) | (1.4, 3, 2) | (1.6, 4, 2) |
| `blur_lens` | radius | 2 | 3 | 4 | 6 | 8 |
| `blur_zoom` | zoom_range | [1, 1.03, step=0.02] | [1, 1.06, step=0.02] | [1, 1.10, step=0.02] | [1, 1.15, step=0.02] | [1, 1.21, step=0.02] |
| `blur_jitter` | shift | 1 | 2 | 3 | 4 | 5 |

Note: `blur_motion` also applies a random angle uniformly sampled from [-90, 90].

## Noise

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `noise_gaussian_RGB` | sigma | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 |
| `noise_gaussian_YCrCb` | sigma_l | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 |
| `noise_gaussian_YCrCb` | sigma_r (=sigma_l × factor) | 0.05 | 0.087 | 0.133 | 0.188 | 0.252 |
| `noise_gaussian_YCrCb` | sigma_b (=sigma_l × factor) | 0.05 | 0.087 | 0.133 | 0.188 | 0.252 |
| `noise_speckle` | sigma (scale) | 0.14 | 0.21 | 0.28 | 0.35 | 0.42 |
| `noise_spatially_correlated` | sigma | 0.08 | 0.11 | 0.14 | 0.18 | 0.22 |
| `noise_poisson` | c | 80 | 60 | 40 | 25 | 15 |
| `noise_impulse` | amount | 0.01 | 0.03 | 0.05 | 0.07 | 0.10 |

## Compression

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `compression_jpeg` | quality | 25 | 18 | 12 | 8 | 5 |
| `compression_jpeg_2000` | quality (dB) | 29 | 27.5 | 26 | 24.5 | 23 |

## Brightness — Brighten

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `brightness_brighten_shfit_HSV` | c (V shift) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| `brightness_brighten_shfit_RGB` | c (RGB shift) | 0.1 | 0.15 | 0.2 | 0.27 | 0.35 |
| `brightness_brighten_gamma_HSV` | gamma | 0.7 | 0.58 | 0.47 | 0.36 | 0.25 |
| `brightness_brighten_gamma_RGB` | gamma | 0.8 | 0.7 | 0.6 | 0.45 | 0.3 |

## Brightness — Darken

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `brightness_darken_shfit_HSV` | c (V shift) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| `brightness_darken_shfit_RGB` | c (RGB shift) | 0.1 | 0.15 | 0.2 | 0.27 | 0.35 |
| `brightness_darken_gamma_HSV` | gamma | 1.5 | 1.8 | 2.2 | 2.7 | 3.5 |
| `brightness_darken_gamma_RGB` | gamma | 1.4 | 1.7 | 2.1 | 2.6 | 3.2 |
| `brightness_vignette` | gamma | 0.5 | 0.875 | 1.25 | 1.625 | 2.0 |

## Contrast

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `contrast_strengthen_scale` | scale | 1.4 | 1.7 | 2.1 | 2.6 | 4.0 |
| `contrast_strengthen_stretch` | scale | 2.0 | 4.0 | 6.0 | 8.0 | 10.0 |
| `contrast_weaken_scale` | scale | 0.75 | 0.6 | 0.45 | 0.3 | 0.2 |
| `contrast_weaken_stretch` | scale | 1.0 | 0.9 | 0.8 | 0.6 | 0.4 |

## Saturation

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `saturate_strengthen_HSV` | scale | 3.0 | 6.0 | 12.0 | 20.0 | 64.0 |
| `saturate_strengthen_YCrCb` | scale | 2.0 | 3.0 | 5.0 | 8.0 | 16.0 |
| `saturate_weaken_HSV` | scale | 0.7 | 0.55 | 0.4 | 0.2 | 0.0 |
| `saturate_weaken_YCrCb` | scale | 0.6 | 0.4 | 0.2 | 0.1 | 0.0 |

## Other distortions

| Function | Param | Sev=1 | Sev=2 | Sev=3 | Sev=4 | Sev=5 |
|----------|-------|-------|-------|-------|-------|-------|
| `oversharpen` | amount | 2 | 2.8 | 4 | 6 | 8 |
| `pixelate` | scale_factor | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 |
| `quantization_otsu` | num_classes | 15 | 11 | 8 | 5 | 3 |
| `quantization_median` | num_classes | 20 | 15 | 10 | 6 | 3 |
| `quantization_hist` | num_classes | 24 | 16 | 8 | 6 | 4 |
