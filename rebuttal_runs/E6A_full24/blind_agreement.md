# E6-A full24 blind-ID agreement (rep1 vs rep2)

| Scenario | GT steps | rep1 | rep2 | exact | type-only | pos type/spec |
|---|---:|---|---|:--:|:--:|:--:|
| blind_0001 | 1 | noise_poisson:5 + quantization_hist:2 + brightness_darken_gamma_RGB:2 | noise_poisson:5 | N | N | 1/1 of 1 |
| blind_0002 | 1 | noise_gaussian_YCrCb:2 | noise_gaussian_YCrCb:2 | Y | Y | 1/1 of 1 |
| blind_0003 | 1 | blur_glass:2 | blur_glass:2 | Y | Y | 1/1 of 1 |
| blind_0004 | 1 | blur_gaussian:1 | blur_motion:1 | N | N | 0/0 of 1 |
| blind_0005 | 1 | noise_impulse:5 | noise_impulse:5 | Y | Y | 1/1 of 1 |
| blind_0006 | 1 | compression_jpeg:1 | compression_jpeg:1 | Y | Y | 1/1 of 1 |
| blind_0007 | 1 | noise_speckle:1 | noise_speckle:1 | Y | Y | 1/1 of 1 |
| blind_0008 | 1 | saturate_weaken_HSV:4 | saturate_weaken_HSV:4 | Y | Y | 1/1 of 1 |
| blind_0009 | 2 | noise_speckle:1 + saturate_strengthen_YCrCb:3 | noise_gaussian_YCrCb:4 + saturate_strengthen_YCrCb:1 | N | N | 1/0 of 2 |
| blind_0010 | 2 | blur_gaussian:1 + brightness_darken_shfit_HSV:4 | blur_gaussian:1 + brightness_darken_gamma_HSV:5 | N | N | 1/1 of 2 |
| blind_0011 | 2 | blur_gaussian:4 | noise_speckle:1 + blur_gaussian:4 | N | N | 0/0 of 1 |
| blind_0012 | 2 | noise_gaussian_RGB:1 + blur_motion:5 + compression_jpeg:3 | blur_motion:5 + compression_jpeg:3 | N | N | 0/0 of 2 |
| blind_0013 | 2 | noise_speckle:2 + compression_jpeg:1 | noise_speckle:3 + compression_jpeg:2 | N | Y | 2/0 of 2 |
| blind_0014 | 2 | brightness_brighten_shfit_HSV:3 + compression_jpeg:2 | brightness_brighten_shfit_HSV:3 + compression_jpeg:2 | Y | Y | 2/2 of 2 |
| blind_0015 | 2 | blur_motion:1 + brightness_darken_shfit_RGB:3 | blur_motion:1 + brightness_darken_shfit_RGB:3 | Y | Y | 2/2 of 2 |
| blind_0016 | 2 | blur_gaussian:4 + compression_jpeg:2 | blur_motion:4 + compression_jpeg:1 | N | N | 1/0 of 2 |
| blind_0017 | 3 | blur_gaussian:4 + compression_jpeg:4 | blur_gaussian:3 + compression_jpeg:4 | N | Y | 2/1 of 2 |
| blind_0018 | 3 | contrast_strengthen_stretch:4 + compression_jpeg:5 | contrast_strengthen_stretch:4 + compression_jpeg:5 | Y | Y | 2/2 of 2 |
| blind_0019 | 3 | blur_lens:4 + compression_jpeg:4 + saturate_weaken_HSV:5 | blur_lens:4 + compression_jpeg:4 + saturate_weaken_HSV:5 | Y | Y | 3/3 of 3 |
| blind_0020 | 3 | blur_gaussian:4 + contrast_weaken_stretch:4 | blur_gaussian:4 + brightness_darken_gamma_RGB:1 + contrast_weaken_stretch:5 | N | N | 1/1 of 2 |
| blind_0021 | 3 | blur_lens:4 + brightness_brighten_shfit_RGB:1 + compression_jpeg:4 | blur_gaussian:3 + brightness_brighten_shfit_RGB:1 + compression_jpeg:4 | N | N | 2/2 of 3 |
| blind_0022 | 3 | blur_glass:3 + compression_jpeg:1 | blur_gaussian:3 + compression_jpeg:1 | N | N | 1/1 of 2 |
| blind_0023 | 3 | blur_glass:4 + contrast_weaken_scale:1 + compression_jpeg:4 | blur_glass:5 + compression_jpeg:4 + contrast_weaken_scale:1 | N | N | 1/0 of 3 |
| blind_0024 | 3 | brightness_darken_gamma_RGB:3 + saturate_strengthen_YCrCb:4 | saturate_strengthen_HSV:2 | N | N | 0/0 of 1 |

Chain exact: **10/24**; type-only: 12/24; steps equal: 19/24.

Position-level type: 0.683 (41 positions); spec: 0.537.

rep1 vs GT chain exact: 9/24; rep2: 15/24.
