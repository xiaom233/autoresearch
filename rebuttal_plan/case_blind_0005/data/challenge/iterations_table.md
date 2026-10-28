| # | 轮次 | 假设管线 | 测试结果 | 决策 | 归因 |
|---:|:--:|---|---|---|---|
| 1 | R0 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5` | PSNR 12.21 / CI 7/10 | REJECT | L：初始 global 假设；PSNR 12.21 / CI 7/10 |
| 2 | R0 | `brightness_brighten_shfit_HSV:2` | PSNR 12.79 / CI 6/10 | REJECT | L：brightness alone；12.79 / 6/10 |
| 3 | R0 | `brightness_brighten_gamma_HSV:2` | PSNR 13.07 / CI 5/10 | REJECT | L：gamma brightness；13.07 / 5/10 |
| 4 | R0 | `saturate_strengthen_HSV:2` | PSNR 12.33 / CI 7/10 | REJECT | L：saturation alone；12.33 / 7/10 |
| 5 | R0 | `contrast_strengthen_stretch:3` | PSNR 10.73 / CI 3/10 | REJECT | L：contrast stretch；10.73 / 3/10 |
| 6 | R0 | `noise_impulse:2` | PSNR 12.86 / CI 8/10 | REJECT | R+L：impulse alone；12.86 / 8/10（规则要求扫 noise） |
| 7 | R0 | `saturate_strengthen_YCrCb:3` | PSNR 8.41 / CI 7/10 | REJECT | L：saturation YCrCb；8.41 / 7/10 |
| 8 | R0 | `brightness_brighten_shfit_RGB:2,saturate_strengthen_HSV:3` | PSNR 12.17 / CI 7/10 | REJECT | L：RGB brightness + sat；12.17 / 7/10 |
| 9 | R0 | `noise_impulse:1` | PSNR 13.54 / CI 8/10 | REJECT | R：impulse severity 1（severity 扫描） |
| 10 | R0 | `noise_impulse:3` | PSNR 12.28 / CI 9/10 | REJECT | R：impulse severity 3（severity 扫描） |
| 11 | R0 | `noise_impulse:3,brightness_brighten_shfit_HSV:1` | PSNR 12.14 / CI 10/10 | REJECT | L：impulse:3 + brightness:1 反向组合；12.14 / 10/10 |
| 12 | R0 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | PSNR 12.13 / CI 10/10 | REJECT | L：brightness:1 + impulse:3；12.13 / 10/10 |
| 13 | R0 | `noise_impulse:3,brightness_brighten_shfit_HSV:2` | PSNR 11.52 / CI 9/10 | REJECT | L：brightness severity 2；11.52 / 9/10 |
| 14 | R1 | `brightness_brighten_gamma_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：gamma+impulse；WEAK |
| 15 | R1 | `saturate_strengthen_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：saturation+impulse；WEAK |
| 16 | R1 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：baseline signal re-verify；WEAK |
| 17 | R2 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:3,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：brightness+sat3+impulse；WEAK |
| 18 | R2 | `brightness_brighten_shfit_HSV:1,noise_impulse:4` | signal WEAK | TRY_ALTERNATIVE | L：impulse:4+brightness；WEAK |
| 19 | R2 | `noise_impulse:4,brightness_brighten_shfit_HSV:1` | signal WEAK | TRY_ALTERNATIVE | L：impulse:4+brightness 反向；WEAK |
| 20 | R2 | `noise_impulse:3,brightness_brighten_shfit_HSV:1` | signal WEAK | TRY_ALTERNATIVE | L：impulse:3+brightness 反向；WEAK |
| 21 | R2 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:2,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：brightness+sat2+impulse；WEAK |
| 22 | R2 | `brightness_brighten_shfit_HSV:1,noise_impulse:3` | signal WEAK | TRY_ALTERNATIVE | L：baseline 复核；WEAK |
| 23 | R4 | `saturate_strengthen_HSV:5,noise_impulse:3` | PSNR 11.11 / CI 7/10 | REJECT | L：sat5+impulse；11.11 / 7/10 |
| 24 | R4 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5` | PSNR 12.21 / CI 7/10 | REJECT | L：brightness+sat5；12.21 / 7/10 |
| 25 | R4 | `brightness_brighten_shfit_HSV:1,saturate_strengthen_HSV:5,noise_impulse:3` | PSNR 11.09 / CI 8/10 | REJECT | L：brightness+sat5+impulse；11.09 / 8/10 |
| 26 | R4 | `compression_jpeg:1` | PSNR 13.55 / CI 5/10 | REJECT | L：JPEG-only；13.55 / 5/10 |
| 27 | R4 | `blur_gaussian:1,noise_impulse:3,brightness_brighten_shfit_HSV:1` | PSNR 11.71 / CI 6/10 | REJECT | L：blur+impulse+brightness；11.71 / 6/10 |
| 28 | R4 | `quantization_hist:1,brightness_brighten_shfit_HSV:1,noise_impulse:3` | PSNR 8.42 / CI 3/10 | REJECT | L：quantization+brightness+impulse；8.42 / 3/10 |
