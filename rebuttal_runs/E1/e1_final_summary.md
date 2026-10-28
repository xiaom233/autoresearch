# E1 rule vs AutoIR (fixed LQ)

## Full test

| Controller | Dataset | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| AutoIR | DIV2K | 24.968 | 0.7326 | 0.4050 |
| AutoIR | LSDIR | 22.956 | 0.6756 | 0.4009 |
| rule | DIV2K | 24.272 | 0.7124 | 0.4237 |
| rule | LSDIR | 22.131 | 0.6491 | 0.4231 |

## DIV2K support-excluded

| Controller | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| AutoIR | 24.969 | 0.7324 | 0.4049 |
| rule | 24.272 | 0.7120 | 0.4238 |

## DIV2K support+calibration-excluded

| Controller | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| AutoIR | 24.961 | 0.7324 | 0.4059 |
| rule | 24.283 | 0.7127 | 0.4241 |

## Paired rule − AutoIR

- DIV2K: ΔPSNR=-0.696 [-3.427,+2.275], wins 8/24, worst -20.527, best +23.003
- LSDIR: ΔPSNR=-0.825 [-3.410,+1.778], wins 8/24, worst -21.208, best +18.969