# Test metrics (fixed LQ, VGG LPIPS, Gaussian SSIM)

## Full test

| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| AutoIR | DIV2K | 24 | 24.968 | 0.7326 | 0.4050 |
| AutoIR | LSDIR | 24 | 22.956 | 0.6756 | 0.4009 |
| DFPIR-original | DIV2K | 24 | 23.993 | 0.7207 | 0.3807 |
| DFPIR-original | LSDIR | 24 | 22.029 | 0.6762 | 0.3752 |
| DFPIR-ft | DIV2K | 24 | 25.509 | 0.7426 | 0.3680 |
| DFPIR-ft | LSDIR | 24 | 23.489 | 0.6981 | 0.3635 |

## DIV2K overlap-excluded (0801-0810 removed)

| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| AutoIR | DIV2K | 24 | 24.963 | 0.7330 | 0.4059 |
| AutoIR | LSDIR | 24 | 22.956 | 0.6756 | 0.4009 |
| DFPIR-original | DIV2K | 24 | 24.014 | 0.7215 | 0.3802 |
| DFPIR-original | LSDIR | 24 | 22.029 | 0.6762 | 0.3752 |
| DFPIR-ft | DIV2K | 24 | 25.501 | 0.7432 | 0.3685 |
| DFPIR-ft | LSDIR | 24 | 23.489 | 0.6981 | 0.3635 |

## Paired vs AutoIR (full test)

- DFPIR-original vs AutoIR / DIV2K: ΔPSNR=-0.975 dB [-4.241, +2.368], wins 15/24, worst -22.286
- DFPIR-original vs AutoIR / LSDIR: ΔPSNR=-0.927 dB [-3.860, +1.996], wins 15/24, worst -21.971
- DFPIR-ft vs AutoIR / DIV2K: ΔPSNR=+0.541 dB [-0.262, +1.371], wins 17/24, worst -3.912
- DFPIR-ft vs AutoIR / LSDIR: ΔPSNR=+0.533 dB [-0.169, +1.279], wins 17/24, worst -3.405