# Test metrics (fixed LQ, VGG LPIPS, Gaussian SSIM)

## Full test

| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| AutoIR | DIV2K | 24 | 24.968 | 0.7326 | 0.4050 |
| AutoIR | LSDIR | 24 | 22.956 | 0.6756 | 0.4009 |
| DFPIR-original | DIV2K | 24 | 23.993 | 0.7207 | 0.3807 |
| DFPIR-original | LSDIR | 24 | 22.029 | 0.6762 | 0.3752 |
| DFPIR-ft-last | DIV2K | 24 | 25.367 | 0.7368 | 0.3694 |
| DFPIR-ft-last | LSDIR | 24 | 23.363 | 0.6931 | 0.3646 |
| DFPIR-ft-best | DIV2K | 24 | 25.651 | 0.7484 | 0.3667 |
| DFPIR-ft-best | LSDIR | 24 | 23.615 | 0.7030 | 0.3623 |
| DFPIR-ft-T3 | DIV2K | 10 | 31.348 | 0.8365 | 0.3014 |
| DFPIR-ft-T3 | LSDIR | 10 | 28.529 | 0.7935 | 0.2971 |
| DFPIR-ft-T1 | DIV2K | 14 | 21.582 | 0.6856 | 0.4134 |
| DFPIR-ft-T1 | LSDIR | 14 | 20.106 | 0.6383 | 0.4089 |

## DIV2K overlap-excluded (0801-0810 removed)

| Controller | Dataset | Scenarios | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| AutoIR | DIV2K | 24 | 24.963 | 0.7330 | 0.4059 |
| AutoIR | LSDIR | 24 | 22.956 | 0.6756 | 0.4009 |
| DFPIR-original | DIV2K | 24 | 24.014 | 0.7215 | 0.3802 |
| DFPIR-original | LSDIR | 24 | 22.029 | 0.6762 | 0.3752 |
| DFPIR-ft-last | DIV2K | 24 | 25.357 | 0.7374 | 0.3698 |
| DFPIR-ft-last | LSDIR | 24 | 23.363 | 0.6931 | 0.3646 |
| DFPIR-ft-best | DIV2K | 24 | 25.645 | 0.7490 | 0.3672 |
| DFPIR-ft-best | LSDIR | 24 | 23.615 | 0.7030 | 0.3623 |
| DFPIR-ft-T3 | DIV2K | 10 | 31.356 | 0.8377 | 0.3012 |
| DFPIR-ft-T3 | LSDIR | 10 | 28.529 | 0.7935 | 0.2971 |
| DFPIR-ft-T1 | DIV2K | 14 | 21.565 | 0.6856 | 0.4143 |
| DFPIR-ft-T1 | LSDIR | 14 | 20.106 | 0.6383 | 0.4089 |

## Paired vs AutoIR (full test)

- DFPIR-original vs AutoIR / DIV2K: ΔPSNR=-0.975 dB [-4.241, +2.368], wins 15/24, worst -22.286
- DFPIR-original vs AutoIR / LSDIR: ΔPSNR=-0.927 dB [-3.860, +1.996], wins 15/24, worst -21.971
- DFPIR-ft-last vs AutoIR / DIV2K: ΔPSNR=+0.399 dB [-0.521, +1.300], wins 16/24, worst -5.219
- DFPIR-ft-last vs AutoIR / LSDIR: ΔPSNR=+0.407 dB [-0.392, +1.196], wins 17/24, worst -4.818
- DFPIR-ft-best vs AutoIR / DIV2K: ΔPSNR=+0.683 dB [-0.037, +1.460], wins 17/24, worst -3.413
- DFPIR-ft-best vs AutoIR / LSDIR: ΔPSNR=+0.659 dB [+0.043, +1.352], wins 17/24, worst -1.993
- DFPIR-ft-T3 vs AutoIR / DIV2K: ΔPSNR=+1.665 dB [+0.888, +2.835], wins 10/10, worst +0.601
- DFPIR-ft-T3 vs AutoIR / LSDIR: ΔPSNR=+1.412 dB [+0.689, +2.550], wins 10/10, worst +0.397
- DFPIR-ft-T1 vs AutoIR / DIV2K: ΔPSNR=-0.019 dB [-0.849, +0.793], wins 7/14, worst -3.413
- DFPIR-ft-T1 vs AutoIR / LSDIR: ΔPSNR=+0.122 dB [-0.588, +0.838], wins 7/14, worst -1.993