# E1 one-shot / no-reflection comparison

| Controller | DIV2K PSNR | DIV2K SSIM | DIV2K LPIPS | LSDIR PSNR | LSDIR SSIM | LSDIR LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| rule-oneshot | 22.206 | — | — | — | — | — |
| rule-full | 24.632 | 0.7213 | 0.4155 | 22.432 | 0.6588 | 0.4139 |
| AutoIR-oneshot | 23.989 | — | — | — | — | — |
| AutoIR-full | 25.268 | 0.7391 | 0.3968 | 23.197 | 0.6837 | 0.3920 |

Paired scenario-level comparisons (full test):

- rule-full - rule-oneshot DIV2K: +2.426 [+1.347,+3.666], wins 18/23, worst -0.572, best +9.854
- AutoIR-full - AutoIR-oneshot DIV2K: +1.279 [+0.391,+2.680], wins 15/23, worst -0.106, best +14.385
- AutoIR-full - rule-oneshot DIV2K: +3.062 [+0.301,+6.108], wins 17/23, worst -14.019, best +25.889
- AutoIR-oneshot - rule-oneshot DIV2K: +1.783 [-1.159,+4.919], wins 14/23, worst -14.014, best +23.790
- AutoIR-full - rule-full DIV2K: +0.636 [-2.441,+3.488], wins 15/23, worst -22.992, best +20.527
- AutoIR-full - rule-full LSDIR: +0.766 [-1.935,+3.451], wins 15/23, worst -18.981, best +21.208
