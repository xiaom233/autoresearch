# DFPIR-from-scratch full test (best/last, DIV2K/LSDIR)

Scope: 24 scenarios; same T_s budget (~15.3k s/scenario), same predicted DC*, same 5 support, random init.
Test: DIV2K 100 + LSDIR 250; 17 deterministic scenarios hash-seeded LQ + 7 nondeterministic scenarios (`blind_0003/0005/0017/0018/0020/0022/0023`) on `E1E2_fixed_lq` fixed LQ. Best/last selected by support PSNR.

| Method | DIV2K PSNR | LSDIR PSNR | vs AutoIR DIV2K/LSDIR | vs DFPIR-ft-best | vs DFPIR-original |
|---|---:|---:|---:|---:|---:|
| DFPIR-scratch best | **23.833** | **21.811** | −1.135 / −1.145 (9/24, 8/24 wins) | −1.817 / −1.804 (5/24, 4/24) | −0.161 / −0.219 (8/24, 8/24) |
| DFPIR-scratch last | **23.638** | **21.667** | −1.330 / −1.289 (8/24, 7/24) | −2.012 / −1.949 (2/24, 1/24) | −0.357 / −0.363 (8/24, 8/24) |
| AutoIR | 24.968 | 22.956 | — | — | — |
| DFPIR-ft-best | 25.650 | 23.615 | +0.682 / +0.659 | — | — |
| DFPIR-ft-last | 25.366 | 23.363 | +0.398 / +0.407 | — | — |
| DFPIR-original (blinded zero-shot) | 23.995 | 22.030 | −0.974 / −0.926 | — | — |

Key result: at the same 3-round T_s budget, random-init DFPIR reaches ~DFPIR-original (blinded zero-shot) but remains below AutoIR (0.46M RestoreNet) and DFPIR-ft from blinded init. Pretrained initialization matters; the DFPIR same-budget weakness is not only a fine-tune budget/learning-rate issue.

Limitations: support-selected checkpoints; SSIM/LPIPS not computed for scratch; 7 nondeterministic scenarios use fixed LQ; not a full convergence claim.

Files: `scratch_full_summary.json`, `test_best_*`, `test_last_*`, `test_*fixed*`, `*/blind_*_summary.json`.
