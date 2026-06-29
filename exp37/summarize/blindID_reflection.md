# exp37 盲识别与反思报告

## R0 盲识别

### 配置

- Agent: 8 并行子 Agent，每个处理 3 个挑战
- 工具: run_full_analysis.py + test_candidate.py + peeled_noise_check.py
- 协议: SKILL.md v12 逐层剥离 + 信号驱动探索
- 验证: 同图模式 PSNR / verify_signals.py

### 逐挑战 R0 预测 vs GT 对比

| Challenge | Verdict | R0 预测 | GT | 函数匹配 | 步数 |
|-----------|---------|---------|----|:--:|:--:|
| blind_0001 | UNCERTAIN | noise_poisson:4 → quantization_hist:2 | noise_poisson:5 | 1/1 | 1 |
| blind_0002 | UNCERTAIN | noise_gaussian_YCrCb:2 | noise_gaussian_YCrCb:2 | 1/1 | 1 |
| blind_0003 | LIKELY | blur_gaussian:2 | blur_glass:2 | 0/1 | 1 |
| blind_0004 | UNCERTAIN | blur_gaussian:1 | blur_motion:1 | 0/1 | 1 |
| blind_0005 | POOR | brightness_brighten_gamma_RGB:3 → saturate_strengthen_HSV:4 | noise_impulse:5 | 0/1 | 1 |
| blind_0006 | LIKELY | compression_jpeg:1 | compression_jpeg:1 | 1/1 | 1 |
| blind_0007 | UNCERTAIN | noise_speckle:1 | noise_speckle:1 | 1/1 | 1 |
| blind_0008 | GOOD | saturate_weaken_HSV:4 | saturate_weaken_HSV:4 | 1/1 | 1 |
| blind_0009 | UNCERTAIN | noise_gaussian_YCrCb:5 → brightness_darken_shfit_HSV:3 → saturate_strengthen_HSV:5 | noise_poisson:2 → saturate_strengthen_YCrCb:3 | 0/2 | 2 |
| blind_0010 | GOOD | blur_gaussian:1 → brightness_darken_gamma_HSV:5 | blur_gaussian:1 → brightness_darken_gamma_HSV:5 | 2/2 | 2 |
| blind_0011 | LIKELY | blur_gaussian:4 | noise_poisson:3 → blur_gaussian:4 | 1/2 | 2 |
| blind_0012 | UNCERTAIN | blur_motion:5 → compression_jpeg:3 → noise_gaussian_RGB:2 | blur_motion:5 → compression_jpeg:3 | 2/2 | 2 |
| blind_0013 | UNCERTAIN | noise_gaussian_RGB:3 → compression_jpeg:2 | noise_speckle:3 → compression_jpeg:2 | 1/2 | 2 |
| blind_0014 | UNCERTAIN | brightness_brighten_shfit_HSV:3 → blur_glass:1 → compression_jpeg:3 | brightness_brighten_shfit_HSV:3 → compression_jpeg:2 | 2/2 | 2 |
| blind_0015 | UNCERTAIN | blur_motion:3 → brightness_darken_gamma_HSV:3 | blur_motion:1 → brightness_darken_shfit_RGB:3 | 1/2 | 2 |
| blind_0016 | LIKELY | blur_motion:4 → compression_jpeg:2 | blur_motion:3 → compression_jpeg:2 | 2/2 | 2 |
| blind_0017 | UNCERTAIN | blur_gaussian:4 → compression_jpeg:1 | noise_impulse:5 → blur_gaussian:3 → compression_jpeg:4 | 2/3 | 3 |
| blind_0018 | POOR | contrast_strengthen_stretch:4 → compression_jpeg:4 | noise_impulse:1 → contrast_strengthen_stretch:4 → compression_jpeg:5 | 2/3 | 3 |
| blind_0019 | UNCERTAIN | compression_jpeg:4 → blur_gaussian:4 → saturate_weaken_HSV:5 | blur_lens:4 → compression_jpeg:4 → saturate_weaken_HSV:5 | 2/3 | 3 |
| blind_0020 | UNCERTAIN | blur_gaussian:4 → contrast_weaken_stretch:4 | noise_poisson:3 → blur_glass:4 → contrast_weaken_stretch:4 | 1/3 | 3 |
| blind_0021 | POOR | compression_jpeg:4 → blur_motion:5 → brightness_brighten_gamma_HSV:3 | blur_gaussian:3 → brightness_brighten_shfit_RGB:1 → compression_jpeg:4 | 1/3 | 3 |
| blind_0022 | UNCERTAIN | compression_jpeg:1 → blur_gaussian:3 | noise_impulse:2 → blur_glass:3 → compression_jpeg:1 | 1/3 | 3 |
| blind_0023 | UNCERTAIN | compression_jpeg:5 → blur_glass:5 → contrast_weaken_scale:1 | noise_gaussian_YCrCb:4 → blur_glass:5 → compression_jpeg:4 | 2/3 | 3 |
| blind_0024 | UNCERTAIN | brightness_darken_gamma_HSV:2 → saturate_strengthen_HSV:1 | noise_gaussian_RGB:4 → blur_gaussian:3 → saturate_strengthen_HSV:2 | 1/3 | 3 |

**函数匹配率**: 28/48 = 58.3%

### 按退化步数统计

| 步数 | 挑战数 | 函数完全匹配 | 匹配率 |
|:--:|:--:|:--:|:--:|
| 1 | 8 | 4 | 50% |
| 2 | 8 | 2 | 25% |
| 3 | 8 | 0 | 0% |

### Verdict 分布

- GOOD: 2
- LIKELY: 4
- POOR: 3
- UNCERTAIN: 15

## R1 反思

### 触发条件

- 新协议: 仅 GOOD/LIKELY + GT PSNR >= 35 跳过
- 其余 23/24 挑战全部触发反思
- 反思方式: Skill 子 Agent，信号驱动 + verify_signals.py

### 逐挑战 R0→R1 修正对比

| Challenge | R0 PSNR | R1 PSNR | Δ | R0 预测 | R1 预测 | 效果 |
|-----------|:--:|:--:|:--:|------|------|:--:|
| blind_0001 | 17.72 | 27.31 | +9.59 | noise_poisson:4→quantization_hist:2 | noise_poisson:4 | ✅ 有效 |
| blind_0005 | 14.04 | 13.87 | -0.17 | brightness_brighten_gamma_RGB:3→saturate_strengthen_HSV:4 | contrast_strengthen_scale:2→brightness_brighten_shfit_RGB:1→saturate_strengthen_HSV:4 | ➖ 持平 |
| blind_0009 | 6.91 | 11.55 | +4.64 | noise_gaussian_YCrCb:5→brightness_darken_shfit_HSV:3→saturate_strengthen_HSV:5 | noise_gaussian_YCrCb:5→brightness_darken_gamma_HSV:4 | ✅ 有效 |
| blind_0012 | 20.51 | 20.70 | +0.19 | blur_motion:5→compression_jpeg:3→noise_gaussian_RGB:2 | blur_motion:5→compression_jpeg:3→noise_gaussian_RGB:1 | ➖ 持平 |
| blind_0014 | 21.03 | 21.53 | +0.50 | brightness_brighten_shfit_HSV:3→blur_glass:1→compression_jpeg:3 | brightness_brighten_shfit_HSV:3→blur_gaussian:1→compression_jpeg:2 | ➖ 持平 |
| blind_0018 | 16.04 | 16.77 | +0.73 | contrast_strengthen_stretch:4→compression_jpeg:4 | contrast_strengthen_stretch:5→noise_gaussian_YCrCb:2→compression_jpeg:5 | ➖ 持平 |
| blind_0019 | 14.93 | 14.62 | -0.31 | compression_jpeg:4→blur_gaussian:4→saturate_weaken_HSV:5 | compression_jpeg:4→blur_lens:4→saturate_weaken_HSV:5 | ➖ 持平 |
| blind_0021 | 14.52 | 17.55 | +3.03 | compression_jpeg:4→blur_motion:5→brightness_brighten_gamma_HSV:3 | compression_jpeg:4→blur_lens:4→brightness_brighten_shfit_HSV:1 | ✅ 有效 |

### 反思成功率

- R1 挑战数: 8
- 有效 (>+0.5dB): 4
- 有害 (<-0.5dB): 0
- 持平: 4
- 平均 Δ: +2.27 dB
