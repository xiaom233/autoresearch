# blind_2001 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | blur_glass(1) → noise_impulse(5) |
| 盲识别预测 | noise_impulse(3) → noise_gaussian_RGB(2) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 24.9167 | — |
| Spec (从零训练) | PARAMS=predicted | 24.14 | -0.78 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 24.12 | -0.80 |

## 分析

- **Spec vs Ft**: Δ = +0.02 dB
- **最佳模型**: M_blind(GT) (24.9167 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (+0.02 dB)，盲识别质量高时两种策略等价
