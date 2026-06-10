# blind_2005 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | blur_gaussian(4) → noise_spatially_correlated(4) |
| 盲识别预测 | blur_gaussian(2) → noise_impulse(1) |

**识别质量**: NEEDS_WORK (CI pass 6/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 20.6833 | — |
| Spec (从零训练) | PARAMS=predicted | 19.33 | -1.35 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 19.49 | -1.19 |

## 分析

- **Spec vs Ft**: Δ = -0.16 dB
- **最佳模型**: M_blind(GT) (20.6833 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (-0.16 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=6/10)，影响了下游训练效果
