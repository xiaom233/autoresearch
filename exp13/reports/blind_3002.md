# blind_3002 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(1) → blur_jitter(4) → noise_impulse(1) |
| 盲识别预测 | oversharpen(2) → blur_gaussian(1) → noise_impulse(1) |

**识别质量**: NEEDS_WORK (CI pass 4/10) | 函数匹配 1/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 21.7789 | — |
| Spec (从零训练) | PARAMS=predicted | 17.73 | -4.05 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 17.85 | -3.93 |

## 分析

- **Spec vs Ft**: Δ = -0.12 dB
- **最佳模型**: M_blind(GT) (21.7789 dB)
- **识别匹配度**: 1/3 函数匹配
- Spec ≈ Ft (-0.12 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=4/10)，影响了下游训练效果
