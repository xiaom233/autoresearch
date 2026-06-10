# blind_3010 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(3) → blur_motion(4) → noise_spatially_correlated(4) |
| 盲识别预测 | noise_impulse(2) → blur_motion(1) → compression_jpeg(2) |

**识别质量**: NEEDS_WORK (CI pass 5/10) | 函数匹配 1/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 18.4422 | — |
| Spec (从零训练) | PARAMS=predicted | 18.63 | +0.19 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 18.76 | +0.32 |

## 分析

- **Spec vs Ft**: Δ = -0.13 dB
- **最佳模型**: Ft(pred) (18.76 dB)
- **识别匹配度**: 1/3 函数匹配
- Spec ≈ Ft (-0.13 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=5/10)，影响了下游训练效果
