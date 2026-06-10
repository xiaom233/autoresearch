# blind_3001 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(2) → noise_gaussian_YCrCb(3) → blur_jitter(3) |
| 盲识别预测 | oversharpen(2) → noise_impulse(4) → compression_jpeg(2) |

**识别质量**: NEEDS_WORK (CI pass 4/10) | 函数匹配 0/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 21.493 | — |
| Spec (从零训练) | PARAMS=predicted | 19.97 | -1.52 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 20.1 | -1.39 |

## 分析

- **Spec vs Ft**: Δ = -0.13 dB
- **最佳模型**: M_blind(GT) (21.493 dB)
- **识别匹配度**: 0/3 函数匹配
- Spec ≈ Ft (-0.13 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=4/10)，影响了下游训练效果
