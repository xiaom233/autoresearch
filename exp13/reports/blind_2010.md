# blind_2010 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_spatially_correlated(2) → compression_jpeg_2000(3) |
| 盲识别预测 | blur_lens(3) → compression_jpeg(1) |

**识别质量**: NEEDS_WORK (CI pass 5/10) | 函数匹配 0/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 23.7559 | — |
| Spec (从零训练) | PARAMS=predicted | 23.14 | -0.62 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 23.18 | -0.58 |

## 分析

- **Spec vs Ft**: Δ = -0.04 dB
- **最佳模型**: M_blind(GT) (23.7559 dB)
- **识别匹配度**: 0/2 函数匹配
- Spec ≈ Ft (-0.04 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=5/10)，影响了下游训练效果
