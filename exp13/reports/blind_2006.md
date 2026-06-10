# blind_2006 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_gaussian_RGB(1) → compression_jpeg(2) |
| 盲识别预测 | noise_gaussian_RGB(2) → compression_jpeg(3) |

**识别质量**: GOOD (CI pass 7/10) | 函数匹配 2/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 27.9514 | — |
| Spec (从零训练) | PARAMS=predicted | 28.33 | +0.38 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 28.4 | +0.45 |

## 分析

- **Spec vs Ft**: Δ = -0.07 dB
- **最佳模型**: Ft(pred) (28.4 dB)
- **识别匹配度**: 2/2 函数匹配
- Spec ≈ Ft (-0.07 dB)，盲识别质量高时两种策略等价
