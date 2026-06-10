# blind_2002 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(5) → blur_glass(3) |
| 盲识别预测 | blur_gaussian(3) → compression_jpeg_2000(2) |

**识别质量**: GOOD (CI pass 9/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 20.3415 | — |
| Spec (从零训练) | PARAMS=predicted | 19.42 | -0.92 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 19.46 | -0.88 |

## 分析

- **Spec vs Ft**: Δ = -0.04 dB
- **最佳模型**: M_blind(GT) (20.3415 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (-0.04 dB)，盲识别质量高时两种策略等价
