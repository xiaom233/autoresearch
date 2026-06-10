# blind_2004 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | blur_zoom(2) → compression_jpeg(5) |
| 盲识别预测 | blur_gaussian(2) → compression_jpeg(4) |

**识别质量**: GOOD (CI pass 7/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 18.9913 | — |
| Spec (从零训练) | PARAMS=predicted | 18.54 | -0.45 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 18.34 | -0.65 |

## 分析

- **Spec vs Ft**: Δ = +0.20 dB
- **最佳模型**: M_blind(GT) (18.9913 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (+0.20 dB)，盲识别质量高时两种策略等价
