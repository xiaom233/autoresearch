# blind_3009 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(1) → noise_speckle(4) → blur_glass(2) |
| 盲识别预测 | noise_gaussian_RGB(1) → blur_gaussian(1) → compression_jpeg_2000(2) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 1/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 21.7048 | — |
| Spec (从零训练) | PARAMS=predicted | 20.94 | -0.76 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 20.81 | -0.89 |

## 分析

- **Spec vs Ft**: Δ = +0.13 dB
- **最佳模型**: M_blind(GT) (21.7048 dB)
- **识别匹配度**: 1/3 函数匹配
- Spec ≈ Ft (+0.13 dB)，盲识别质量高时两种策略等价
