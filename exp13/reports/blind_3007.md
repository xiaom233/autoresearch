# blind_3007 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_spatially_correlated(5) → compression_jpeg_2000(2) → blur_glass(4) |
| 盲识别预测 | noise_spatially_correlated(1) → compression_jpeg_2000(2) → blur_lens(2) |

**识别质量**: GOOD (CI pass 7/10) | 函数匹配 2/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 20.438 | — |
| Spec (从零训练) | PARAMS=predicted | 19.61 | -0.83 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 19.6 | -0.84 |

## 分析

- **Spec vs Ft**: Δ = +0.01 dB
- **最佳模型**: M_blind(GT) (20.438 dB)
- **识别匹配度**: 2/3 函数匹配
- Spec ≈ Ft (+0.01 dB)，盲识别质量高时两种策略等价
