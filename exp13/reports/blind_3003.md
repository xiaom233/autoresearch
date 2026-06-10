# blind_3003 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_impulse(2) → compression_jpeg(1) → blur_glass(1) |
| 盲识别预测 | blur_lens(4) → noise_gaussian_RGB(1) → compression_jpeg(2) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 1/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 24.8281 | — |
| Spec (从零训练) | PARAMS=predicted | 23.38 | -1.45 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 15.7 | -9.13 |

## 分析

- **Spec vs Ft**: Δ = +7.68 dB
- **最佳模型**: M_blind(GT) (24.8281 dB)
- **识别匹配度**: 1/3 函数匹配
- ✅ Spec 优于 Ft，说明从零训练对此退化更有效
