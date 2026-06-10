# blind_3005 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | noise_speckle(4) → blur_gaussian(2) → compression_jpeg_2000(4) |
| 盲识别预测 | blur_gaussian(5) → compression_jpeg_2000(1) → saturate_weaken_HSV(1) |

**识别质量**: GOOD (CI pass 7/10) | 函数匹配 2/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 18.5844 | — |
| Spec (从零训练) | PARAMS=predicted | 16.92 | -1.66 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 17.28 | -1.30 |

## 分析

- **Spec vs Ft**: Δ = -0.36 dB
- **最佳模型**: M_blind(GT) (18.5844 dB)
- **识别匹配度**: 2/3 函数匹配
- ✅ Ft 优于 Spec，盲预训练特征对此退化有迁移价值
