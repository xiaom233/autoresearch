# blind_2007 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg(4) → noise_gaussian_YCrCb(5) |
| 盲识别预测 | oversharpen(3) → noise_impulse(5) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 0/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 23.645 | — |
| Spec (从零训练) | PARAMS=predicted | 19.3 | -4.34 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 19.25 | -4.39 |

## 分析

- **Spec vs Ft**: Δ = +0.05 dB
- **最佳模型**: M_blind(GT) (23.645 dB)
- **识别匹配度**: 0/2 函数匹配
- Spec ≈ Ft (+0.05 dB)，盲识别质量高时两种策略等价
