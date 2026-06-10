# blind_2003 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg(3) → noise_impulse(4) |
| 盲识别预测 | noise_gaussian_RGB(1) → noise_impulse(3) |

**识别质量**: GOOD (CI pass 9/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 26.5626 | — |
| Spec (从零训练) | PARAMS=predicted | 26.76 | +0.20 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 26.72 | +0.16 |

## 分析

- **Spec vs Ft**: Δ = +0.04 dB
- **最佳模型**: Spec(pred) (26.76 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (+0.04 dB)，盲识别质量高时两种策略等价
