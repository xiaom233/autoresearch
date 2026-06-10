# blind_2009 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg(4) → blur_jitter(3) |
| 盲识别预测 | noise_impulse(1) → compression_jpeg(5) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 1/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 22.284 | — |
| Spec (从零训练) | PARAMS=predicted | 20.66 | -1.62 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 20.69 | -1.59 |

## 分析

- **Spec vs Ft**: Δ = -0.03 dB
- **最佳模型**: M_blind(GT) (22.284 dB)
- **识别匹配度**: 1/2 函数匹配
- Spec ≈ Ft (-0.03 dB)，盲识别质量高时两种策略等价
