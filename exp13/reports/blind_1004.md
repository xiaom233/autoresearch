# blind_1004 — 1退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | blur_jitter(5) |
| 盲识别预测 | noise_gaussian_RGB(3) |

**识别质量**: GOOD (CI pass 8/10) | 函数匹配 0/1

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 21.2329 | — |
| Spec (从零训练) | PARAMS=predicted | 18.82 | -2.41 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 18.78 | -2.45 |

## 分析

- **Spec vs Ft**: Δ = +0.04 dB
- **最佳模型**: M_blind(GT) (21.2329 dB)
- **识别匹配度**: 0/1 函数匹配
- Spec ≈ Ft (+0.04 dB)，盲识别质量高时两种策略等价
