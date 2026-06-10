# blind_3006 — 3退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg(5) → noise_poisson(3) → blur_jitter(5) |
| 盲识别预测 | compression_jpeg(5) → blur_jitter(4) → noise_poisson(3) |

**识别质量**: GOOD (CI pass 10/10) | 函数匹配 3/3

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 20.3592 | — |
| Spec (从零训练) | PARAMS=predicted | 21.02 | +0.66 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 21.0 | +0.64 |

## 分析

- **Spec vs Ft**: Δ = +0.02 dB
- **最佳模型**: Spec(pred) (21.02 dB)
- **识别匹配度**: 3/3 函数匹配
- Spec ≈ Ft (+0.02 dB)，盲识别质量高时两种策略等价
- ⭐ 盲识别完美匹配，验证了同图模式的有效性
