# blind_1002 — 1退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(2) |
| 盲识别预测 | compression_jpeg_2000(2) |

**识别质量**: GOOD (CI pass 10/10) | 函数匹配 1/1

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 27.5243 | — |
| Spec (从零训练) | PARAMS=predicted | 28.23 | +0.71 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 28.0 | +0.48 |

## 分析

- **Spec vs Ft**: Δ = +0.23 dB
- **最佳模型**: Spec(pred) (28.23 dB)
- **识别匹配度**: 1/1 函数匹配
- Spec ≈ Ft (+0.23 dB)，盲识别质量高时两种策略等价
- ⭐ 盲识别完美匹配，验证了同图模式的有效性
