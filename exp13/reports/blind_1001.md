# blind_1001 — 1退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | blur_gaussian(5) |
| 盲识别预测 | blur_gaussian(5) |

**识别质量**: GOOD (CI pass 10/10) | 函数匹配 1/1

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 20.9222 | — |
| Spec (从零训练) | PARAMS=predicted | 21.85 | +0.93 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 20.27 | -0.65 |

## 分析

- **Spec vs Ft**: Δ = +1.58 dB
- **最佳模型**: Spec(pred) (21.85 dB)
- **识别匹配度**: 1/1 函数匹配
- ✅ Spec 优于 Ft，说明从零训练对此退化更有效
- ⭐ 盲识别完美匹配，验证了同图模式的有效性
