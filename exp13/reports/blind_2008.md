# blind_2008 — 2退化

## 退化管线

| 来源 | Pipeline |
|------|----------|
| GT (ground truth) | compression_jpeg_2000(1) → blur_jitter(4) |
| 盲识别预测 | noise_gaussian_RGB(5) → compression_jpeg(5) |

**识别质量**: NEEDS_WORK (CI pass 6/10) | 函数匹配 0/2

## 训练结果 (EPOCH=2, VAL=GT)

| 模型 | 训练参数 | PSNR | Δ vs M_blind |
|------|---------|:--:|:--:|
| M_blind (盲基线) | 随机退化 | 21.8457 | — |
| Spec (从零训练) | PARAMS=predicted | 19.24 | -2.61 |
| Ft (微调) | PARAMS=predicted + LOAD_CKPT | 19.24 | -2.61 |

## 分析

- **Spec vs Ft**: Δ = +0.00 dB
- **最佳模型**: M_blind(GT) (21.8457 dB)
- **识别匹配度**: 0/2 函数匹配
- Spec ≈ Ft (+0.00 dB)，盲识别质量高时两种策略等价
- ⚠️ 盲识别需要改进 (CI=6/10)，影响了下游训练效果
