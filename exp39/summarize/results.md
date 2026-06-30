# exp39 实验结果总结

> 36 组完成 (9 退化 × 4 架构), 128×128, BATCH=32, Direct 训练
> 基线: Direct + Swin (454K)

---

## 一、SimpleGate 结果 (vs Swin)

| 退化 | Swin | SimpleGate | Δ | 判定 |
|------|:--:|:--:|:--:|:--:|
| C (contrast) | 24.80 | 24.52 | -0.28 | ≈0 |
| N (noise) | 28.37 | 28.36 | -0.01 | ≈0 |
| B (blur) | 25.41 | 25.47 | +0.06 | ≈0 |
| **L2 (contrast+noise)** | 25.73 | **26.51** | **+0.78** | ✅ |
| D3 (JPEG+blur) | 23.86 | 23.88 | +0.02 | ≈0 |
| G2 (stretch+noise) | 26.70 | 26.36 | -0.34 | ≈0 |
| BS (bright+noise) | 27.77 | 27.84 | +0.07 | ≈0 |
| S5 (motion×3) | 20.85 | 19.99 | -0.86 | ⚠️ |
| SF5 (gamma×3) | 22.85 | 22.71 | -0.14 | ≈0 |

**统计**: 1 胜 / 1 负 / 7 平, Best +0.78, Worst -0.86

**结论**: SimpleGate **可常开**。仅在三退化 (S5) 上轻微有害 (-0.86)，其余退化安全。

---

## 二、SCA 结果 (已废弃)

| 退化 | Δ vs Swin |
|------|:--:|
| C (contrast) | **+13.08** |
| G2 (stretch+noise) | **-7.04** ❌ |
| BS (bright+noise) | -4.09 ❌ |
| N (noise) | -3.55 ❌ |
| SF5 (gamma) | -3.29 ❌ |
| D3 (JPEG+blur) | -2.34 ❌ |
| B (blur) | -2.19 ❌ |
| S5 (motion) | -2.12 ❌ |
| L2 (contrast+noise) | -1.12 ⚠️ |

**结论**: 除纯 contrast 外全有害 (-1~-7 dB)。**废弃**。

---

## 三、关键发现

1. **SimpleGate 是唯一安全的零成本组件**: -32K 参数, L2 上 +0.78, 8/9 安全
2. **SCA 极度危险**: 纯 contrast +13 dB 但 8/9 退化有害, 风险远大于 ColorPre
3. **256 vs 128 不兼容**: L2: 128 比 256 低 2.46 dB, D3: 128 高 2.48 dB, 不能横向对比
4. **SimpleGate Epoch 2 崩溃**: D3 ckpt1=23.88→ckpt2=15.40, 建议更频繁 checkpoint

---

## 四、待续（结构组件 Layer 2）

Layer 2 实验 (SwiGLU/GDFN/FPro/FiLM-GCM, 需 EMBED_DIM 对齐) 尚未启动。
预计额外 45 组, 约 30 min 墙钟。
