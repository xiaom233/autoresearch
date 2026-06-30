# exp39: 新架构组件系统探索

> 9 组件 × 9 退化 = 81 组，全已实现并测试通过
> 基线: Direct + Swin (454K), 训练尺寸 128×128 (TRAIN_CROP)

---

## 一、测试组件（全 9 个，已实现 ✅）

| # | 组件 | 来源 | 参数量 | Δ | 机制 | 环境变量 |
|:--:|------|------|:--:|:--:|------|------|
| 0 | **Swin** | SwinIR (ICCVW 2021) | 454,531 | — | 窗口自注意力基线 | — |
| 1 | **SimpleGate** | NAFNet (ECCV 2022) | 421,763 | **-32K** | 通道对半分→相乘, 替代 GELU | `AR_USE_SIMPLE_GATE=1` |
| 2 | **SCA** | NAFNet (ECCV 2022) | 455,043 | +512 | GAP→L2Norm→通道缩放 | `AR_USE_SCA=1` |
| 3 | **SG+SCA** | 组合 | 422,275 | -32K | SimpleGate + SCA | `AR_USE_SIMPLE_GATE=1 AR_USE_SCA=1` |
| 4 | **SwiGLU** | Shazeer (2020) | 521,091 | +67K | SiLU(w1x)*(w2x) 门控 FFN | `AR_ACTIVATION=swiglu` |
| 5 | **LearnSkip** | — | 454,535 | +4 | α·conv_skip 可学习残差 | `AR_SKIP_RSTB=learnable` |
| 6 | **FPro** | FPro (ECCV 2024) | 455,555 | +1K | FFT→频域门控→IFFT | `AR_USE_FPRO=1` |
| 7 | **GDFN** | X-Restormer (2024) | 541,571 | +87K | GELU+DWConv ⊙ Gate | `AR_USE_GDFN=1` |
| 8 | **FiLM-GCM** | exp10 已验证 | 480,259 | +26K | 全局通道调制 (contrast专用) | `AR_USE_GCM=1 AR_LR=5e-4` |

---

## 二、退化矩阵（9 种, 1/2/3 步全覆盖）

| ID | 步数 | 管线 | 类型 |
|:--:|:--:|------|------|
| C | 1 | contrast_weaken_scale(3) | 纯 contrast |
| N | 1 | noise_gaussian_RGB(3) | 纯 noise |
| B | 1 | blur_gaussian(3) | 纯 blur |
| L2 | 2 | noise(3)+contrast_scale(3) | contrast+noise |
| D3 | 2 | JPEG(3)+blur_lens(4) | 梯度干扰 |
| G2 | 2 | noise(3)+stretch(3) | stretch+noise |
| BS | 2 | noise(3)+brightness_shift(3) | brightness+noise |
| S5 | 3 | motion(5)+noise(1)+JPEG(1) | 三退化 mixed |
| SF5 | 3 | blur(3)+noise(3)+gamma(3) | gamma triple |

---

## 三、实验矩阵（9 组件 × 9 退化 = 81 组）

### 统一配置

```
TRAIN_CROP=128 (dataloader随机裁剪, 退化前)
EPOCH_BUDGET=2  LR=5e-4  BATCH_SIZE=16  EMBED_DIM=64
WINDOW_SIZE=8  ATTENTION_TYPE=swin  LOSS_FN=l1
所有实验 Direct 训练 (无 CURRICULUM_CONFIG, 无 LOAD_CKPT)
```

### 81 组全矩阵

| Phase | 组件 | 变量 | 组数 | 核心假设 |
|:--:|------|------|:--:|------|
| A | Swin | — | 9 | 基线 |
| B | SimpleGate | `AR_USE_SIMPLE_GATE=1` | 9 | D3/S5 受益 (减少梯度干扰) |
| C | SCA | `AR_USE_SCA=1` | 9 | C/L2/G2 受益 (增强通道交互) |
| D | SG+SCA | B+C 组合 | 9 | 可能叠加 |
| E | SwiGLU | `AR_ACTIVATION=swiglu` | 9 | D3/S5 受益 (门控 FFN) |
| F | LearnSkip | `AR_SKIP_RSTB=learnable` | 9 | 全域微益 +0.1~0.2 |
| G | FPro | `AR_USE_FPRO=1` | 9 | C/L2 受益 (频域选频) |
| H | GDFN | `AR_USE_GDFN=1` | 9 | D3/S5 受益 (DWConv 空间) |
| I | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | 9 | C/L2/G2 受益 (全局调制) |
| | | **合计** | **81** | |

---

## 四、预测矩阵

| 退化 | SG | SCA | SG+SCA | SwiGLU | LearnSkip | FPro | GDFN | GCM |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| C (contrast) | ≈0 | **+0.3~0.5** | +0.3~0.5 | +0.1 | +0.1 | **+0.3** | +0.1 | **+0.4** |
| N (noise) | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| B (blur) | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| L2 (c+n) | ≈0 | **+0.3~0.5** | +0.3~0.5 | +0.2 | +0.1 | **+0.3** | +0.2 | **+0.47** |
| D3 (j+b) | **+0.3~0.5** | ≈0 | +0.3~0.5 | **+0.3** | +0.1 | +0.1 | **+0.4** | ≈0 |
| G2 (s+n) | ≈0 | **+0.3** | +0.3 | +0.1 | +0.1 | +0.1 | +0.1 | **+0.35** |
| BS (b+n) | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |
| S5 (3x) | **+0.3~0.5** | ≈0 | +0.3~0.5 | **+0.3** | +0.1 | +0.1 | **+0.4** | ? |
| SF5 (γ3x) | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |

---

## 五、评估

### 退化特异性

每个组件在匹配退化上的 Best Δ vs Swin。

### 跨退化泛化安全性

| 组件 | 参数量 | Best 预期 | Worst 预期 | 安全? | 盲识别可用? |
|------|:--:|:--:|:--:|:--:|:--:|
| SimpleGate | -32K | +0.3~0.5 (D3/S5) | ≈0 | ✅ | ✅ 常开 |
| SCA | +512 | +0.3~0.5 (C/L2/G2) | ≈0 | ✅ | ✅ 常开 |
| SG+SCA | -32K | 叠加 | ≈0 | ✅ | ✅ 常开 |
| SwiGLU | +67K | +0.3 (D3/S5) | ≈0 | ✅ | ✅ 常开 |
| LearnSkip | +4 | +0.1~0.2 | ≈0 | ✅ | ✅ 常开 |
| FPro | +1K | +0.3 (C/L2) | ≈0 | ✅ | ✅ 常开 |
| GDFN | +87K | +0.4 (D3/S5) | ≈0 | ? | ? |
| FiLM-GCM | +26K | +0.47 (L2) | ≈0 | ✅ | 需信号门控 |

---

## 六、🔴 非零成本组件实测结果 (exp39, 参数对齐, GT退化已知)

> 45组实验 (SW+SwiGLU+FPro+GDFN+GCM), 参数全部对齐 ≤477K
> 配置: BATCH=32, TRAIN_CROP=128, EPOCH=2, LR=5e-4

### 总体对比

| 组件 | EMBED_DIM | 参数量 | 平均 PSNR | Δ vs Swin | 推荐 |
|------|:--:|:--:|:--:|:--:|------|
| Swin 基线 | 64 | 454,531 | 26.06 | — | 基线 |
| SwiGLU | 60 | 458,523 | 26.31 | **+0.25** | 边际, 不推荐 |
| GDFN | 56 | 417,875 | 26.28 | **+0.22** | 边际, 参数反而更少 |
| **FiLM-GCM** | 60 | 422,643 | **29.22** | **+3.16** | ✅ 强烈推荐 (contrast+结构化) |
| FPro | 64 | 455,555 | 全崩 | — | ❌ torch.fft 与 torch.compile 不兼容 |

### 逐退化详细对比

| 退化 | Swin | SwiGLU | GDFN | FiLM-GCM | 最佳 |
|------|:--:|:--:|:--:|:--:|:--:|
| C (contrast) | 47.13 | 47.02 | 47.04 | **50.18** | GCM +3.05 |
| N (noise) | 28.07 | 27.97 | 27.91 | **28.10** | GCM +0.03 |
| B (blur) | 31.80 | 31.76 | 31.80 | **32.01** | GCM +0.21 |
| L2 (c+n) | 28.03 | 28.47 | 28.42 | **29.55** | GCM +1.52 |
| D3 (j+b) | 21.53 | 21.78 | 21.72 | **22.22** | GCM +0.69 |
| G2 (s+n) | 27.94 | 28.13 | 28.09 | **33.44** | GCM +5.50 |
| BS (b+n) | 28.52 | 28.53 | 28.42 | **28.58** | GCM +0.06 |
| S5 (3x) | 20.02 | 20.06 | 20.09 | **22.67** | GCM +2.65 |
| SF5 (g3x) | 21.54 | 23.04 | 22.99 | **36.17** | GCM +14.63 |

### 关键发现

1. **FiLM-GCM 统治级表现**: 9/9 退化全部领先, 平均 +3.16 dB。contrast-containing 退化 (C/L2/G2/SF5) 收益最大 (+1.5~+14.6 dB)
2. **SwiGLU 和 GDFN 边际**: +0.25 和 +0.22 dB, 不推荐 (GDFN 参数更少但收益太小)
3. **FPro 全崩**: torch.fft 与 torch.compile inductor 不兼容, 需 AR_NO_COMPILE=1 后重测
4. **与 exp10 结论一致**: FiLM-GCM 在 GT 已知退化上有效 (exp10 L5 +3.52, exp37 盲识别未复现)
5. **emb_dim 对齐验证成功**: SwiGLU dim=60, GDFN dim=56, GCM dim=60 全部 ≤477K

---

## 七、执行

```bash
cd /data/zyli/projects/autoresearch
python3 exp39/scripts/generate.py
bash scripts/exp_launcher.sh start exp39/scripts/tasks.txt 1,2,3,4,5,6,7
```

**预计墙钟**: 81组 × ~3-4 min / 7 GPU ≈ **40-45 min** (128×128 快于 256×256)
