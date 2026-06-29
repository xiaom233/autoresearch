# exp39: 新架构组件系统探索

> 两层设计：零成本组件 (5 种, 全覆盖) + 结构组件 (4 种, 重点退化)
> 基线: Direct + Swin (454K)
> 总计: **75 组**

---

## 一、组件清单

### Layer 1: 零/负参数组件 (全覆盖 9 退化)

| # | 组件 | 来源 | Δ 参数 | 机制 |
|:--:|------|------|:--:|------|
| 1 | **SimpleGate** | NAFNet (ECCV 2022) | **-32K** | 通道对半分→相乘, 替代 GELU |
| 2 | **SCA** | NAFNet (ECCV 2022) | **+512** | GAP→L2Norm→通道缩放 |
| 3 | **SG+SCA** | 组合 | **-32K** | SimpleGate + SCA |
| 4 | **SwiGLU** | Shazeer (2020) | +98K | 双线性门控 SiLU(w1x)*(w2x) |
| 5 | **LearnableSkip** | — | +4 | α·conv_skip 可学习残差权重 |

### Layer 2: 结构组件 (5 关键退化重点验证)

| # | 组件 | 来源 | Δ 参数 | 机制 |
|:--:|------|------|:--:|------|
| 6 | **GDFN** | X-Restormer (2024) | +50K? | 门控 + 3×3 DWConv FFN |
| 7 | **FPro-lite** | FPro (ECCV 2024) | +5K | FFT→高低频分解→门控调制 |
| 8 | **DFPIR-Perturb** | DFPIR (CVPR 2025) | +10K | 通道shuffle + 注意力掩码扰动 |
| 9 | **Swin-GCM** | exp10 已验证 | +26K | FiLM-GCM (contrast+结构化 +3.52) |

---

## 二、退化矩阵

### 全覆盖 (9 退化 — Layer 1 使用)

| ID | 步数 | 管线 |
|:--:|:--:|------|
| C | 1 | contrast_weaken_scale(3) |
| N | 1 | noise_gaussian_RGB(3) |
| B | 1 | blur_gaussian(3) |
| L2 | 2 | noise(3)+contrast_scale(3) |
| D3 | 2 | JPEG(3)+blur_lens(4) |
| G2 | 2 | noise(3)+stretch(3) |
| BS | 2 | noise(3)+brightness_shift(3) |
| S5 | 3 | motion(5)+noise(1)+JPEG(1) |
| SF5 | 3 | blur(3)+noise(3)+gamma(3) |

### 重点退化 (5 — Layer 2 使用)

C / L2 / D3 / G2 / S5 — 覆盖 contrast、梯度干扰、stretch、三退化

---

## 三、实验矩阵

### 统一配置

```
EPOCH_BUDGET=2  LR=5e-4  BATCH_SIZE=16  EMBED_DIM=64
WINDOW_SIZE=8  ATTENTION_TYPE=swin  LOSS_FN=l1
所有实验 Direct 训练
```

### Phase A: Swin 基线 (9 组)

| ID | 退化 |
|------|:--:|
| SW_C ~ SW_SF5 | 全部 9 退化 |

`AR_PARAMS_PATH=exp39/degradation/{ID}.json AR_VAL_PARAMS_PATH=exp39/degradation/{ID}.json`

### Phase B: SimpleGate (9 组)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| SG_C~B | 单步 3 | ≈0 |
| SG_L2 | L2 | ≈0 |
| SG_D3 | D3 | **+0.3~0.5** (梯度干扰) |
| SG_G2/BS | G2/BS | ≈0 |
| SG_S5 | S5 | **+0.3~0.5** (三退化) |
| SG_SF5 | SF5 | ≈0 |

`AR_USE_SIMPLE_GATE=1`

### Phase C: SCA (9 组)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| SCA_C | C | **+0.3~0.5** (contrast) |
| SCA_N/B | N/B | ≈0 |
| SCA_L2 | L2 | **+0.3~0.5** |
| SCA_D3 | D3 | ≈0 |
| SCA_G2 | G2 | **+0.2~0.3** (stretch) |
| SCA_BS | BS | ≈0 |
| SCA_S5/SF5 | S5/SF5 | ≈0 |

`AR_USE_SCA=1`

### Phase D: SG+SCA (9 组)

`AR_USE_SIMPLE_GATE=1 AR_USE_SCA=1`

### Phase E: SwiGLU (9 组)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| GLU_C | C | +0.1~0.2 |
| GLU_N/B | N/B | ≈0 |
| GLU_L2 | L2 | +0.1~0.3 |
| GLU_D3 | D3 | **+0.2~0.4** |
| GLU_G2/BS | G2/BS | ≈0 |
| GLU_S5 | S5 | **+0.2~0.4** |
| GLU_SF5 | SF5 | ≈0 |

`AR_ACTIVATION=swiglu`

### Phase F: LearnableSkip (9 组)

`AR_SKIP_RSTB=learnable`

### Phase G: GDFN (5 组 — 重点退化)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| GDFN_C | C | +0.1~0.2 |
| GDFN_L2 | L2 | +0.2~0.3 |
| GDFN_D3 | D3 | **+0.3~0.5** (DWConv 帮助空间) |
| GDFN_G2 | G2 | +0.1~0.2 |
| GDFN_S5 | S5 | **+0.3~0.5** |

`AR_USE_GDFN=1` (需实现)

### Phase H: FPro-lite (5 组 — 重点退化)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| FPRO_C | C | **+0.3~0.5** (频域选频) |
| FPRO_L2 | L2 | **+0.3~0.5** |
| FPRO_D3 | D3 | +0.1~0.3 |
| FPRO_G2 | G2 | +0.1~0.2 |
| FPRO_S5 | S5 | +0.1~0.3 |

`AR_USE_FPRO=1` (需实现)

### Phase I: FiLM-GCM 复现 (5 组 — 重点退化)

| ID | 退化 | 预期 |
|------|:--:|:--:|
| GCM_C | C | +0.3~0.5 |
| GCM_L2 | L2 | +0.47 (exp38 E1 复现) |
| GCM_D3 | D3 | ≈0 |
| GCM_G2 | G2 | +0.35 (exp38 F9 复现) |
| GCM_S5 | S5 | ? |

`AR_USE_GCM=1 AR_LEARNING_RATE=0.0005`

---

## 四、实验统计

| Phase | 组件 | 退化覆盖 | 组数 | 实现状态 |
|:--:|------|:--:|:--:|:--:|
| A | Swin 基线 | 9 | 9 | ✅ 已有 |
| B | SimpleGate | 9 | 9 | ✅ 已实现 |
| C | SCA | 9 | 9 | ✅ 已实现 |
| D | SG+SCA | 9 | 9 | ✅ 已实现 |
| E | SwiGLU | 9 | 9 | ✅ 已有 `AR_ACTIVATION=swiglu` |
| F | LearnableSkip | 9 | 9 | ✅ 已有 `AR_SKIP_RSTB=learnable` |
| G | GDFN | 5 | 5 | 🔧 需实现 |
| H | FPro-lite | 5 | 5 | 🔧 需实现 |
| I | FiLM-GCM | 5 | 5 | ✅ 已有 `AR_USE_GCM=1` |
| **合计** | **9 组件** | | **75** | |

---

## 五、预期结果对比

| 组件 | 参数量 | 主要收益退化 | Best Δ | Worst Δ | 安全? |
|------|:--:|------|:--:|:--:|:--:|
| SimpleGate | -32K | D3/S5 | +0.3~0.5 | ≈0 | ✅ |
| SCA | +512 | C/L2/G2 | +0.3~0.5 | ≈0 | ✅ |
| SG+SCA | -32K | D3/L2 | 叠加 | ≈0 | ✅ |
| SwiGLU | +98K | D3/S5 | +0.2~0.4 | ≈0 | ✅ |
| LearnableSkip | +4 | 全域微益 | +0.1~0.2 | ≈0 | ✅ |
| GDFN | +50K | D3/S5 | +0.3~0.5 | ? | ? |
| FPro-lite | +5K | C/L2 | +0.3~0.5 | ? | ? |
| FiLM-GCM | +26K | C/L2/G2 | +0.47 | ≈0 | ✅ |

---

## 六、执行

```bash
# Layer 1 (Phase A-F): 54 组, 可直接启动
python3 exp39/scripts/generate_layer1.py
bash scripts/exp_launcher.sh start exp39/scripts/tasks_layer1.txt 1,2,3,4,5,6,7

# Layer 2 (Phase G-I): 21 组, 实现后启动
# 需先实现: AR_USE_GDFN=1, AR_USE_FPRO=1
python3 exp39/scripts/generate_layer2.py
bash scripts/exp_launcher.sh start exp39/scripts/tasks_layer2.txt 1,2,3,4,5,6,7
```

**预计墙钟**: Layer1 54组 ~40min + Layer2 21组 ~15min ≈ **55min**
