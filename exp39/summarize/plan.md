# exp39: 新架构组件探索 — 退化特异性 vs 跨退化泛化

> 目标：测试 3 个 0 参数新组件在 9 种退化上的特异性 + 泛化安全性
> 对比：全部 vs Direct+Swin 基线

---

## 一、测试组件

| # | 组件 | 来源 | 参数量 | 机制 |
|:--:|------|------|:--:|------|
| 1 | **SimpleGate** | NAFNet (ECCV 2022) | **0** | 通道对半分 → 逐元素相乘，替代 GELU |
| 2 | **SCA** | NAFNet | **~0** | GAP → L2 Norm → 通道缩放，无激活/无 Sigmoid |
| 3 | **SG+SCA** | 组合 | **~0** | SimpleGate + SCA 联合 |

---

## 二、退化测试矩阵（9 种，覆盖 1/2/3 步）

| # | 管线 | 步数 | 类型 | 预期敏感 |
|:--:|------|:--:|------|:--:|
| C | contrast_scale(3) | 1 | 纯对比度 | SCA |
| N | noise_gaussian_RGB(3) | 1 | 纯噪声 | ≈0 |
| B | blur_gaussian(3) | 1 | 纯模糊 | ≈0 |
| L2 | noise(3)+contrast_scale(3) | 2 | 全局+噪声 | SCA? |
| D3 | JPEG(3)+blur_lens(4) | 2 | 梯度干扰 | SimpleGate |
| G2 | noise(3)+stretch(3) | 2 | 非contrast全局 | SCA? |
| BS | noise(3)+brightness_shift(3) | 2 | 简单全局 | ≈0 |
| S5 | motion(5)+noise(1)+JPEG(1) | 3 | 复杂mixed | SimpleGate |
| SF5 | blur(3)+noise(3)+gamma(3) | 3 | gamma triple | ≈0 |

> 单退化 (C/N/B) 验证"组件对单独退化有效还是需要复合退化才触发"
> 三退化 (S5/SF5) 验证"组件在复杂场景是否退化"

---

## 三、实验矩阵（9 退化 × 4 架构 = **36 组**）

### 统一配置

```
EPOCH_BUDGET=2  LR=5e-4  BATCH_SIZE=16  EMBED_DIM=64
WINDOW_SIZE=8  ATTENTION_TYPE=swin  LOSS_FN=l1
所有实验 Direct 训练
```

### Phase A: SimpleGate（9 组）

**假设**: 门控激活减少梯度干扰，多步退化 (D3/S5) 受益最大。

| ID | 退化 | 预期 Δ |
|------|------|:--:|
| SG_C | contrast_scale(3) | ≈0 (单步无干扰) |
| SG_N | noise_gaussian_RGB(3) | ≈0 |
| SG_B | blur_gaussian(3) | ≈0 |
| SG_L2 | noise+contrast | ≈0 |
| SG_D3 | JPEG+blur | **有益** (梯度干扰) |
| SG_G2 | noise+stretch | ≈0 |
| SG_BS | noise+brightness | ≈0 |
| SG_S5 | motion+noise+JPEG | **有益** (三退化) |
| SG_SF5 | blur+noise+gamma | ≈0 |

### Phase B: SCA（9 组）

**假设**: 通道注意力增强全局退化，contrast/stretch 受益最大。

| ID | 退化 | 预期 Δ |
|------|------|:--:|
| SCA_C | contrast_scale(3) | **有益** (通道级变换) |
| SCA_N | noise_gaussian_RGB(3) | ≈0 |
| SCA_B | blur_gaussian(3) | ≈0 |
| SCA_L2 | noise+contrast | **有益** |
| SCA_D3 | JPEG+blur | ≈0 (空间退化) |
| SCA_G2 | noise+stretch | **有益** (通道级) |
| SCA_BS | noise+brightness | ≈0 (shift 太简单) |
| SCA_S5 | motion+noise+JPEG | ≈0 |
| SCA_SF5 | blur+noise+gamma | ≈0 (gamma 太简单) |

### Phase C: SG+SCA 联合（9 组）

**假设**: SG 减少梯度干扰 + SCA 增强通道交互，可能叠加。

| ID | 退化 | 预期 Δ |
|------|------|:--:|
| SSC_C | contrast_scale(3) | SCA 主导 |
| SSC_N | noise_gaussian_RGB(3) | ≈0 |
| SSC_B | blur_gaussian(3) | ≈0 |
| SSC_L2 | noise+contrast | SCA 主导 |
| SSC_D3 | JPEG+blur | SG 主导 |
| SSC_G2 | noise+stretch | SCA 主导 |
| SSC_BS | noise+brightness | ≈0 |
| SSC_S5 | motion+noise+JPEG | SG 主导 |
| SSC_SF5 | blur+noise+gamma | ≈0 |

### Phase D: Swin 基线（9 组）

| ID | 退化 | 备注 |
|------|------|------|
| SW_C | contrast_scale(3) | — |
| SW_N | noise_gaussian_RGB(3) | — |
| SW_B | blur_gaussian(3) | — |
| SW_L2 | noise+contrast | exp38 A1=28.19 参考 |
| SW_D3 | JPEG+blur | exp38 C3=21.38 参考 |
| SW_G2 | noise+stretch | exp38 F8=27.79 参考 |
| SW_BS | noise+brightness | exp38 F1=28.62 参考 |
| SW_S5 | motion+noise+JPEG | exp38 B1=22.25 参考 |
| SW_SF5 | blur+noise+gamma | — |

---

## 四、评估

### 退化特异性

每个组件在匹配退化上的收益量级 vs Swin。

### 跨退化泛化安全性

```
每个组件 × 9 退化:
  Best Δ   = 最大正收益
  Worst Δ  = 最大负收益 (是否在任何退化上有害)
  Win Rate = Δ>0 的退化比例
  Safe?    = Worst Δ ≥ -0.3 (无显著有害)
```

| 组件 | Best Δ | Worst Δ | Win Rate | 安全? |
|------|:--:|:--:|:--:|:--:|
| SimpleGate | | | | |
| SCA | | | | |
| SG+SCA | | | | |

### 与 exp38 组件对比

| 组件 | 参数量 | L2 Δ | 泛化安全? |
|------|:--:|:--:|:--:|
| ColorPre | +0.8K | +0.62 | ❌ saturation 有害 |
| FiLM-GCM | +26K | +0.47 | ✅ 全局安全 |
| SimpleGate | **0** | ? | ? |
| SCA | **~0** | ? | ? |
| SG+SCA | **~0** | ? | ? |

---

## 五、总结

| 项目 | 数值 |
|------|:--:|
| 退化类型 | 9 (3 单步 + 4 双步 + 2 三步) |
| 架构变体 | 4 (SG / SCA / SG+SCA / Swin基线) |
| **总实验数** | **9 × 4 = 36 组** |
| 预计墙钟 (7 GPU) | ~25-30 min |

---

## 六、执行

```bash
# 1. 准备退化管线
# 复用: exp38/degradation/{L2_dual,D3_dual,S5_triple,G2_dual,BS_dual,SF5_triple}.json
# 新建: C_pure.json, N_pure.json, B_pure.json (单退化)

# 2. model.py 添加 SimpleGate/SCA 开关
#    AR_USE_SIMPLE_GATE=1  AR_USE_SCA=1

# 3. 生成 + 启动
python3 exp39/scripts/generate.py
bash scripts/exp_launcher.sh start exp39/scripts/tasks.txt 1,2,3,4,5,6,7

# 4. 汇总
python3 exp39/scripts/collect.py
```
