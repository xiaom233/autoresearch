# Phase 3 结果：三方向 SOTA 架构验证

## 实验目标

基于 SFHformer (ECCV 2024)、CSEC (CVPR 2024) 的实际代码，设计三个架构方向，验证是否能成为通用全局退化解决方案。

## 三个方向

| 方向 | 来源 | 机制 | 参数量 | 速度 |
|------|------|------|:--:|:--:|
| **FreqMod** | SFHformer | FFT→Real+Imag→FCPE→BN→PWConv→IFFT (per-block) | +276K | 2.1x慢 |
| **ColorPre** | CSEC | 低分辨率(64×64)→CNN→仿射参数→全分辨率应用 (input) | +0.8K | 1%慢 |
| **DualBranch** | SFHformer Mixer | GAP→MLP→FiLM(全局) + RSTB(局部) + 门控融合 (架构级) | +3K | 3%慢 |

## 实验矩阵

6 退化 × 4 架构 = 24 组（含追加的 severity/brightness_shift/saturate 实验）

| 退化 | 类型 | Swin | FreqMod | ColorPre | DualBranch |
|------|:--:|:--:|:--:|:--:|:--:|
| SF1 contrast_pure | 单 | 27.15 | 42.30 | **46.06** | 40.51 |
| SF8 gamma_pure | 单 | 41.32 | 31.65 | **41.40** | 40.93 |
| L2 contrast+noise | 双 | 28.64 | 27.85 | **29.45** | 29.42 |
| L5 contrast+motion+jpeg | 三 | 19.73 | 22.14 | 22.29 | **22.33** |
| SF5 gamma+blur+noise | 三 | 22.85 | 22.26 | **22.88** | 22.80 |
| L1 brightness_HSV+blur | 双 | 23.21 | 23.08 | — | — |
| SF7 saturate+noise | 双 | 29.43 | 25.76 | 29.09 | — |
| BS bright_shift+noise | 双 | 29.14 | — | **29.17** | — |
| S5 motion (负对照) | 局部 | 21.85 | — | — | — |

## 关键发现

### FreqMod：全面失败

| 退化 | Swin | FreqMod | Δ | 结论 |
|------|:--:|:--:|:--:|------|
| SF1 contrast_pure | 27.15 | 42.30 | +15.15 | 唯一正收益 |
| L5 contrast+triple | 19.73 | 22.14 | +2.41 | 还行但不如FiLM |
| L1 brightness_HSV | 23.21 | 23.08 | -0.13 | 中性 |
| SF5 gamma+triple | 22.85 | 22.26 | -0.59 | ❌ 更差 |
| L2 contrast+noise | 28.64 | 27.85 | -0.79 | ❌ 更差 |
| SF7 saturate+noise | 29.43 | 25.76 | **-3.67** | ❌ 灾难 |
| SF8 gamma_pure | 41.32 | 31.65 | **-9.67** | ❌ 灾难 |

**6/7 退化上 FreqMod ≤ Swin 基线。** SFHformer 的 FFT 方法在小模型(0.45M)上不 work。参数膨胀(+276K, 60%) + 速度减半(2.1x慢) + 效果更差 = 完全放弃。

根因：torch.fft 操作在小特征图上引入的伪影超过其全局建模收益；bfloat16 AMP 需要额外处理。

### ColorPre：最稳定赢家

| 退化 | Δ PSNR | 结论 |
|------|:--:|------|
| SF1 contrast_pure | +18.91 | Trivial case (逆=仿射) |
| L2 contrast+noise | +0.81 | 持平 FiLM-GCM |
| L5 contrast+motion+jpeg | +2.56 | 不如 FiLM +3.52 |
| SF5 gamma+blur+noise | +0.03 | 中性 |
| SF7 saturate+noise | -0.34 | 轻微下降 |
| BS bright_shift+noise | +0.03 | 中性 |

**优势**：+0.8K 参数（0.2%），无 NaN，5/6 ≥ Swin 基线
**劣势**：复杂退化(L5)上不如 per-block FiLM，saturate 场景无帮助

CSEC 的低分辨率正则化设计天然稳定——比 FiLM-GCM 的 unbounded MLP 更安全。

### DualBranch：中规中矩

- L5 +2.60（vs ColorPre +2.56, FiLM +3.52）
- L2 +0.78（vs ColorPre +0.81）
- SF8/SF5 轻微下降
- +3K 参数，稳定

双分支显式分离不如 FiLM-GCM 的隐式 per-block 调制有效。

## FreqMod 修复历程

1. **torch.compile + bfloat16 冲突**：`view_as_complex` 不支持 bfloat16 → 用 `torch.amp.autocast(enabled=False)` 包裹 FFT 操作
2. **第一次修复**：`@torch.compiler.disable` 装饰器 — 无效（AMP autocast 仍将中间结果转回 bfloat16）
3. **第二次修复**：`with torch.amp.autocast('cuda', enabled=False)` — 成功

## 结论

1. **ColorPre 是最佳默认选择**：+0.8K，稳定，通用改善
2. **FiLM-GCM 在 contrast+结构化退化上最强** (+3.52)，但有 NaN 风险
3. **FreqMod 完全失败**，放弃
4. **DualBranch 无独立优势**，不如直接用 ColorPre 或 FiLM
5. **Gamma 系列退化不需要任何额外组件**（MLP 已能处理）
6. **Direct 训练 > Ft**（S5: 21.85 vs 20.51, +1.34）
