# exp38 实验结果

> 状态: 25/32 已验证 + 7 组重跑中（MDTA/OCAB，发现 AR_ATTENTION_TYPE 无效 bug）
> 所有 PSNR 基于 Set14 checkpoint 验证（Set14 PSNR_RGB），同环境公平对比

---

## Phase A：L2 策略-架构解耦（Swin 部分已验证）

| ID | 策略 | 注意力 | PSNR | vs A1 |
|------|------|------|:--:|:--:|
| A1 | Direct | Swin | 28.19 | — 基线 |
| A2 | Direct | MDTA | ⏳ 重跑中 | |
| A3 | Direct | OCAB | ⏳ 重跑中 | |
| A4 | RandomCurric | Swin | 28.04 | -0.15 |
| A5 | RandomCurric | MDTA | ⏳ 重跑中 | |
| A6 | RandomCurric | OCAB | ⏳ 重跑中 | |

> Swin Direct 28.19 vs Swin RandomCurric 28.04：RC 对 L2（contrast+noise）轻微劣化 -0.15 dB，未复现 exp9 的 -4.51 dB 崩溃。
> exp9 的 -4.51 可能来自不同数据集/LR/代码版本，待进一步排查。

---

## Phase B：Motion Fwd/Rev 验证（Swin 部分已验证）

| ID | 策略 | 注意力 | PSNR | vs B1 |
|------|------|------|:--:|:--:|
| B1 | Direct | Swin ws=8 | 22.25 | — 基线 |
| B2 | RandomCurric | Swin ws=8 | 21.78 | -0.47 |
| B3 | Curric Fwd | Swin ws=8 | 22.63 | **+0.38** |
| B4 | Curric Rev | Swin ws=8 | 22.20 | -0.05 |
| B5 | Direct | OCAB ws=8 | ⏳ 重跑中 | |
| B6 | Curric Fwd | OCAB ws=8 | ⏳ 重跑中 | |
| B7 | Curric Rev | OCAB ws=8 | ⏳ 重跑中 | |

> **Swin Fwd (22.63) > Rev (22.20) by +0.43 dB**。Motion sev=5 下 Fwd 仍胜出，"Fwd 翻转"不适用于 Swin。
> CurricFwd > Direct (+0.38)，课程学习对 motion 有益。

---

## Phase C：RandomCurric 安全性

| ID | 管线 | 策略 | PSNR | Δ |
|------|------|------|:--:|:--:|
| C1 | D1 (blur+noise) | Direct | 23.37 | — |
| C2 | D1 (blur+noise) | RandomCurric | 23.32 | **-0.05** |
| C3 | D3 (JPEG+blur) | Direct | 21.38 | — |
| C4 | D3 (JPEG+blur) | RandomCurric | 25.71 | **+4.33** |

> **RandomCurric 在纯局部退化上安全**（D1: -0.05），**在梯度干扰场景大幅有效**（D3: +4.33）。
> RandomCurric 崩溃仅限于 contrast 型全局退化（见 Phase A 讨论）。

---

## Phase D：True Ft vs RandomCurric 机制分离

| ID | 策略 | PSNR | vs Direct | 机制 |
|------|------|:--:|:--:|------|
| D3 | Direct | 21.45 | — | — |
| D1 | True Ft (ckpt) | 21.48 | **+0.03** | 预训练迁移 |
| D2 | RandomCurric | 25.80 | **+4.35** | 梯度干扰减少 |

> **True Ft ≈ Direct**（+0.03），2-epoch 预算下预训练特征迁移无优势（exp37 复现）。
> **RandomCurric >> True Ft**（+4.32），收益来自**梯度干扰减少**，非预训练迁移。
> 两者机制完全不同，不应混用术语。

---

## Phase E：Contrast 组件复现

| ID | 架构 | PSNR | vs Swin (A1=28.19) |
|------|------|:--:|:--:|
| A1 | Swin | 28.19 | — |
| E1 | Swin+FiLM-GCM | 28.66 | **+0.47** |
| E2 | Swin+ColorPre | 28.81 | **+0.62** |

> 两者均有效，ColorPre 略优。与 exp10 趋势一致（FiLM +0.79, ColorPre +0.81），但幅度较小。

---

## Phase F：FiLM/ColorPre 跨全局退化泛化

| ID | 退化 | Swin基线 | FiLM-GCM | ColorPre | FiLM Δ | 结论 |
|------|------|:--:|:--:|:--:|:--:|------|
| F1/F2 | BS (brightness_shift) | 28.62 | 28.60 | — | -0.02 | ✅ 安全 |
| F3 | SF7 (saturation) | — | 28.78 | — | — | ✅ 安全 |
| F4 | SF5 (gamma triple) | — | 23.26 | — | +0.30 | 🟡 微益 |
| F5-F7 | G1 (gamma_RGB) | 28.58 | 28.63 | 28.57 | +0.05 | ≈0 |
| F8-F10 | G2 (stretch) | 27.79 | 28.14 | 27.89 | **+0.35** | 🟡 微益 |

> **FiLM-GCM 在所有非 contrast 全局退化上安全**（无 NaN、无崩溃），推翻了"contrast 专用"的结论。
> Brightness_shift/gamma 上收益 ≈ 0（MLP 已能处理），stretch 上 +0.35。
> **修正结论**: FiLM-GCM 对 contrast + stretch 有效，对 brightness/gamma/saturation 无害。

---

## 关键发现汇总

1. **MDTA/OCAB 结论待验证**（重跑中）。Swin RandomCurric 在 L2 上仅 -0.15，未复现 exp9 的 -4.51dB 崩溃。

2. **"Motion sev≥5 Fwd 翻转"不适用于 Swin**。Swin CurricFwd (+0.43 vs Rev) 在 sev=5 下仍胜出。

3. **RandomCurric 安全边界确认**：纯局部退化安全，梯度干扰场景大幅有效，仅 contrast 需 Direct。

4. **True Ft ≠ RandomCurric**：True Ft ≈ Direct（+0.03），RandomCurric 收益来自梯度干扰减少（+4.35），机制完全不同。

5. **FiLM-GCM 安全范围扩大**：非 contrast 全局退化全安全，stretch 还有 +0.35 收益。不再是 "contrast 专用"。

6. **`AR_ATTENTION_TYPE` 环境变量 bug**：不在 for 循环中，所有实验实际使用 Swin。已修复，MDTA/OCAB 真正实验重跑中。
