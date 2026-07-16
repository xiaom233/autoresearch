---
name: image-degradation-simulator-simple
description: Simplified image degradation simulator. Each degradation category has only one representative function. Judgment is simplified — direct PSNR testing replaces complex signal analysis. Use when subtype diversity is not needed.
---

# Image Degradation Simulator — Simple

**每种退化类别仅保留一个代表函数，判断逻辑直接 PSNR 测试，不做复杂信号分析。**

🔴 **输出约束：predicted_params.json 的 pipeline 只能使用下方退化库中列出的 12 个代表函数。禁止输出任何非代表函数（如 blur_motion、noise_speckle 等）。**

**同图模式优先**：实验生成挑战时使用 `--same-image` 标志。

## 退化库（每类一个代表）

| 类别 | 代表函数 |
|------|------|
| blur | `blur_gaussian` |
| noise | `noise_gaussian_RGB` |
| compression | `compression_jpeg` |
| brighten | `brightness_brighten_shfit_RGB` |
| darken | `brightness_darken_shfit_RGB` |
| contrast_strengthen | `contrast_strengthen_scale` |
| contrast_weaken | `contrast_weaken_scale` |
| saturate_strengthen | `saturate_strengthen_HSV` |
| saturate_weaken | `saturate_weaken_HSV` |
| oversharpen | `oversharpen` |
| pixelate | `pixelate` |
| quantization | `quantization_median` |

用法: `add_distortion(img, severity, distortion_name)`，uint8 RGB，severity 1-5。

## 🔴 唯一工具脚本

| # | 脚本 | 用途 |
|---|------|------|
| 1 | `test_candidate.py --target X --clean C --pipeline "fn:sev,..." --thinking "..." --reflection R` | 施加退化 + PSNR + 更新 reflection |

**只用这一个脚本。** 不需要 run_full_analysis.py、peeled_noise_check.py、verify_signals.py 等复杂信号分析工具。

## 简化判断工作流（直接 PSNR 测试）

```
对每个退化类别，用代表函数直接 PSNR 测试:

Step 1: 逐类别探测 (每类 ≤2 次)
  for 每个类别:
    test_candidate.py 代表函数:3  (先测中等严重度)
    ├─ PSNR > clean_vs_target (有改善) → 该类别存在
    │   → test_candidate.py 代表函数:1-5 (批量找最优 severity)
    └─ PSNR ≈ baseline (无改善) → 该类别不存在，跳过

Step 2: 组合已检测到的退化 (≤2 次)
  按物理管线顺序排列:
    noise→blur→compression→global (传感器→光学→存储→ISP)
    blur→noise→compression→global (备选)
  test_candidate.py 测试 1-2 种排列

Step 3: 保存 → predicted_params.json + reflection.json
```

**预算**: ≤(12×2 + 2) = **≤26 次**，但通常 5-10 次（多数类别不存在）。

### 判断规则

```
类别存在判断:
  PSNR(代表函数:3) - PSNR(clean, target) > 1dB → 存在
  否则 → 不存在

Severity 选择:
  对存在的类别，test 1-5，取 PSNR 最高者

PSNR 不可靠场景 (含 noise 时):
  noise 存在时，PSNR 对非 noise 退化不可靠
  → noise 的 severity 用噪声统计 (std 匹配)
  → 非 noise 退化信任初始 PSNR 改善信号
```

## 管线顺序规则

```
固定顺序: noise → blur → compression → global
仅此一种排列 (简化版不枚举多种排列)
```

## 判定标准

| verdict | 条件 |
|------|------|
| LIKELY | 确定性退化 PSNR >= 35dB |
| UNCERTAIN | PSNR 20-35dB，或含 noise |
| POOR | 所有候选 PSNR < 20dB |

## 强制交付物

| 文件 | 内容 |
|------|------|
| `predicted_params.json` | `{"pipeline":[...], "analysis":{"verdict":"...", "psnr":N}}` |
| `reflection.json` | `{"initial_analysis":..., "iterations":[...], "final_decision":{...}}` |

## 🚫 绝对禁止

1. **不允许写任何 Python 脚本文件**
2. **禁止使用 run_full_analysis.py / peeled_noise_check.py 等信号分析脚本** — 只用 test_candidate.py
3. **禁止枚举所有退化组合** — 只测探测到的类别
4. **禁止 `run_in_background: true`**
5. **禁止凭记忆判定** — 判定前 Read SKILL.md
6. 🔴 **禁止在 predicted_params.json 中输出 12 个代表函数之外的退化**（如 blur_motion、noise_speckle 等）— pipeline 只能含退化库列出的 12 个函数

---

## 与完整版 skill 的核心差异

| 维度 | 完整版 | 简化版 |
|------|------|------|
| 退化函数 | 26+ | **12** |
| 噪声子类型区分 | §B 6步检查 | **无，仅 gaussian_RGB** |
| 模糊子类型区分 | MTF 决策树 | **无，仅 gaussian** |
| 信号分析 | run_full_analysis + 多脚本 | **无，直接 PSNR** |
| 判断方式 | 信号驱动 → 候选 → PSNR 验证 | **直接 PSNR 测试** |
| 工具脚本 | 3 个 | **1 个 (test_candidate.py)** |
| 典型测试次数 | ~25 | **~7** |
