---
name: image-degradation-simulator
description: Use this skill whenever the user needs to analyze degraded images, identify distortion types and severity, simulate image degradation using x_distortion algorithms, or reproduce degradation effects on clean images. Trigger when the user mentions image degradation, distortion simulation, x_distortion, degrading clean images, matching image quality degradation, noise/blur/compression analysis, reproducing image distortions, "make this image look like that one", or understanding what degradations are present in an image.
---

# Image Degradation Simulator

Analyze degraded images, identify present distortion types and their severity.

**同图模式优先**：实验生成挑战时使用 `--same-image` 标志。

## 🔴 盲识别标准工具集（仅 3 个脚本）

Agent 只使用以下 3 个脚本，禁止使用其他分析脚本：

| # | 脚本 | 用途 | 频率 |
|---|------|------|:--:|
| 1 | `run_full_analysis.py --target X --clean C --output O` | 信号报告（整合 noise_prior + blur_kernel + global + compression） | 1次/挑战 |
| 2 | `peeled_noise_check.py --target X --clean C --pipeline "fn:sev,..."` | 剥离确定性退化后的 §B 自动诊断 | 1次/挑战 |
| **3** | **`test_candidate.py --target X --clean C --pipeline "..." --thinking "..." --reflection R`** | **施加退化 + PSNR/信号验证 + 自动更新 reflection.json** | **每个候选** |

**禁止使用**: noise_prior.py (被 run_full_analysis 内部调用), analyze_degradation.py (旧版), global_degradation_analyzer.py (旧版), model_diagnosis.py (反思用), apply_multi.py + compare_degradation.py (已被 test_candidate.py 替代)

### 标准工作流（3 步，~10 次 test_candidate.py）

```
Step 0: run_full_analysis.py → full_analysis.json
        → step1: block_boundary, unique_G (compression 0%FP)
        → step3: blur_subtype, residual_anisotropy (FFT), pca_ratio, glass_score
        → step2: noise_prior_sigma
        → step4: variance_ratio, mean_shift (global)
        → 手动写入 reflection.json: initial_analysis (mode, sigma, suspected/ruled_out)

Step 0.5: 🆕 Qwen 噪声门 (exp37 消融: Qwen 噪声存在性判断优于Skill)
        IF sigma 边界 (2 < sigma < 10) 或信号矛盾:
          → 发送原图给 Qwen: "这些图像中是否存在随机噪声(gaussian/speckle/poisson/impulse)? 只需回答 YES/NO"
          → Qwen YES → 继续 Step 3 噪声精细诊断
          → Qwen NO  → 跳过 Step 3, 标记 NO_NOISE
          → 预算: 1次 API 调用, 仅边界/矛盾时触发
        ELSE:
          → sigma < 2 → NO_NOISE (跳过Step3)
          → sigma > 10 → 确认有噪声 (进入Step3)

Step 1: Compression (IF block_boundary>1.1)
        test_candidate.py JPEG 1-5 → PSNR 选最优 → pipeline_comp

Step 2: Blur — 🔴 决策树, 不盲目 PSNR
        IF residual_anisotropy > 1.7 → motion (94% recall, 0% FP, 384-case校准)
           → test_candidate.py 调 severity (1-2次)
        ELIF glass_score > 0.25 + sigma < 3 → glass
           → test_candidate.py 调 severity (1-2次)
        ELSE → gaussian (lens 为 alternative, MTF已知不可区分)
           → test_candidate.py 调 severity (1-2次)

Step 3: peeled_noise_check.py --pipeline "jpeg:X,blur:Y"
        → 🏆 自动噪声诊断
        → test_candidate.py 验证 (1次)
        → IF sigma < 2 → NO_NOISE

Step 4: 手动写入 final_decision → 保存 predicted_params.json
```

test_candidate.py 自动写入: iterations, psnr_ranking, decision
Agent 手动写入: initial_analysis, noise_6step_check, final_decision

**预算: 5(JPEG) + 2(blur) + 1(noise) + 2(顺序) = ≤10 次 test_candidate.py (σ<2 时)**

### 🔴 验证模式自动切换

```
run_full_analysis.py → noise_prior_sigma
  ├─ σ < 2 (无噪声): PSNR 可用
  │    compare_degradation.py → PSNR > 40dB → 确认
  │
  └─ σ ≥ 2 (有噪声): PSNR 不可靠 → 切换到信号验证
       verify_signals.py --target X --simulated S --clean C --pipeline "..."
       → MATCH:    全部信号匹配 → 退化正确, severity 正确
       → PARTIAL:  多数信号匹配 → 类型正确, 🔴 severity 扫 ±1
       → WEAK:     少数匹配 → 🔴 severity 扫 ±2, 无改善再换类型
       → MISMATCH: 无匹配 → 🔴 先扫 severity ±2, 仍不行换子类型
```

### 🔴 Severity 扫描（强制，不可跳过）

**当 verify_signals 返回 PARTIAL/WEAK/MISMATCH 时，必须先扫 severity，再换类型：**

```
for sev in [est-2, est-1, est+1, est+2]:  # 限制 1-5
    same_function, different_severity → verify_signals
    if MATCH → 立即停止, 保存
    if 改善(score↑) → 更新 best

全部 sev 未达 MATCH → 取 best score 的 severity → 然后才换子类型
```

**原因**: exp34 审计发现 — 多数 rejection 是 severity 不对（如 sev=1 预测给实际 sev=4 的退化），不是类型不对。

### 🔴 噪声耦合时切换信号验证（禁止 PSNR）

**当 noise 存在时（sigma > 2），PSNR 对所有退化都不可靠**，因为 noise seed 失配会导致像素级对比失真。必须切换到噪声鲁棒的信号验证：

| 退化 | 噪声下 PSNR | 替代验证信号 | 阈值 |
|------|:--:|------|:--:|
| blur | ❌ | Wiener 核自洽: `corr(h_target, h_simulated) > 0.9` | 相关系数 |
| | | `spectral_slope` 差值: `|slope(target) - slope(simulated)| < 0.08` | 噪声仅偏移 ±0.03 |
| JPEG | ❌ | `block_boundary` 差值: `|bb(target) - bb(simulated)| / bb(target) < 5%` | 噪声同向衰减 8×8 块 |
| | | DCT zero_ratio 差值: `|zr(target) - zr(simulated)| < 0.05` | — |
| contrast | ❌ | noise-corrected `variance_ratio`: `|var(target)-var(sim)| / var(target) < 5%` | 前提: sigma 已知 |
| | | `histogram_shape` 相关系数 > 0.95 | 噪声平滑直方图不改变结构 |
| brightness | ✓ (σ<10) | `mean_shift` 差值: `|ms(target)-ms(sim)| < 0.5/255` | 噪声均值为零 |
| noise | — | §B 6步逐一对比 + `sigma` 误差 < 15% | 剥离确定性后重检 |
| | | 剥离后残差结构检测: `autocorr_FWHM < 3px` + FFT 无低频突起 | 残差应无结构 |

**验证链（有噪声时强制执行）**：
```
1. 验证确定性部分 (blur/JPEG/contrast): 全部用信号验证，不用 PSNR
2. 确认剥离干净: autocorr_FWHM < 3px + block_boundary < 1.02
3. §B on peeled residual → noise 类型+severity
4. 仅当所有信号验证通过 → 保存
   任一不通过 → 修正确定性参数 → 重剥离 → 重回 Step 1（最多 2 轮）
```

**PSNR 仍有价值的场景（噪声不存在时）**：
- σ < 2: PSNR > 40dB → 确定性退化高置信度确认
- σ < 2: PSNR 30-40dB → severity 可能在 ±1 范围内
- σ ≥ 2: 放弃 PSNR，全部走信号验证

### 🔴 管线顺序物理约束

**物理事实**:

```
传感器噪声 → 光学模糊 → ISP(对比度/饱和度/量化) → JPEG 存储
                ↑
           brightness 可在此前 (环境光, 模拟传感器前照明条件)
```

- `noise < blur`: 传感器噪声先于光学模糊（不可违反）
- `noise+blur < JPEG`: 存储是最后一步（不可违反）
- `brightness ↔ noise`: brightness **随机** priority=0 (环境光, blur前) 或 priority=2 (ISP亮度, blur后)
- `contrast/saturation/quantization ↔ JPEG`: ISP 后期，顺序取决于是否二次编辑

**PSNR 测试顺序（按概率从高到低）**：

```
含 brightness (环境光):
  1. brightness→noise→blur→global→JPEG   (自然光→传感器→光学→ISP→存储)
  2. noise→brightness→blur→global→JPEG   (传感器→环境光→光学→ISP→存储)

不含 brightness (标准相机管线):
  3. blur→noise→global→JPEG   (相机直出, >80% 场景)
  4. noise→blur→global→JPEG   (相机直出, 噪声先于光学模糊)
  5. blur→noise→JPEG→global   (后期编辑)
  6. noise→blur→JPEG→global   (后期编辑 + 噪声优先)
```

前两种覆盖含 brightness 的场景，前四种覆盖绝大多数。

**blind_challenge.py 已强制**：`noise/brightness(priority=0) → blur(priority=1) → contrast/saturation/quantization/JPEG(priority=2)`，同优先级随机 shuffle。

### 🔴 信号信任规则（禁止 PSNR 覆盖）

| 信号 | 可靠性 | 来源 | 规则 |
|------|:--:|------|------|
| block_boundary > 1.1 | **0% FP** | exp17/18 | → JPEG 确认 |
| blur Hybrid DT (depth=3+4) | **83.4%** | 3840-case, 10图 | → 默认 depth=3, glass_detected→depth=4 |
| noise §B DT (depth=4, blur→noise) | **87.4%** | 3840-case, GT剥离 | → 信任 noise 类型（仅在 blur→noise 顺序） |
| noise §B (noise→blur) | 54.7% ≈ 随机 | 同上 | → **不可用**，标记 UNCERTAIN |
| Wiener 核自洽 (噪声下) | **>90%** | 3840-case, blur 正确性验证 | → 噪声耦合时替代 PSNR 验证 blur |
| spectral_slope 差值 (噪声下) | **±0.03** | 噪声偏移远小于 blur 变化 | → 噪声耦合时替代 PSNR 验证 blur severity |
| block_boundary 差值 (噪声下) | **<5%** | 噪声同向衰减 | → 噪声耦合时替代 PSNR 验证 JPEG |

**PSNR 仅用于: 无噪声时(σ<2)的 severity 确认。有噪声时全部验证走信号模式（见上方噪声耦合信号验证表）。**

### 🔴 Verdict 强制规则（来源: exp34/37 审计）

**🔴 exp37 修正: verdict 按退化步骤独立评估，不再搞整体一刀切。**

```
确定性退化 (blur/compression/contrast/brightness/saturation):
  PSNR >= 40dB → LIKELY  (可触发专用组件)
  PSNR 30-40dB → UNCERTAIN (severity微调)
  PSNR < 30dB → UNCERTAIN/POOR

噪声退化 (noise_*):
  verify_signals MATCH → UNCERTAIN (噪声seed失配, 永不LIKELY)
  verify_signals PARTIAL → UNCERTAIN
  其余 → POOR

整体 overall_verdict:
  所有步骤 LIKELY → LIKELY
  任一 POOR → POOR
  其余 → UNCERTAIN
```

**exp34 教训**: 噪声下 PSNR 是随机数，noise 步骤永不标 LIKELY。
**exp37 教训**: 噪声步骤 UNCERTAIN 不应阻止 contrast 步骤 LIKELY 触发 FiLM-GCM。

### 🔴 verify_signals.py 用法

```bash
.venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/verify_signals.py \
  --target <degraded.png> --simulated <candidate.png> --clean <clean.png> \
  --pipeline "blur_gaussian:3,noise_gaussian_RGB:2"
```

**输出解读**（Agent 直接读取文字摘要）：

| Verdict | 含义 | 动作 |
|------|------|------|
| MATCH | 所有信号匹配 | → 退化参数正确，保存 |
| PARTIAL | 多数信号匹配 | → 类型正确，severity 微调 |
| WEAK | 少数信号匹配 | → 类型可能错误，换假设或降低置信度 |
| MISMATCH | 无信号匹配 | → 退化错误，重做假设 |

**核心对比信号**（阈值来自 3840-case 校准）：
- blur: `spectral_slope` 差值 < 0.08, Wiener 核相关 > 0.85
- noise: `sigma` 误差 < 20%, 残差 autocorr FWHM < 4px
- compression: `block_boundary` 差值 < 8%
- global: `mean_shift` 差值 < 0.005, `variance_ratio` 差值 < 8%

### 🔴 已知失败模式与反思策略

### 🔴 失败模式与自动反思机制

#### 盲识别阶段反思（保存 predicted_params 前触发）

| 触发条件 | 失败模式 | 发生率 | 反思动作 |
|------|------|:--:|------|
| blur DT 预测 gaussian + PSNR lens 高 ≥5dB | lens 被误判为 gaussian | ~15% | 替换为 lens，保存 alternative |
| blur DT 预测 motion + PSNR gaussian 高 ≥5dB | gaussian 被误判为 motion (DT FP) | ~18% | PSNR 胜出，替换为 gaussian |
| noise→blur 顺序 PSNR 胜出 | 噪声类型不可诊断 | 54.7% | 标记 UNCERTAIN，noise 类型不做判断 |
| peeled §B extreme_pct>50% + noise_prior sigma<5 | extreme_pct 假阳性 (JPEG/blur 伪影) | ~30% | 忽略 §B IMPULSE，信任 noise_prior 类型 |
| unique_G<200 + block_boundary<1.1 + JPEG 不改善 PSNR | compression 假阳性 (noise/quantization 导致) | ~20% | 检查 noise/量化是否独立导致 unique_G 减少 |
| DT-PSNR 矛盾 unresolved | 超出 x_distortion 能力或 OOD 退化 | ~5% | POOR，留空 alternatives，等训练 PSNR 仲裁 |

#### 训练后反思（Phase 5 完成后，GT 重评估触发）

| 触发条件 | 可疑根因 | 反思动作 |
|------|------|------|
| GT PSNR < 20 + R0 PSNR > 25 (gap > 5dB) | Blind ID 严重有误 | 🔴 重做盲识别：重跑 run_full_analysis，信号重释 |
| GT PSNR < 30 + verdict=POOR/UNCERTAIN | 盲识别部分错误 | Skill 子 Agent 反思：剥离已知退化→重检 §B/MTF |
| GT PSNR > 35 + verdict=LIKELY | 盲识别大概率正确 | 跳过反思，直接收录 |
| R0 < DFPIR (Specialist 弱于大模型) | 训练策略或架构问题 | 检查 finetune_strategy.md，考虑架构升级 |
| 训练崩溃 (NaN/PSNR<<10) | Ft 在全局退化上崩溃 | 切换 Direct，检查 contrast→Swin 规则 |

#### 反思执行协议

```
1. 盲识别阶段反思:
   - Agent 在保存预测前自动检查上述触发条件
   - 触发时 → 执行对应反思动作 → 更新 predicted_params
   - 仍无法解决 → 标记 UNCERTAIN/POOR，等训练后仲裁

2. 训练后反思:
   - GT 重评估后自动触发
   - 🔴 必须通过 Skill 子 Agent 执行（禁止主 Agent 手动修改 params）
   - 子 Agent: Read SKILL.md → 重跑 run_full_analysis → 信号重释 → ≤5修正假设 → PSNR/信号验证
   - 🔴 判定条件 (区分噪声/确定性):
     预测管线不含 noise → 仿真 PSNR 改善 > 2dB 或 > 35dB → 保存 R1 params
     预测管线含 noise → verify_signals 改善 (WEAK→PARTIAL→MATCH) 或 §B 统计改善 → 保存 R1
     注: 噪声 seed 失配导致 PSNR 不可信, σ估计在blur/JPEG存在时可能不准确,
         直接用预测管线是否含 noise_* 函数判断, 比 σ 更可靠
   - 无改善 → BEYOND_CAPABILITY → 更新 reflection.json
```

## 🚫 绝对禁止（违反立即停止）

以下任何一条都会导致 CPU 100% 持续数小时、结果质量差、实验作废：

1. **不允许写任何 Python 脚本文件**：只使用已有的 Skill 脚本。
2. **禁止 PSNR 枚举搜索** 🔴 exp31 审计: 4/8 Agent 超标, 最高 ~80 次。使用信号驱动二元对比协议（见下方），≤25 PSNR 测试/挑战，3 轮硬终止。
3. **禁止跨类别盲目组合**：每轮只测 ≤ 2 个候选，信号→决策映射表直接给候选，不遍历。
4. **禁止 `run_in_background: true` 启动多个并行搜索**。
5. **禁止编写子进程调用脚本**。
6. **禁止凭记忆做噪声/模糊类型判定** 🔴：噪声判定前必须 `Read SKILL.md offset=42 limit=25`；模糊判定前必须 `Read SKILL.md offset=67 limit=20`。

**正确做法**：run_full_analysis.py 获取信号 → 🔴 逐层剥离 → 信号驱动探索协议 → PSNR 最终验证 → 保存。

---

## 🔴🔴 逐层剥离（v12 — 解决双退化信号污染）

> 来源: exp31 v1/v2/v3 — 三版协议 noise 识别率始终 46%，根因: §B 6步在混合残差上不可靠。
> 核心改变: 确定性退化必须先识别并剥离，然后在剥离后的残差上做 §B 检查。

### 为什么必须剥离

```
❌ 错误: residual = target - clean  (包含 blur + noise 混合)
   → §B 6步在混合残差上: spatial_corr 被 blur 抬高, var_slope 被 compression 扭曲
   → noise_prior wavelet 被 blur 衰减 (sigma 低估 50%+)
   
✅ 正确: 
   Step A: 先识别确定性退化 (compression/blur)
   Step B: simulate(clean, deterministic_pipeline) → simulated
   Step C: peeled_residual = target - simulated
   Step D: 在 peeled_residual 上做 §B 6步 + noise_prior
```

### 强制执行规则

```
1. 如果 run_full_analysis 检测到 compression (unique_G<200 或 block_boundary>1.1):
   → 先确定 compression 参数 → 剥离 → 再做 noise/blur 判断

2. 如果 run_full_analysis 检测到 blur (gm_ratio<0.85 且无强噪声, 或 spectral_slope_ratio>1.08):
   → 先确定 blur 参数 → 剥离 → 再做 noise 判断
   
3. 优先级: compression > blur > noise
   compression 的 block_boundary 是 0% FP 最强信号 → 优先剥离
   blur 的 spectral_slope 在 noise 存在时仍可用 → 次优先
   noise 的 §B 信号最容易被污染 → 最后判断
```

### 剥离后的噪声判断（§B 重跑）🔴 使用 peeled_noise_check.py

剥离确定性退化后，**必须运行 peeled_noise_check.py** 获取 §B 6步数值：
```bash
.venv/bin/python3 .claude/skills/image-degradation-simulator/scripts/peeled_noise_check.py \
  --target <degraded.png> --clean <clean.png> \
  --pipeline "compression_jpeg:3,blur_gaussian:2"
```
输出：
- §B 6步完整数值表（extreme_pct, vm_slope, var_slope, spatial_corr, rgb_ratio, cross_ch_corr）
- noise_prior wavelet sigma 估计
- 🏆 自动噪声类型推断 + severity 建议

**Agent 只需读取输出，不需自己计算。** 根据推断结果选择噪声候选。
⚠️ extreme_pct 会被 JPEG 残留放大 → 结合 noise_prior sigma 判断真实噪声强度。

---

## 🔴 信号驱动探索协议（v12 — 逐层剥离 + 全覆盖）

### 协议总览

```
run_full_analysis.py
       ↓
信号报告 (signals + failure_risks + reflection_strategies)
       ↓
┌──────────────────────────────────────────────────────────┐
│ Round 1: 信号驱动全覆盖 (关键!)                            │
│   每个退化类别基于信号选出 3-4 个 plausible 候选             │
│   ⚠️ 不是 2 选 1 — 必须覆盖主要混淆对                       │
│   噪声: gaussian_RGB + YCrCb + speckle + poisson 至少3种   │
│   模糊: gaussian + lens + motion 至少3种                   │
│   用 PSNR/统计匹配排序，取 top-2 进入 R2                    │
│   预算: ≤12 次 (3-4候选/类别 × 2-3类别)                     │
├──────────────────────────────────────────────────────────┤
│ Round 2: 胜者 vs 跨族替代                                  │
│   R1 top-1 vs R1 top-2 (同类别替代)                        │
│   R1 top-1 vs best cross-family candidate                 │
│   预算: ≤8 次 (4对 × 2)                                     │
├──────────────────────────────────────────────────────────┤
│ Round 3: Severity 微调 + 掩盖推理                           │
│   对 R2 胜者微调 severity ±1 (≤2次)                        │
│   若 PSNR<30 且不含 compression → 掩盖推理测 JPEG 1-5      │
│   预算: ≤5 次                                               │
└──────────────────────────────────────────────────────────┘
总预算: ≤25 次 (12+8+5)。🔴 超限 → 立即终止，标记 UNCERTAIN。
```

### 🔴 决策点强制重读

**在做以下决策前，必须先 Read SKILL.md 对应章节（禁止凭记忆）：**

| 决策点 | 重读内容 | 时机 |
|------|------|------|
| noise 类型判定 | `Read SKILL.md offset=42 limit=25` (§B 6步检查) | Step 2 开始前 |
| blur 类型判定 | `Read SKILL.md offset=67 limit=20` (§C Blur判定) | Step 3 开始前 |
| quantization 判定 | `Read SKILL.md offset=30 limit=15` (§A 残差先行) | 看到 unique_G<200 时 |
| 保存预测前 | `Read SKILL.md offset=24 limit=8` (强制检查清单) | 保存 predicted_params 前 |

**成本**: 每次 Read ~200 tokens，单挑战最多 4 次重读 ≈ 800 tokens。对比当前平均 35 次 PSNR 测试（每次 apply+compare ≈ 5000 tokens compute），开销 < 2%。

### 信号→候选映射表

**run_full_analysis.py 输出 → Agent 决策（直接映射，不枚举）：**

| run_full_analysis 信号 | 决策 | 候选 1 | 候选 2 |
|------|------|------|------|
| `noise_prior sigma > 5` | 噪声存在 | 从 noise_prior 类型推断 | 从 §B 6步统计推断 |
| `rgb_ratio > 1.4 + ric < 0.15` | YCrCb 噪声 | noise_gaussian_YCrCb | noise_gaussian_RGB (对照) |
| `var_slope > 1.0 + vm_slope≈0` | Poisson 噪声 | noise_poisson | noise_gaussian_RGB (对照) |
| `vm_slope > 0.01` (>0.005 for sev≤3) | Speckle 噪声 | noise_speckle | noise_gaussian_RGB (对照) |
| `extreme_pct > 0.3%` | Impulse 噪声 | noise_impulse | noise_gaussian_RGB (对照) |
| `spatial_corr > 0.15 + 无JPEG` | SC 噪声 | noise_spatially_correlated | noise_gaussian_RGB (对照) |
| 以上全不满足 | Gaussian_RGB | noise_gaussian_RGB | — |
| `gm_ratio < 0.85` (无 noise) or `spectral_slope_ratio > 1.08` | Blur 存在 | 从 MTF 信号推断子类型 | blur_gaussian (默认对照) |
| `anisotropy_ratio > 4.0 + h_v_ratio≠1.0` | Motion blur | blur_motion | blur_gaussian (对照) |
| `radial_ratio > 2.0` | Lens blur | blur_lens | blur_gaussian (对照) |
| `glass_score > 0.25` | Glass blur | blur_glass | blur_gaussian (对照) |
| 以上不满足 + blur 存在 | Gaussian blur | blur_gaussian | — |
| `block_boundary > 1.1` | JPEG | compression_jpeg | — |
| `unique_G < 200 + var_slope<1.0` | Compression/Quant | compression_jpeg | quantization |
| `variance_ratio ≠ 1.0` | Contrast/Gamma | contrast_weaken/strengthen | gamma |
| `mean_shift > 0.08 + variance_ratio≈1.0` | Brightness | brightness_shift | — |

### Round 1 信号驱动全覆盖规则 🔴 核心

**这不是 2 选 1。这是信号驱动的全混淆覆盖。**

```
噪声类别 (基于 noise_prior + §B 6步):
  必测: noise_gaussian_RGB (对照基线)
  信号触发候选:
    rgb_ratio > 1.4 + ric < 0.15 → + noise_gaussian_YCrCb
    vm_slope > 0.005 → + noise_speckle  
    var_slope > 1.0 + vm_slope≈0 (在残差上!) → + noise_poisson
    extreme_pct > 0.3% → + noise_impulse
    spatial_corr > 0.15 + 无JPEG → + noise_spatially_correlated
  → 至少测试 3-4 种噪声类型 (覆盖主要混淆对: speckle/gaussian, poisson/gaussian)
  
模糊类别 (基于 gm_ratio / spectral_slope_ratio / MTF):
  必测: blur_gaussian (对照基线)
  信号触发候选:
    anisotropy > 4.0 + h_v_ratio 偏30% → + blur_motion
    radial_ratio > 2.0 → + blur_lens  
    glass_score > 0.25 → + blur_glass
  → 至少测试 3 种模糊类型 (gaussian+lens+motion, 文档已知MTF无法区分)
  
压缩类别:
  必测: compression_jpeg (唯一确定性选项)
  替代: compression_jpeg_2000 (仅在 unique_G 模式不同于JPEG时测)
```

### Round 2 二元对比规则

**对 R1 top-2 进行二元对比。**

1. **同族对比**: R1 top-1 vs R1 top-2（确认该类别最佳子类型）
2. **跨族对比**: R1 top-1 vs best cross-family candidate（排除跨族混淆）
3. **Control test**: 最佳候选 vs baseline（去除该步），确认该步贡献
4. 最多 8 次 (4对 × 2候选)

### Round 3 Severity 微调

1. R2 胜者 ±1 severity (≤4次)
2. 若 PSNR<30 且不含 compression → 掩盖推理: 测试 compression_jpeg 1-5
3. 总 ≤5 次

### 终止条件（满足任一即停止，禁止继续搜索）

| 条件 | 行动 |
|------|------|
| 确定性部分 PSNR > 40dB + 噪声统计匹配 | → LIKELY，保存 |
| 已完成 3 轮探索 | → 标记 UNCERTAIN，保存当前最佳 |
| PSNR 测试 ≥ 25 次 | 🔴 **硬终止**，保存当前最佳，标记 UNCERTAIN |
| 连续 2 轮无改善 (无噪声: PSNR<2dB; 含噪声: 信号级无改善) | → BEYOND_CAPABILITY，保存 R0 |
| 所有 candidate PSNR < 20dB | → POOR，保存全部候选 |

### PSNR 测试预算

```
总预算: ≤ 25 次/挑战

分配:
  Round 1 (信号→候选):   ≤ 8 次 (≤ 2候选/类别 × 4类别)
  Round 2 (二元对比):     ≤ 8 次 (≤ 4 对比 × 2 候选)
  Round 3 (severity微调): ≤ 4 次 (≤ 2 退化 × ±1 severity)
  掩盖推理补充:           ≤ 5 次 (compression_jpeg 1-5)
  
  🔴 超预算 → 立即终止，标记 UNCERTAIN，在 thinking_process.json 中记录原因
```

---

## 🔴 强制检查清单（保存预测前必须逐项确认，不可跳过）

exp17 盲化评估（35 单退化）发现：50% 的失败案例不是阈值/指标问题，而是 **Agent 跳过了应该执行的检查步骤**。此清单强制完成每一步。

### A. 残差先行原则 ⚠️ exp17 头号教训

**在 target 上看到 unique_G 减少或 extreme% 升高时，绝不能直接下结论。必须先算残差。**

```
❌ 错误: 看到 unique_G < 25 → "quantization!"
✅ 正确: 看到 unique_G < 25 → 先算 residual = target - clean
         → residual 上 var_slope > 1.0 → "Poisson 噪声，不是 quantization"
```

**强制规则**: 任何关于 quantization 的判定，必须在计算残差并确认 `var_slope < 1.0`（排除 Poisson）之后才能做出。

### B. 噪声判别步骤强制顺序（不可跳步）

对每个残差，**必须按顺序执行全部 6 项检查**，记录每项结果后再下结论。可辅助使用 wavelet 噪声先验：

```
🆕 0. wavelet:      noise_prior.py --target <t> --clean <c> (同图模式)
      → 仅以下两项已验证可靠 (exp18测试 10/10):
        - gaussian_RGB vs gaussian_YCrCb: 通道 σ 比 > 1.4 → YCrCb ✅
        - 纯噪声 σ 估计: sev=1~2 范围内 ±30% (blur存在时低估, 不可靠)
      → 不可用于: speckle/poisson 检测 (0/10), 混合退化噪声估计

□ 1. impulse:     extreme% = ___ (> 0.3%? → impulse)
□ 2. speckle:      vm_slope = ___ (> 0.01? → speckle; sev=1 时 > 0.005 也考虑)
□ 3. poisson:     var_slope = ___ (> 1.0 AND vm_slope≈0? → poisson)
□ 4. spatial:     spatial_corr = ___ (0.15-0.5? → spatially_correlated)
□ 5. YCrCb:       rgb_ratio = ___ (> 1.4? → YCrCb)
                   🆕 辅助: noise_prior σ_per_channel max/min ratio > 1.4 → YCrCb ✅
□ 6. gaussian:    以上都不满足 → gaussian_RGB
```

**在 reflection.json 中必须记录这 6 项的数值**，不能只写"判断为 gaussian"。

### C. Blur 判定强制规则

```
□ 1. gm_ratio < 0.85 → blur 可能存在（mild blur 区 0.75-0.85 不能排除）
□ 2. 🆕 angular_fft.anisotropy_ratio > 4.0 AND h_v_ratio ≠ 1.0 (>30%偏差) → motion blur
     来源: exp18测试 5/5 motion正确, MTF频域分析对motion最可靠
□ 3. radial_ratio > 2.0 → lens blur 候选，但需 PSNR 验证 (⚠️ 内容依赖, 非可靠判据)
     exp18测试: gaussian↔lens MTF无法区分 (两者MTF曲线形状相同)
□ 4. 非确定性 blur (glass/jitter) PSNR 不会 > 40dB，不要反复调 severity 追求高 PSNR
□ 5. 指标超过阈值 → 信任指标，不要用"直觉"否定
```

### C2. 经验信号速查表 (来源: exp17+exp18, 32 cases)

以下规律从 32 组盲识别实验中系统提取，用于辅助 Agent 判断"退化是否存在"，而非精确识别子类型。

#### 退化存在性判断 (区分纯退化 vs 混合退化)

⚠️ **关键发现**: 信号可靠性在混合退化中急剧下降。必须分场景使用。

| 退化 | 信号 | 阈值 | 纯退化 | 混合退化 | 结论 |
|------|------|------|:--:|:--:|------|
| **comp/quant 存在** | `unique_G` | < 200 | N/A | 召回33% **FP=0%** | ⭐⭐⭐ 从不误报, 误报率0% |
| **global 存在** | `Cr_variance` | < 100 | N/A | **召回100%** FP=17% | ⭐⭐⭐ 混合中仍可靠 |
| **blur 存在** | `gm_ratio` | < 0.85 | 召回100% | 召回71% FP=20% | ⭐⭐ 混合中下降 |
| **noise 存在** | `overshoot_ratio` | > 0.5 | 不可用 | 召回35% **FP=90%** | ❌ 混合中完全失效 |
| **global 存在** | `saturation_mean` | < 50 | 不可用 | 召回33% FP=33% | ❌ 混合中失效 |

**混合退化中信号污染的原因**:
- `gm_ratio`: noise/compression 增加梯度 → 掩盖 blur 的梯度下降 → 漏检
- `overshoot_ratio`: compression 也产生 overshoot → noise 不存在时也 > 0.5 → 大量误报
- `saturation_mean`: 多步退化互相抵消/增强饱和效果 → 不可靠

#### 不可靠的信号 (不要用于判断)

| 信号 | 原因 |
|------|------|
| `flat_region_variance` 判断 noise | 噪声/无噪声均值差仅 30%，大量重叠 |
| `laplacian_variance_ratio` 判断 blur severity | 混合退化时噪声/锐化拉高 laplacian，sev=4 反而 > sev=2 |
| `radial_ratio` 判断 lens blur | 内容依赖 (clean 图本身 radial_ratio 可达 5) |

#### PSNR 适用范围

```
14/32 纯确定性退化 (无 noise) → PSNR 判据可用
18/32 含 noise             → PSNR 对噪声部分无效, 需统计匹配
10/32 混合 (noise+blur)     → 最困难, PSNR + 统计都不可靠
```

#### 关键推论 (仅使用混合退化中验证过的信号)

1. **unique_G < 200 → 一定有 compression 或 quantization** (FP=0%, 纯/混合均可靠)
2. **Cr_variance < 100 → 一定有 global 退化** (混合中召回100%, FP=17%)
3. **gm_ratio < 0.85 → blur 很可能存在** (混合中召回71%, 但 noise/comp 可能掩盖)
4. **overshoot_ratio 在混合退化中不可用于判断 noise** (FP=90%!)
5. **纯退化 (1步) 中信号远更可靠** → 先判断退化步数, 单步退化 PSNR 直接可用

### D. 保存前自检

```
□ 1. 如果预测包含 quantization → 残差 var_slope < 1.0 已确认?
□ 2. 如果预测 noise 子类型 → 6 项检查全部完成并记录?
□ 3. 如果预测 blur 子类型 → radial_ratio / dir_change 数值已记录?
□ 4. 确定性退化 PSNR > 40dB 或非确定性退化有充分指标证据?
□ 5. reflection.json 包含完整的 6 项噪声检查数值?
```

### 多退化并行处理

以上规则针对**单个退化**的搜索。多个独立的退化可以并行处理，按退化复杂度分配：

### Agent 任务分配策略 (仅主 Agent/orchestrator 参考，子 Agent 不可见)

🔴 **Agent 命名和 prompt 严格中性**：不包含退化步数信息。子 Agent 不知道自己的挑战是几步退化。

| 退化步数 | Agent 数 | 每 Agent 处理 | 原因 |
|------|:--:|:--:|------|
| 1 步 | 1/6-8挑战 | 6-8个 | 最简单，处理快 |
| 2 步 | 1/2-3挑战 | 4-5个 | 中等复杂度 |
| 3 步 | 1/1-2挑战 | 6-8个 | 最复杂，需2倍时间 |


## 附录: x_distortion 退化库

| 类别 | 函数 |
|------|------|
| blur | gaussian, motion, glass, lens |
| noise | gaussian_RGB, gaussian_YCrCb, speckle, poisson, impulse |
| compression | jpeg |
| global | brightness(8), contrast(4), saturation(4), quantization_median, quantization_hist |

用法: `add_distortion(img, severity, distortion_name)`，返回 uint8 RGB。

**🔴 仅上表列出的函数可用。** 以下已禁用且不可预测: `quantization_otsu`, `blur_zoom`, `noise_spatially_correlated`, `compression_jpeg_2000`, `oversharpen`, `pixelate`。

## 附录: reflection.json 格式

Agent 只需手动写入 3 个字段，其余由 `test_candidate.py` 自动维护。

### Agent 手动写入

```json
{
  "initial_analysis": {
    "run_full_analysis_summary": {"compression": "...", "blur": "...", "noise": "...", "global": "..."},
    "mode": "PSNR|SIGNAL (sigma=X.X)"
  },
  "noise_6step_check": {"extreme_pct": 0.12, "vm_slope": 0.003, "var_slope": 0.85,
    "spatial_corr": 0.08, "rgb_ratio": 1.05, "cross_ch_corr": 0.12, "diagnosis": "..."},
  "final_decision": {
    "pipeline": [
      {"function": "...", "severity": N, "verdict": "LIKELY|UNCERTAIN|POOR"},
      ...
    ],
    "overall_verdict": "LIKELY|UNCERTAIN|POOR",
    "rationale": "..."
  }
}
```

### test_candidate.py 自动维护

```
iterations[]: {hypothesis(pipeline + thinking), test_result(PSNR/CI或MATCH/PARTIAL等), decision}
psnr_ranking[]: 按 PSNR 降序排列
```

**强制**: noise_6step_check 必须含 6 项数值，不能只写结论。initial_analysis.mode 必须注明 sigma 值。

### 强制字段规则

| 字段 | 何时必须 | 内容要求 |
|------|------|------|
| `psnr_ranking` | 所有挑战 | **全量排名**，含 PSNR<30 的也要记录（test_candidate.py 自动维护） |
| `noise_6step_check` | 含 noise 的挑战 | **6 项数值完整记录**，不能只写结论 |
| `iterations[].hypothesis.thinking` | 所有 | 每个假设必须注明信号来源和推理过程 |
| `iterations[].test_result.psnr_rgb` | PSNR 模式 | 每次测试的精确 PSNR 值（test_candidate.py 自动计算） |
| `iterations[].test_result.matched_signals` | SIGNAL 模式 | 列出匹配/不匹配的具体信号名 |
| `final_decision.alternatives` | PSNR>=30 且 gap<10dB | 跨族备份候选 |
| `reflection_R1` | R1 反思触发时 | 信号重释过程 + 修正来源追踪 |

**🔴 exp35 审计教训**: reflection.json 不完整导致无法追溯盲识别决策。多个挑战 PSNR 缺失、6步检查数值缺失、假设缺乏信号来源。test_candidate.py + 强制字段规则确保后续实验可完整追溯每个决策链：**信号 → 假设 → PSNR → 判决**。
