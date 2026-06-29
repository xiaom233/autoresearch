# exp38：exp9/exp10 审计修正与策略解耦验证

> 来源：exp9/exp10 审计报告（2024-06-29）
> 目标：验证并修正 exp9/exp10 中被策略-架构耦合污染的结论，修正训练策略文档

---

## 一、审计发现的问题

### 🔴 问题 1：L2 MDTA/OCAB crash 的策略-架构耦合

**现状**：finetune_strategy.md 声称 "contrast 全局退化 → Swin（避免 MDTA/OCAB）"，引用 L2_f_mdta=-9.78 dB 和 L2_f_ocab=-10.38 dB 作为证据。

**问题**：L2_f_mdta 和 L2_f_ocab 的 `_f_` 后缀表示 RandomCurric 策略（`AR_CURRICULUM_CONFIG=0:random,...`）。L2 上 RandomCurric 本身已崩溃 -4.51 dB（L2f=24.21 vs L2d=28.72）。MDTA/OCAB 的额外劣化无法确定是：
- A) 架构本身不适合 contrast（与策略无关）
- B) RandomCurric 崩溃 + 架构不匹配叠加
- C) RandomCurric 崩溃主导，架构影响很小

**缺失的关键实验**：L2 Direct+MDTA 和 L2 Direct+OCAB。

### 🔴 问题 2："Motion sev≥5 Fwd 翻转"过度泛化

**现状**：finetune_strategy.md 声称 "motion sev ≥ 5 时 Fwd 翻转"。

**实际数据**（S5: motion sev=5）：
```
基础 Curric (Swin): S5cf(Fwd)=21.80 > S5cr(Rev)=19.15  → Fwd 胜出 +2.65 dB
TA 架构变体:        S5_fwd_TA=16.54 < S5_rev_TA=22.43   → Rev 胜出 +5.89 dB
```

**问题**：翻转仅在特定 TA 架构下出现，基础 Swin Curric 仍 Fwd > Rev。文档将其表述为通用规律是错误的。

### 🔴 问题 3：T2 基于已禁用退化函数

T2 管线含 `noise_spatially_correlated` 和 `compression_jpeg_2000`（SKILL.md 明确禁用）。~75 组 T2 实验的结论不可引用到策略推导中。

### 🟡 问题 4："Ft" 术语全局误用

exp9/exp10 中所有标记 "Ft" 的实验实际是 Random Curriculum（`AR_CURRICULUM_CONFIG=0:random,...`），不涉及 `AR_LOAD_CKPT`。原始文档（exp9/summarize/）和 result.tsv 未修正。

### 🟡 问题 5：exp10 L2 架构对比缺少 MDTA/OCAB

exp10 的 L2 架构实验（FiLM、ColorPre、CSN、PCP 等）全部在 Direct 训练下。但 MDTA/OCAB 未在 Direct 下测试。finetune_strategy.md 关于 MDTA/OCAB 的结论依赖的是 exp9 L2_f_mdta/L2_f_ocab（RandomCurric 策略）。

### 🔴 问题 6：FiLM-GCM 在非 contrast 全局退化上零验证

**现状**：finetune_strategy.md 称 "contrast 是唯一同时满足策略受限和组件敏感的全局退化"，"其他全局退化要么策略灵活要么组件无效"。但 FiLM-GCM 在 brightness_shift、saturation、gamma、stretch 四种非 contrast 全局退化上从未测试。

**exp10 已有数据 vs 缺口**：

| 全局退化类型 | 退化管线 | Swin 基线 | ColorPre | FiLM-GCM | 
|------|------|:--:|:--:|:--:|
| contrast_scale + noise | L2 | 28.64 | +0.81 | +0.79 |
| brightness_shift + noise | BS | 29.14 | +0.03 | **❌ 未测** |
| saturation + noise | SF7 | 29.43 | -0.34 | **❌ 未测** |
| gamma + blur + noise | SF5 | 22.85 | +0.03 | **❌ 未测** |
| gamma_RGB + noise | G1 | 29.33 | **❌ 未测** | **❌ 未测** |
| stretch + noise | G2 | 28.33 | **❌ 未测** | **❌ 未测** |

**风险**：如果 FiLM-GCM 在非 contrast 全局退化上也有效（即使 +0.3~0.5 dB），"contrast 专用组件"的结论就需要修正。如果 FiLM 像 L1 brightness_HSV 一样 NaN，则安全边界需要明确。

---

## 二、实验设计

### 退化管线定义

**来自 exp9**（策略解耦）：

| 管线 | 内容 | 验证目标 |
|------|------|------|
| `L2_dual.json` | noise_gaussian_RGB(3) + contrast_weaken_scale(3) | 策略-架构解耦 |
| `D1_dual.json` | blur_gaussian(3) + noise_gaussian_RGB(3) | RandomCurric 安全基线 |
| `D3_dual.json` | compression_jpeg(3) + blur_lens(4) | True Ft vs RandomCurric |
| `S5_triple.json` | blur_motion(5) + noise_gaussian_RGB(1) + compression_jpeg(1) | Motion Fwd/Rev |

**来自 exp10**（组件跨退化泛化）：

| 管线 | 内容 | 验证目标 |
|------|------|------|
| `BS_dual.json` | noise_gaussian_RGB(3) + brightness_darken_shfit_RGB(3) | brightness_shift — FiLM 安全性 |
| `SF7_dual.json` | noise_gaussian_RGB(3) + saturate_weaken_HSV(3) | saturation — FiLM 是否有害 |
| `SF5_triple.json` | blur_gaussian(3) + noise_gaussian_RGB(3) + brightness_darken_gamma_RGB(3) | gamma+blur+noise — FiLM 有效性 |
| `G1_dual.json` | noise_gaussian_RGB(3) + brightness_darken_gamma_RGB(3) | gamma_RGB — 纯 gamma+noise |
| `G2_dual.json` | noise_gaussian_RGB(3) + contrast_weaken_stretch(3) | stretch — 非 contrast 全局 |

### 统一训练配置

```
EPOCH_BUDGET=2  LR=5e-4  BATCH_SIZE=16  EMBED_DIM=64
AMP_DTYPE=bfloat16  LOSS_FN=l1  LR_SCHEDULE=constant
VAL_PARAMS_PATH=同 PARAMS_PATH  (防泄露，仅对比 Spec 模型)
```

---

### Phase A：L2 策略-架构解耦（6 组）

**核心问题**：MDTA/OCAB 在 contrast 退化上的表现差，是架构问题还是 RandomCurric 策略问题？

| ID | 策略 | 注意力 | 环境变量 | 预期 PSNR | 目的 |
|------|------|------|------|:--:|------|
| A1 | Direct | Swin | — | ~28.70 | 基线复现（exp9 L2d=28.72） |
| A2 | Direct | MDTA | `AR_ATTENTION_TYPE=mdta` | ? | **关键：首次测试 Direct+MDTA** |
| A3 | Direct | OCAB | `AR_ATTENTION_TYPE=ocab` | ? | **关键：首次测试 Direct+OCAB** |
| A4 | RandomCurric | Swin | `AR_CURRICULUM_CONFIG=0:random,7547:...` | ~24.21 | 复现崩溃（exp9 L2f=24.21） |
| A5 | RandomCurric | MDTA | A4 + MDTA | ~18.92 | 复现崩溃（exp9 L2_f_mdta=18.92） |
| A6 | RandomCurric | OCAB | A4 + OCAB | ~18.32 | 复现崩溃（exp9 L2_f_ocab=18.32） |

**判定逻辑**：
- 若 A2/A3 ≈ A1（~28.70）→ MDTA/OCAB 对 contrast 无害，**问题在 RandomCurric 策略**
- 若 A2/A3 ≈ A5/A6（~18-19）→ MDTA/OCAB 确实不适合 contrast，**架构结论成立**
- 若 A2/A3 介于 A1 与 A5/A6 之间（如 ~23-25）→ 部分耦合，需量化拆分

---

### Phase B：Motion Fwd/Rev 通用性验证（7 组）

**核心问题**：S5 motion sev=5 下，Fwd/Rev Curric 的真正规律是什么？

| ID | 策略 | 注意力 | 环境变量 | 预期 |
|------|------|------|------|:--:|
| B1 | Direct | Swin | — | 基线 (exp9 S5d=21.80) |
| B2 | RandomCurric | Swin | `AR_CURRICULUM_CONFIG=0:random,...` | exp9 S5f=20.51 |
| B3 | Curric Fwd | Swin | `AR_CURRICULUM_CONFIG=0:S5_fwd1,5000:S5_triple` | exp9 S5cf=21.80 |
| B4 | Curric Rev | Swin | `AR_CURRICULUM_CONFIG=0:S5_rev1,5000:S5_triple` | exp9 S5cr=19.15 |
| B5 | Direct | OCAB+ws16 | `AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16` | ? |
| B6 | Curric Fwd | OCAB+ws16 | B3 + OCAB+ws16 | exp9 S5_f_ocab_ws16=22.23 |
| B7 | Curric Rev | OCAB+ws16 | B4 + OCAB+ws16 | ? |

**判定逻辑**：
- 若 B3 > B4（Swin Fwd > Rev）→ **确认"Fwd 翻转"不适用于 Swin**，是架构相关现象
- 若 B6 > B3（OCAB Fwd > Swin Fwd）→ OCAB 在 motion 上有架构优势
- 若 B7 > B6（OCAB Rev > OCAB Fwd）→ OCAB 的 Rev 优势才是翻转的根源

---

### Phase C：RandomCurric 安全性验证（4 组）

**核心问题**：RandomCurric 在纯局部退化上是否安全？它和 Direct 的准确差距？

| ID | 管线 | 策略 | 预期 |
|------|------|------|:--:|
| C1 | D1 (blur+noise) | Direct | exp9 D1d=22.92 |
| C2 | D1 (blur+noise) | RandomCurric | exp9 D1f=23.04 |
| C3 | D3 (JPEG+blur) | Direct | exp9 D3d=20.99 |
| C4 | D3 (JPEG+blur) | RandomCurric | exp9 D3f=25.84 |

全部 Swin 架构，EPOCH=2，LR=5e-4。

**判定逻辑**：
- C2 ≥ C1 → RandomCurric 在纯局部上安全/有益
- C4 >> C3 → D3 是梯度干扰典型案例（RandomCurric 大幅有效）
- 两组都 ≥ Direct → RandomCurric 通用安全，仅 contrast 型全局退化有风险

---

### Phase D：True Ft vs RandomCurric 对比（3 组）

**核心问题**：RandomCurric 的效果来自"减少梯度干扰"还是"预训练特征迁移"？

| ID | 策略 | 环境变量 | 预期 |
|------|------|------|:--:|
| D1 | True Ft | `AR_LOAD_CKPT=resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt` | exp37: ≈Direct |
| D2 | RandomCurric | `AR_CURRICULUM_CONFIG=0:random,7547:D3_dual` | exp9 D3f=25.84 |
| D3 | Direct | — | exp9 D3d=20.99 |

全部 D3 管线，Swin 架构，EPOCH=2，LR=5e-4。

**判定逻辑**：
- D1 ≈ D3 → True Ft 在有限预算下无优势（exp37 结论复现）
- D2 >> D1 → RandomCurric 的收益来自梯度干扰减少，不是预训练迁移
- D2 >> D3 且 D1 ≈ D3 → **确认** RandomCurric ≠ True Ft，机制完全不同

---

### Phase E：contrast 架构组件 Direct 复现（2 组）

**核心问题**：FiLM-GCM 和 ColorPre 在 Direct+L2 上是否仍有效？（已知 exp10 数据，复现验证）

| ID | 架构 | 环境变量 | 预期 |
|------|------|------|:--:|
| E1 | Swin+FiLM-GCM | `AR_USE_GCM=1 AR_LEARNING_RATE=5e-4` | exp10 G4_L2_FiLM=29.49 |
| E2 | Swin+ColorPre | `AR_USE_COLOR_PRE=1` | exp10 P3_L2_ColorPre=29.45 |

全部 Direct 训练，LR=5e-4，L2 管线。

---

### Phase F：FiLM-GCM/ColorPre 跨全局退化泛化（10 组）

**核心问题**：FiLM-GCM 在非 contrast 全局退化（brightness/saturation/gamma/stretch）上是否也有效？当前 "contrast 专用" 的结论是否可靠？

**exp10 已知数据**：
- BS: ColorPre +0.03（trivial），FiLM 未测
- SF7: ColorPre -0.34（有害），FiLM 未测
- SF5: ColorPre +0.03（trivial），FiLM 未测
- G1: 两者均未测
- G2: 两者均未测

| ID | 退化 | 架构 | 环境变量 | exp10 基线 | 目的 |
|------|------|------|------|:--:|------|
| F1 | BS | Swin (Direct) | — | 29.14 | 基线对齐 |
| F2 | BS | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | ❌ | brightness_shift — FiLM 是否安全？（L1 NaN 教训） |
| F3 | SF7 | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | ❌ | saturation — FiLM 是否像 ColorPre 一样有害？ |
| F4 | SF5 | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | ❌ | gamma triple — gamma 场景 FiLM 有效性 |
| F5 | G1 | Swin (Direct) | — | 29.33 | gamma_RGB 基线对齐 |
| F6 | G1 | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | ❌ | gamma_RGB+noise — 纯 gamma 非 contrast |
| F7 | G1 | ColorPre | `AR_USE_COLOR_PRE=1` | ❌ | gamma_RGB — ColorPre 在 gamma 上 |
| F8 | G2 | Swin (Direct) | — | 28.33 | stretch 基线对齐 |
| F9 | G2 | FiLM-GCM | `AR_USE_GCM=1 AR_LR=5e-4` | ❌ | stretch — FiLM 在非 contrast stretch 上 |
| F10 | G2 | ColorPre | `AR_USE_COLOR_PRE=1` | ❌ | stretch — ColorPre 在 stretch 上 |

全部 Direct 训练，EPOCH=2，LR=5e-4。

**注意**：BS 的 ColorPre（F2 改）、SF7 的 ColorPre、SF5 的 ColorPre 已由 exp10 测试，此处不重复。（BS: +0.03, SF7: -0.34, SF5: +0.03）

**判定逻辑**：
- 若 F2/F3/F4/F6/F9 全部 ≈ 对应 Swin 基线（±0.3 dB）→ **确认** FiLM-GCM 仅对 contrast 有效，"contrast 专用"结论成立
- 若某非 contrast 退化上 FiLM > +0.5 dB → **需修正** finetune_strategy.md，扩大 FiLM 适用范围
- 若 F2（brightness_shift）NaN → FiLM 风险不限于 brightness_HSV，全局 shift 型也有风险
- 若 F3（saturation）< -1 dB → FiLM 像 ColorPre 一样在 saturation 上有害

---

## 三、实验总结

| Phase | 实验数 | 核心问题 | 预计墙钟（8 GPU） |
|:--:|:--:|------|:--:|
| A | 6 | L2 策略-架构解耦 | ~10 min |
| B | 7 | Motion Fwd/Rev 通用性 | ~10 min |
| C | 4 | RandomCurric 安全性 | ~5 min（与 A/B 并行） |
| D | 3 | True Ft vs RandomCurric | ~5 min |
| E | 2 | contrast 组件复现 | ~5 min |
| **F** | **10** | **FiLM/ColorPre 跨全局退化泛化** | **~10 min** |
| **合计** | **32** | | **~25-30 min** |

每实验 ~5 min GPU 时间（EPOCH=2），8 GPU 并行排队消化。

---

## 四、预期产出

### 策略文档修正

1. **L2 MDTA/OCAB 结论修正**：根据 Phase A 结果，明确标注是"架构问题"还是"策略耦合"，或给出分离后的量化估计。

2. **Motion Fwd/Rev 规则修正**：根据 Phase B 结果，将"motion sev≥5 Fwd 翻转"修正为更精确的表述，区分 Swin vs OCAB 的不同行为。

3. **RandomCurric 安全边界**：根据 Phase C 结果，明确 RandomCurric 在纯局部退化上安全，仅在 contrast 型全局退化上有崩溃风险。

4. **True Ft 与 RandomCurric 分离**：根据 Phase D 结果，确认两者机制不同（梯度干扰减少 vs 预训练迁移）。

5. **T2 数据标记**：标注 T2 实验基于已禁用函数，不可引用。

6. **FiLM-GCM 适用范围**：根据 Phase F 结果，明确 FiLM-GCM 是否真的是 "contrast 专用"，还是在其他全局退化上也有效/有害。

### finetune_strategy.md 待修正条目

| 条目 | 当前表述 | 待修正为 |
|------|------|------|
| 架构选择表 §五 | "contrast → Swin（避免 MDTA/OCAB）" | 根据 Phase A 结果修正 |
| 机制 2 §一 | "motion sev≥5 Fwd 翻转" | 根据 Phase B 结果修正 |
| 全局退化 §二 | "RandomCurric 崩溃 -4.51" | ✅ 正确，补充"True Ft 未测试" |
| 术语 §八 | "exp9/10 Ft → RandomCurric" | ✅ 已修正，保持 |
| T2 引用 | 多处引用 T2 作为 motion+SC 案例 | 标注"已禁用函数，不可引用" |
| 组件适用范围 §五 | "contrast 是唯一组件敏感的全局退化" | 根据 Phase F 结果修正 |
| FiLM-GCM 限制 §七 | "FiLM-GCM: brightness_HSV 上 NaN" | 根据 Phase F 结果补充其他退化上的安全性 |

---

## 五、执行步骤

```
Step 1: 复制退化管线
  # 来自 exp9
  cp exp9/degradation/{L2,D1,D3}_dual.json exp38/degradation/
  cp exp9/degradation/S5_triple.json exp38/degradation/
  cp exp9/degradation/{L2,S5}_fwd1.json exp38/degradation/
  cp exp9/degradation/{L2,S5}_rev1.json exp38/degradation/
  # 来自 exp10 (Phase F)
  cp exp10/degradation/SF_BrightShift.json exp38/degradation/BS_dual.json
  cp exp10/degradation/SF7_saturate_dual.json exp38/degradation/SF7_dual.json
  cp exp10/degradation/SF5_gamma_triple.json exp38/degradation/SF5_triple.json
  cp exp10/degradation/G1_noise_gamma.json exp38/degradation/G1_dual.json
  cp exp10/degradation/G2_noise_stretch.json exp38/degradation/G2_dual.json

Step 2: 生成任务队列
  python3 exp38/scripts/generate_experiments.py

Step 3: 8 GPU 并行启动
  bash scripts/exp_launcher.sh cleanup
  bash scripts/exp_launcher.sh start exp38 exp38/scripts/phase5_tasks.txt 0,1,2,3,4,5,6,7

Step 4: 等待完成 → 汇总 PSNR
  python3 exp38/scripts/collect_results.py

Step 5: 更新 finetune_strategy.md
  根据实验结果修正所有受影响的结论

Step 6: 更新 exp9/summarize/ 术语
  将 "Ft/Fine-tune" 全局替换为 "RandomCurric"
```
