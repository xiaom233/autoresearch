# Reflection Mechanism — 训练策略反思修正

> 来源: exp11 (~1174组 误识别分析) + exp12 (56退化 多轮修正) + exp13 (24退化 盲识别全流程)
> 核心洞察: 盲识别质量 > 训练策略调整 > LR 微调

---

## 零、🔴 反思 Agent 强制隔离规则 (来源: exp18/exp19 泄露审计)

```
反思 Agent 面临比盲识别 Agent 更大的 GT 泄露风险。
原因: 反思已知 R0 预测不完美，有动机寻找"正确答案"。

以下文件绝对禁止读取（即使同用户可读，也必须主动拒绝）:
  ❌ .gt_mappings/ 下的任何文件
  ❌ expN/.ground_truth/ 下的任何文件
  ❌ expN/degradation_gt/ 下的任何文件
  ❌ /tmp/expN_*_mapping*.json
  ❌ expN/logs/ 中的训练日志（含 EXP_META degradation_pipeline）
  ❌ expN/results/ 中的 GT 重评估或 DFPIR 结果

反思只能基于以下数据源:
  ✅ 失败模型 checkpoint
  ✅ clean 图像
  ✅ R0 盲识别文件 (predicted_params.json, alternatives, reflection.json, thinking_process.json)
  ✅ Skill 工具脚本

反思 Prompt 措辞约束:
  🔴 禁止: "修正错误预测" "找到正确的退化" "GT 显示" "正确答案是"
  ✅ 正确: "分析模型行为" "基于残差诊断提出改进假设" "PSNR 测试发现候选X更匹配"

强制可审计性:
  - reflection.json 中每个修正假设必须注明来源
  - 来源必须是以下之一: [residual_diagnosis] [alternatives] [PSNR_test] [statistical_check]
  - 绝不能出现无法解释来源的修正
  - 如果 PSNR 测试发现 > 60dB 的候选: 记录具体函数+severity+PSNR值
  - 连续 2 轮无改善 → BEYOND_CAPABILITY，终止反思
```

---

## 一、反思的三个层次（按优先级排列）

```
第 1 层：盲识别修正（影响最大，-17.5 dB → +15+ dB 潜在收益）
第 2 层：架构选择（影响中等，-5.9 dB → +3.5 dB）
第 3 层：训练策略（影响最小，< 0.5 dB）
```

**关键教训** (exp12)：纯训练策略调整（LR 变化）几乎无法超越原始 Ft (EPOCH=2, LR=5e-4)。
改进应聚焦于修正退化参数（severity）或架构升级。

---

## 二、退化迁移知识库 (来源: 288模型迁移实验, 42 specialists × cross-testing)

反思 Agent 应基于以下经验规律判断哪些候选值得测试，哪些不值得。

### Severity 偏差的代价

不同退化对 severity 错误的容忍度差异巨大：

| 退化类型 | sev±1 PSNR 损失 | 反思建议 |
|------|:--:|------|
| **JPEG** | -0.2 dB | 几乎无损失。sev 偏差不需要优先修正 |
| **blur (gaussian/lens)** | +0~-3 dB | 可接受。sev 偏差不是主要问题 |
| **noise (gaussian/speckle)** | -1~-3 dB | 可接受 |
| **brightness_gamma** | **-17 dB** | 致命。gamma 的 severity 必须精确，优先修正 |

### 子类型互换的代价

如果 R0 预测了 `blur_gaussian(3)`，但真实退化是 `blur_lens(3)`，模型迁移损失：

| 互换方向 | 损失 | 规律 |
|------|:--:|------|
| **gaussian ↔ lens** (sev≥3) | 1-4 dB | 几乎可互换。lens 模型在 gaussian 目标上有时甚至更好 |
| **gaussian/lens → glass** | 2-7 dB | 场景依赖。sev=3 时较接近，sev=1 时差距大 |
| **gaussian/lens → zoom** | 1-5 dB | zoom 专家反而表现更差！其他 blur 模型在 zoom 上更好 |
| **glass → gaussian/lens** (sev=1) | 5-8 dB | 低严重度时不可互换 |
| **poisson ↔ speckle** | 1-3 dB | 相对安全 |
| **gaussian_RGB ↔ YCrCb** | 1-5 dB | 可测试，但非优先 |
| **impulse → 任何其他 noise** | **21 dB** | 绝对不可互换。impulse 专家极度专一 |

### JPEG 反思 100% 可靠

基于 7-case 受控实验：当怀疑缺失 compression 时，PSNR 测试 JPEG(1-5) 的成功率是 100%（3/3 找到正确答案，2/3 像素级完美匹配）。compression 是确定性退化，PSNR 验证完全可靠。

### Blur 反思边际有效

PSNR 测试不同 blur 子类型可以找到改善方向，但增益通常在 0.1-2.0 dB 范围。blur 子类型之间的 PSNR 差距 <3dB，不可靠地区分。

### Noise 反思不可用 PSNR

所有 noise 类型的 PSNR 都受随机种子影响。不能用 PSNR 选择 noise 类型——必须使用 §B 统计检查。

### 反思 Agent 使用指南

当分析 R0 预测时，参考以上规律判断优先级：

1. 如果 R0 含 **brightness_gamma** → 优先检查 severity 是否精确（±1 代价 17dB）
2. 如果 R0 含 **noise_impulse** → 绝不替换为其他 noise 类型（代价 21dB）
3. 如果 R0 预测 **blur_gaussian** 且 alternatives 含 **blur_lens** → 安全 swap，直接换（代价 <4dB）
4. 如果怀疑缺失 **compression_jpeg** → 100% 值得 PSNR 测试
5. 如果 R0 含 **blur_zoom** → 考虑换成 gaussian/lens——其他模型在 zoom 上可能更好

---

## 三、训练驱动两阶段反思 (来源: exp21 双步反思失效审计)

### 问题

exp21 证明: 双步退化中 raw 残差无法解耦两个错误的信号。8/8 R1 修正全部不匹配 GT，平均退步 -1.31 dB。

### 方案: 先训练消除一个退化，再检测第二个

```
Stage 1: Train(model_A, R0 最自信的 1 步)
         model_A(target) → 消除退化A
         residual_A = target - model_A 的输出
        
Stage 2: 在 residual_A 上重跑检测
         → 第二步的信号现在被显式暴露
         → 选出候选 B → Train(model_AB, A+B)
         → PSNR 显著提升 → 修正成功
```

### 适用条件

- R0 步数 ≥ 2（或 R0 只检测到 1 步但实际有 2 步）
- R0 中有至少 1 步是高置信度（逐级剥离信号强）
- 额外成本: 1 次训练（约 1.5h），仅在双步案例触发

### 失败保护

- 最多 2 次额外训练（Stage 1 + Stage 2）
- 每阶段候选测试 ≤ 5 次
- 3 次尝试无改善 → BEYOND_CAPABILITY

---

## 二、反思触发条件 — 基于可观察信号的决策 ⚠️ 核心

> 来源: exp13 24 组盲识别验证。关键发现: **CI pass 不等于盲识别正确**。
> CI 测量视觉相似性，不测量函数正确性。函数 0/3 匹配时 CI 仍可达 8/10。

### 反思优先级：从可观察信号反推根因

不依赖 GT（GT 在盲识别场景下不可访问），仅用训练后可观察的 PSNR 信号判断是否需要修正：

#### 条件 A (必须修正): Spec < M_blind - 3dB

```
Spec 验证 PSNR 比盲基线 M_blind 低 3dB 以上
→ 盲识别几乎肯定有严重错误
→ 必须重新盲识别，修正 predicted_params
```

验证: exp13 中 3/3 满足此条件的退化函数匹配 ≤ 1/3
- 3004: -7.28 dB (函数 0/3, CI=6/10)
- 2007: -4.34 dB (函数 0/2, **CI=8/10** ← CI 漏报!)
- 3002: -4.05 dB (函数 1/3, CI=4/10)

#### 条件 B (建议修正): Spec < M_blind - 1dB AND CI < 7/10

```
Spec 轻微劣于盲基线，且盲识别置信度低
→ 盲识别可能有偏差
→ 检查 severity 估计是否合理
```

验证: exp13 中额外捕获 4 组需要关注的退化

#### 条件 C (训练异常): |Spec - Ft| > 3dB

```
Spec 和 Ft 差距超过 3dB
→ 训练策略不匹配
→ 如果 Ft << Spec: Ft 负迁移（LOAD_CKPT 冲突），检查是否需要 Direct
→ 如果 Spec << Ft: 盲识别参数可能不合理，检查 severity
```

验证: 3003 Spec=23.38 >> Ft=15.70 (+7.68 dB, Ft 崩溃)

#### 条件 D (低置信度): CI < 5/10

```
盲识别置信度极低
→ 即使 Spec ≈ M_blind，盲识别参数也可能不可靠
→ 建议重新盲识别
```

### 条件组合覆盖验证

| 条件组合 | exp13 捕获 | 漏报 |
|---------|:--:|:--:|
| CI < 7/10 单独 | 2/3 严重错误 | 2007 (CI=8/10 漏报!) |
| 条件 A+B+C+D | **3/3 严重错误** + 所有异常 | 0 |

### CI 的局限性

| 限制 | 证据 |
|------|------|
| CI 高 ≠ 函数正确 | 1004: CI=8/10, 函数 0/1 匹配 |
| CI 高 ≠ 下游 PSNR 好 | 2007: CI=8/10, Spec=-4.34 dB |
| CI 低 ≠ 函数错误 | 3010: CI=5/10, Spec=+0.19 (勉强可用) |

**结论**: CI 是盲识别迭代的收敛信号，不应作为"识别成功"的唯一判据。
反思决策应以**训练后 PSNR 信号**为主（Spec vs M_blind），CI 作为辅助参考。

---

## 三、盲识别诊断参考规则 (来源: exp11)

以下规则来自 ~1174 组交叉测试，可在盲识别阶段辅助判断，但不替代训练后 PSNR 信号。

### 容易混淆的退化对

| 退化对 | PSNR Gap | 已知特征 |
|--------|:--:|------|
| BS ↔ SF7 | 16-20 dB | 全局颜色变换相互混淆 |
| L4 ↔ SF7 | 16-20 dB | — |
| SF5 ↔ SF7 | 14-21 dB | — |
| L2 ↔ L6 | 15-16 dB | 相同全局类型 |
| L2 ↔ L4 | 13 dB | 相同全局，不同步数 |

### 图像特征规则

```
if edge > 0.10 AND chroma > 0.03: → 高置信误识别 (98% precision)
if entropy > 2.0 AND color_asymmetry > 0.08: → 类型误识别
if edge_frac < 0.13 AND entropy > 2.0: → 严重度偏差
```

基准: edge=0.044±0.020, chroma=0.013±0.010, entropy=0.92

---

## 四、第二层：架构选择修正

### 架构-退化匹配规则

| 退化特征 | 推荐架构 | 避免 |
|---------|---------|------|
| 严重 motion blur (sev ≥ 5) | OCAB + ws=16 + SwiGLU | MDTA |
| 随机噪声为主 | 任意 (Swin 即可) | — |
| contrast 全局退化 | Swin (只用 Swin) | MDTA, OCAB |
| contrast + 结构化局部 | Swin + FiLM-GCM | ColorPre |
| 纯 gamma/brightness_shift | Swin 基线 (无需额外) | — |
| 盲识别不确定 (低置信度) | DualBranch 或 CSN | ColorPre (泛化差) |

### 推荐附加组件

| 组件 | 参数 | 适用场景 | 效果 | 风险 |
|------|:--:|------|:--:|------|
| ColorPre | +0.8K | 默认首选，全局退化 | 最稳定 | 跨退化泛化差 |
| CSN | +1K | 通用，最轻量 | L5 +1.39 | 效果温和 |
| FiLM-GCM | +26K | contrast+结构化 | L5 +3.52 | L1 NaN (超出参数限制需减小 EMBED_DIM) |
| DualBranch | +3K | 需要鲁棒性时 | 最鲁棒 (10.1 gap) | 速度 -3% |

---

## 五、第三层：训练策略修正

### 修正决策树

```
Spec < Ft?
├── 全局退化 (contrast 类型)?
│   → 废弃 Ft, 改用 Direct (来源: exp9, L2 Ft -4.51 dB)
├── Δ > 3 dB → 严重问题
│   ├── 退化复杂 (≥2种) → 检查是否局部退化 Ft 可用
│   └── 退化简单 (1种) → 检查是否训练崩溃, 重跑
├── 1 < Δ ≤ 3 dB → 中度差距
│   └── 检查 severity 是否合理: R1 常高估 severity
├── 0.3 < Δ ≤ 1 dB → 轻微差距
│   └── Ft + LR=5e-4, 可能已接近当前架构最优
└── Δ ≤ 0.3 dB → 基本持平
    └── 不追额外训练 (已收敛)
```

### 关键教训 (来源: exp12)

1. **降 LR (5e-4→1e-4) 很少帮助 Ft** — 仅在特殊案例有效 (如 T20, blur sev=4 + 低LR 防止盲特征被覆盖)
2. **R1 最常见的错误是高估 severity** — 多组失败案例源于 severity 被错误上调
3. **Ft (LR=5e-4, EPOCH=2) 在局部退化上已接近当前架构的训练策略上限**
4. **loss 函数选择影响 < 0.25 dB** (来源: exp9, l1/mse/huber/l1+edge/l1+fft 之间差异极小)，**不应作为 R2 修正方向**

---

## 六、多轮修正机制

### R1 (首轮修正)
- 如果 Spec << M_blind: 检查盲识别 → 修正 predicted_params (最高优先级)
- 如果 Spec 接近 M_blind 但 << Ft: 检查 severity → 调整参数 ±1
- 如果全局退化用了 Ft: 切换到 Direct
- 如果架构选择不当: 参考架构速查表更换

### R2 (二次修正)
- R1 后 Δ > 0.5 dB → 继续调整
- 方向：调整架构附件 (添加 CSN/ColorPre)、尝试不同的 Curric 方向
- **不能增加 EPOCH** (必须 EPOCH=2)
- **不建议尝试不同 loss 函数** (差异 < 0.25 dB, 已验证无效)

### R3 (三次修正)
- R2 后仍有明显差距 → 尝试 Curric、Cascade 等更大改动
- 最多 3 轮，之后标记"已收敛"

### 收敛判定
- Δ < 0.3 dB → 已收敛，不值得继续
- R1/R2 倒退 → 修正方向错误，回退到前一版本

---

## 七、训练公平性规则 ⚠️

1. **EPOCH_BUDGET 必须统一 = 2** — 与基线 Spec/Ft 一致。禁止 EPOCH=3
2. **模块参数 ≤ 基线 × 1.05 (≤ 477K)** — 架构对比必须在同参数下 (来源: exp10 原则 6)
3. **VAL_PARAMS_PATH 必须锁定为 GT** — R1 只允许修改 PARAMS (exp12 的致命错误)
4. **PSNR 提升必须 > 0.5 dB 才值得继续下一轮**
5. **所有对比在同一 GT 上评估** — 否则毫无意义

---

## 八、全局退化处理规则 (来源: exp9 + exp10)

| 规则 | 内容 |
|------|------|
| 识别 | 管线含 contrast_weaken/strengthen、brightness_*、saturation_*、gamma_* |
| contrast 型 | Direct (不用 Ft)，Swin 架构 |
| brightness 型 | Direct 保守，Ft 在 L1 上安全 (+0.41) |
| 架构 | Swin (不用 MDTA/OCAB) |
| 附加 | 可选 ColorPre (+0.8K) 或 CSN (+1K) |
| FiLM-GCM 预警 | brightness_HSV + blur 组合：gaussian_blur 平滑 HSV 极端值 → 统计量偏移大 → GCM 学到极大 scale/shift → 训练后期 NaN。有额外 noise (L4) 时特征多样性增加 = 隐式正则化 → 不 NaN |
| 验证 | exp9 L2(contrast+noise): Ft 崩溃 -4.51 dB; exp10: L1(brightness+blur): Ft 安全 +0.41 |
