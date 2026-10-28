# Controller 层对比：LLM controller vs rule-based controller

> 目的：说明 E1 中 rule-based controller 的构造方式，以及它和 AutoIR（LLM/Skill controller）的对比设计、blind_0003 的具体表现、整体 24 场景结果和审计限制。
>
> 相关档案：`rebuttal_plan/case_blind_0003/report.md`（blind_0003 逐轮规则/LLM 归因）。
>
> 核心结论：**rule-based controller 不是另一个修复网络，而是把 AutoIR 的 controller 层替换成一个确定性的 signal→pipeline 状态机。** 它复用同一套信号工具、同一批 support、同一训练预算和同一个固定 LQ 测试集；区别只在“如何构造管线、如何反思、如何选 checkpoint”。它在 blind_0003 上赢 AutoIR，但在 24 场景整体上低于 AutoIR（−0.70/−0.82 dB，8/24 胜），因此应作为控制变量 ablation，而不是“规则系统更好”的证据。

---

## 1. 它到底是什么

| 项目 | 内容 |
|---|---|
| 代码 | `rebuttal_runs/scripts/rule_controller.py`（612 行） |
| 冻结规则 | `rebuttal_runs/E1/rules_frozen.json`，版本 `exp41-rebuttal-rule-v1` |
| 冻结时间 | `2026-10-24`（E1 执行于 2026-10-25，先冻结后跑） |
| controller 类型 | `deterministic signal-to-pipeline state machine; no LLM` |
| 输入 | 每个场景 5 对 support（`clean_i.png` / `degraded_i.png`） |
| 信号工具 | `run_full_analysis.py`（RFA） |
| 噪声验证 | `verify_signals.py` |
| 训练 | 与 AutoIR 相同：Direct、2 epoch、LR 5e-4、batch 16、L1、seed 42、DIV2K train shards |
| 测试 | 与 AutoIR 相同：固定 LQ，DIV2K 100 + LSDIR 250；17/24 场景 LQ hash 完全一致，7/24 非确定性场景使用 `E1E2_fixed_lq` 的同一份固定 LQ |
| 是否使用 GT | 否；GT 只在最终测试评分时使用 |
| 是否使用 LLM | 否；没有自然语言 agent 在决策回路中 |

**关键定位：** rule controller 不是“没有 reflection 的单轮预测器”。它有 R0→R1→R2 三轮状态机，只是“反思”由 support 仿真分数、support PSNR 和冻结候选表驱动，而不是由 LLM 的自然语言推理驱动。

---

## 2. 构造总览

```text
5 对 support
  ↓
① 每对图跑 run_full_analysis.py（RFA），缓存信号
  ↓
② dc_from_one：信号 → 每对图的退化步骤
  ↓
③ aggregate：5 对图 3/5 多数投票 + severity 中位数 → R0 管线
  ↓
④ rs_for：按冻结规则选架构（OCAB / GCM / Swin+SG）
  ↓
⑤ candidates_for：枚举 R1/R2 候选（near-miss、删步、severity±、子类型替换）
  ↓
⑥ score_pipeline：确定性用 PSNR，含 noise 用 verify_signals
  ↓
⑦ plan/decide：R0→R1→R2 状态机 + support PSNR 选 checkpoint → 固定 LQ 测试
```

---

## 3. 逐层构造

### 3.1 每对图信号提取

对 5 对 support 分别运行：

```bash
python run_full_analysis.py --target degraded_i.png --clean clean_i.png
```

结果缓存到：

```text
rebuttal_runs/E1/cache/{scenario}_{i}.json
```

这一步和 AutoIR/Skill 用的是同一个脚本。因此 rule controller 并不是“另一套信号处理”，而是**同一个信号层上的确定性决策层**。E1 的对比是 controller 层的 ablation，不是信号工具层的 ablation。

### 3.2 单对图信号 → 退化步骤（`dc_from_one`，`rule_controller.py:81`）

规则全部硬编码/冻结：

| 类别 | 触发阈值 | 子类型 | severity |
|---|---|---|---|
| compression | `block_boundary > 1.05` | `compression_jpeg` | 按 `[1.05, 1.15, 1.3, 1.5]` 映射 |
| noise | `IMPULSE` verdict 或 `sigma >= 2.0` | 从 verdict 字符串映射：YCrCb / RGB / speckle / poisson / impulse | 优先 `estimated_severity`，否则按 sigma 分档 |
| blur | `gm_ratio < 0.85` | 默认 `blur_gaussian`；`FFT_anisotropy > 1.7` → `blur_motion` | 按 `gm_ratio` 的 `[0.1, 0.2, 0.35, 0.5]` 映射 |
| contrast | `variance_ratio` 不在 `[0.85, 1.15]` 且脚本判断 contrast | `contrast_weaken_scale` / `contrast_strengthen_scale` | 按方差偏离度分档 |
| brightness | `abs(mean_shift) > 15` | `brightness_*_shfit_RGB`；verdict 含 GAMMA 时用 gamma 版本 | 按均值偏移分档 |
| saturation | `saturation_chroma_ratio` 不在 `[0.75, 1.4]` | `saturate_weaken_HSV` / `saturate_strengthen_HSV` | 按偏离度分档 |

注意：

- compression 的阈值是 **1.05**，比 SKILL 文档里的 1.1 更敏感；因为后面还有 PSNR 预验证，允许先多测再排除。
- 代码只用 `block_boundary` 触发 compression；`unique_G < 200` 在规则说明中仅作辅助，不单独触发。
- 如果 verdict 是 IMPULSE，跳过 global 类别（impulse 会污染全局统计）。
- 每个单对图最多保留 3 步，并按物理顺序排序：`noise/brightness → blur → contrast/saturation → compression`（`rule_controller.py:54-55`）。

### 3.3 5 对图聚合（`aggregate`，`rule_controller.py:185`）

- 每个类别必须至少 **3/5** 票才进入管线；
- 函数名取多数票；
- severity 取同意该函数的票的**中位数**；
- 最多 3 步；
- 按物理顺序排序。

如果没有任何类别达到 3/5，走**预注册 fallback**：

```text
median sigma >= 1.0  → noise_gaussian_RGB:1
elif median gm < 1.0 → blur_gaussian:1
elif median bb > 1.0 → compression_jpeg:1
else                 → noise_gaussian_RGB:1
```

这不是 LLM 临时决定的，是 `rules_frozen.json`/代码里写死的保守兜底。

### 3.4 架构规则（`rs_for`，`rule_controller.py:202`）

| 条件 | 架构 |
|---|---|
| `blur_motion` 且 `FFT_anisotropy > 1.7` | OCAB，embed 30，window 16，SimpleGate |
| 有 contrast + 有 blur/compression + `variance_ratio` 不在 `[0.85,1.15]` | Swin + FiLM-GCM + SimpleGate |
| 其他 | Swin + SimpleGate（默认） |

这跟 `finetune_strategy.md` 的规则一致；rule controller 只是把它写成了确定性函数。训练配置也固定为：

```text
Direct / 2 epoch / LR 5e-4 / batch 16 / L1 / seed 42
DIV2K train shards
```

### 3.5 候选枚举（`candidates_for`，`rule_controller.py:335`）

对 R0 中票数最低的类别，枚举以下候选（每轮最多 15 个）：

1. **near-miss candidates**：信号接近但未达硬阈值的类别（比如 `block_boundary > 1.0`、`gm_ratio < 1.0`、`sigma >= 1.0`），追加为一步（`near_miss_candidates`，`:289`）；
2. **删步**：去掉票数最低的类别；
3. **severity 扫描**：`±1, ±2`；
4. **子类型替换**：
   - blur：`gaussian / glass / lens / motion`
   - noise：`gaussian_RGB / gaussian_YCrCb / speckle / poisson / impulse`
   - contrast：`weaken/strengthen × scale/stretch`
   - brightness：`darken/brighten × shift/gamma`
   - saturation：`weaken/strengthen × HSV/YCrCb`

**blind_0003 的 glass 就在 blur 子类型替换列表里。** 所以 rule controller 不是从信号“发现”了 glass，而是它的固定枚举里本来就有 glass；这是解释它为什么在该场景赢 AutoIR 的关键。

### 3.6 候选评分（`score_pipeline`，`rule_controller.py:255`）

- 管线**不含 noise**：用 `x_distortion.apply_pipeline` 模拟 → 计算与 target 的 PSNR；`score = mean PSNR`。
- 管线**含 noise**：模拟后保存图像 → 调 `verify_signals.py` → 把 `MATCH/PARTIAL/WEAK/MISMATCH` 映射为 `1.0/0.66/0.33/0.0` → `score = mean`；同时记录 PSNR，但不用它做噪声类型判断。

判定阈值（`rules_frozen.json`）：

```text
确定性管线 good: score >= 40 dB
含噪声管线 good: score >= 0.99（即几乎 5/5 都 MATCH）
R1/R2 最小提升: 确定性 +0.5 dB；噪声 +0.15 signal
每轮最多 15 个候选
最多 3 轮
```

### 3.7 R0→R1→R2 状态机（`plan` / `decide`）

`plan` 先为每个场景写出 R0 管线和预计算的 R1/R2 候选，再训练 R0。`decide` 根据训练结果决定是否训练下一轮：

```text
train R0
if low_confidence_R0 or NaN or R0_support_PSNR < 25:
    train R1
if R1_support_PSNR - R0_support_PSNR > 0.5
   and no NaN
   and R2 candidate exists:
    train R2
```

其中：

- `low_confidence_R0 = not (score_R0 >= 40 dB / 0.99)`；
- R2 候选必须是“除 R1 之外的、比 R0 高 0.5 dB 以上的第二名”；
- 如果 R0 出现 NaN 且架构含 GCM，自动回退到 Swin+SG；
- `low_confidence_R1` 在代码里默认 `True`，所以 R2 的实际门槛主要是“R1 提升 > 0.5 dB + R2 候选存在 + 无 NaN”。这是一个实现细节，论文里如果要严格描述，应说明这一点。

### 3.8 最终模型选择（`merge_e1_selection.py:38-45`）

不是看 test，而是看 support：

1. 每个 round 取 support PSNR 最高的 checkpoint；
2. 跨 round 取 `top - 0.05 dB` 以内的候选；
3. 其中优先 `_fixed`（修正映射）轮次，再优先更大的 round 号；
4. 选中后只在固定 LQ 测试集上评估一次。

E1 的实际选择分布：

```text
24 个场景：R0 5 个，R0_fixed 1 个，R1 18 个，R2 0 个
R0/R1/R2 训练任务数：24 / 19 / 9
```

也就是说，rule controller 的 R2 训练了 9 个场景，但**最终没有一个 R2 被 support PSNR 选中**；它主要停在 R0 或 R1。

---

## 4. 和 LLM controller 的构造对比

| 维度 | rule-based controller | LLM controller（AutoIR / Skill Agent） |
|---|---|---|
| 信号提取 | `run_full_analysis.py` | 同一个脚本 |
| 噪声验证 | `verify_signals.py` | 同一个脚本 |
| 信号解释 | 固定阈值映射 | LLM 阅读信号并解释 |
| 候选生成 | 固定枚举（near-miss/删步/severity±/子类型） | LLM 可提出枚举外候选（blind_0003 的 glass） |
| 反思 | support 分数 + PSNR 状态机 | 多轮 Skill reflection，带文本证据 |
| 架构规则 | 冻结的 `rs_for` | 同样的 `finetune_strategy` 规则 |
| 训练/测试 | 同一预算、同一 support、同一固定 LQ | 同一预算、同一 support、同一固定 LQ |
| 失败模式 | 阈值/候选表/映射 bug；固定枚举漏掉正确候选 | 过度保守规则压制正确候选；从 GT 反推结构（R4 双 blur） |

所以 E1 的对比是**控制变量 ablation**：两边共享信号层、训练、测试，只替换 controller。不是“小模型 vs 大模型”，也不是“规则系统 vs 端到端学习”。

---

## 5. blind_0003 具体怎么走

`rebuttal_runs/E1/decisions/blind_0003.json`：

```text
R0 聚合：blur_gaussian:2，5/5 票
R0 仿真分数：32.339 dB（< 40 → low_confidence_R0 = true）
候选打分：
  blur_glass:2        33.715  ← 最高
  blur_gaussian:2     32.339  ← R0 基线（不在候选列表里）
  blur_lens:2         32.079
  ...
R1 = blur_glass:2（33.715 - 32.339 = 1.376 > 0.5）
R1 训练 support PSNR = 28.40
R2 候选 = None（第二名 lens:2 只有 32.079，未超过 32.339+0.5）
trigger_R2 = false
最终选择 R1 glass:2
固定 LQ：DIV2K 27.937 / LSDIR 24.192
```

对比 AutoIR 在 blind_0003 的盲选：R1 `gaussian:2`，固定 LQ 25.526 / 22.332。

所以 rule controller 在该场景 +2.41 / +1.86 dB，**原因是它的固定子类型列表里有 glass，而且它直接用 PSNR 排名采纳；AutoIR 的 LLM 也发现了 glass（36.65 / CI 10/10），但被“blur 子类型 PSNR 仅 2% 可信”的规则压住，直到 R3 用 GT PSNR=19.33 才切换。**

这也是本案例最重要的结论：**rule controller 的胜利是“固定候选表 + 标量选择”的胜利，不是信号理解的胜利；LLM 的贡献是发现了候选表外的 glass，但它的选择规则没有允许采纳。**

### 5b. blind_0020 反例：规则的高置信伪信号无法被自身证伪

`rebuttal_runs/E1/decisions/blind_0020.json` + `rebuttal_plan/case_blind_0020/report.md`：

```text
规则聚合：blur_gaussian 5/5、saturation_weaken 5/5、contrast_weaken_scale 4/5
R0 = gaussian:5 + scale:5 + sat:5，仿真 19.83；Swin+GCM → step 11319 起 loss NaN
R1 fallback Swin+SG，候选只扰动最低票类别 contrast → stretch:5，仿真 25.69
R1 support 15.736；fixed-LQ DIV2K 15.991 / LSDIR 15.227
R2 = 预计算 second-best（scale:3），support 14.246，test 14.118 → 不选
```

规则的两处结构性限制：

1. `dc_from_one` 的 blur 分支只按 anisotropy 输出 `blur_motion`/`blur_gaussian`，**没有 glass 分支**；
   即使分析器在 pair4 报 `GLASS_BLUR`，规则仍输出 gaussian。
2. `candidates_for` 只扰动**票数最低**的类别（本案例 contrast 4/5）；blur 和 saturation 都是 5/5，
   因此 R1/R2 从不测试 glass、也不移除 saturation。R1/R2 是 plan 阶段一次性预计算的 best/second-best，
   不是根据训练结果生成的新假设。

LLM 的相反路径：R0 用反事实 PSNR 拒绝 noise/saturation/JPEG；R2 主动测试 `glass_score=0.246`
（刚好低于 0.25 阈值）的 `blur_glass:4`，仿真从 37.73 升到 **39.56 dB（CI 10/10，+1.83）**。
fixed-LQ：AutoIR R2 **18.084 / 17.409** vs 规则 R1 **15.991 / 15.227**。
同架构（Swin+SG）纯 DC 对比：AutoIR R1 16.975 vs 规则 R1 15.991（**+0.984 / +1.194**），
说明 DC 本身有贡献；最终 18.084 还叠加了 ColorPre（同架构控制进行中）。

这不是“LLM 普遍更好”的证据（E1-oneshot 总体 CI 跨 0），而是回答 Reviewer 3 的机制证据：
**agent 的贡献是证据驱动的假设生成/证伪和假阳性剔除，不是新架构。**

---

## 6. 整体 E1 结果：rule controller 不是更好的系统

不要因为 blind_0003 的 +2.41 dB 就认为 rule controller 更好。E1 全量 24 场景：

| Controller | DIV2K PSNR | DIV2K SSIM | DIV2K LPIPS | LSDIR PSNR | LSDIR SSIM | LSDIR LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| AutoIR | 24.968 | 0.7326 | 0.4050 | 22.956 | 0.6756 | 0.4009 |
| rule | 24.272 | 0.7124 | 0.4237 | 22.131 | 0.6491 | 0.4231 |

Paired rule − AutoIR：

```text
DIV2K: -0.696 dB [-3.427, +2.275], wins 8/24
LSDIR: -0.825 dB [-3.410, +1.778], wins 8/24
worst: -20.5 dB, best: +23.0 dB
```

所以正确的表述是：

> rule controller 是一个**强但高方差的确定性 ablation**；它在 8/24 场景赢，在 16/24 输，整体低于 AutoIR。blind_0003 是它赢的 8 个场景之一。E1 的作用是隔离 controller 层，而不是证明规则系统更好。

**极值案例**：`best +23.0` 就是 `blind_0005`（impulse noise）。规则控制器在 R1 选 `noise_impulse:5`，固定 LQ 44.92/41.14；AutoIR 盲选 `brightness:1+impulse:3`，21.92/22.17。这个案例说明：规则控制器在候选表预置正确项时可以极强，也直接证明该 restorer 足以处理 impulse；详见 `case_blind_0005/report.md`。

**补充：无反思 one-shot 对照**。为区分“反思”与“controller”，新增 E1-oneshot（规则 R0 预测、只训练一次、不跑 R1/R2）和 AutoIR-oneshot（AutoIR R0 预测、只训练一次）。23 个共同场景的 DIV2K 结果：  
- 规则反思：`rule-full − rule-oneshot = +2.43 dB [1.35,3.67]`，18/23 胜；  
- LLM 反思：`AutoIR-full − AutoIR-oneshot = +1.28 dB [0.39,2.68]`，15/23 胜，且被 `blind_0014 +14.38` 一个场景主导；  
- 有反思的 controller 差异：`AutoIR-full − rule-full = +0.636 dB [−2.44,3.49]`，不显著。  
结论：反思本身有效，但**LLM 反思的平均增益并不优于规则反思**；LLM 的优势更偏“初始预测/个别尾部场景”。E1-oneshot 在 `blind_0020` 的规则 R0 训练 NaN，按 **1/24 terminal failure** 报告；E1-full 通过 R1 fallback 恢复，不静默删除、不做 0 dB 插补。详见 `E1_oneshot_no_reflection.md`。逐场景选择差异见 `LLM_vs_rule_choices.md`。

---

## 7. 构造上的关键审计点

1. **规则冻结时间**：`rules_frozen.json` 版本 `exp41-rebuttal-rule-v1`，`frozen_at=2026-10-24`，E1 在 2026-10-25 执行；规则是先冻结再跑，不是事后调参。
2. **不是所有 severity map 都从 JSON 读取**：`mean_shift` 的 severity 在代码里用 `abs(ms)/15` 的归一化偏离实现，和 JSON 里的 `[15,25,40,60]` 在切点等价，但严格说代码不是完全由 `rules_frozen.json` 驱动。
3. **噪声映射 bug**：`NOISE_FUNCS` 里 `GAUSSIAN` 曾先于 `GAUSSIAN_YCRC` 匹配，导致 5 个场景的 YCrCb 被映射成 RGB；`E1_fix` 修正后重跑，`merge_e1_selection.py` 在平票时优先 `_fixed`，最终选中 1 个 `R0_fixed`。这说明 rule controller 的“确定性”也依赖代码/映射正确性，必须审计。
4. **R2 实际很少被选中**：9 个 R2 训练，最终 0 个被选；support PSNR 的 0.05 dB 容差 + 优先更高 round 的规则并没有让 R2 胜出。这说明 rule controller 的“第二轮反思”在 support 选择上收益有限。
5. **5 张 support 的选择噪声**：最终选择完全基于 5 张 support 的 PSNR；0.05 dB 容差很小，容易受单图波动影响。这也是为什么 rule controller 的方差很大。
6. **候选表是人工设计的**：glass 在 blind_0003 的候选表里，所以它赢了；但这也意味着 rule controller 的优势依赖于“候选表是否覆盖了 GT 子类型”。如果 GT 子类型不在表里，它不会像 LLM 那样主动扩展（除非 near-miss 逻辑碰巧覆盖）。
7. **LQ 配对**：17/24 场景 A0/E1/one-shot 的 LQ hash 一致；7/24 非确定性场景（0003/0005/0017/0018/0020/0022/0023）原始 A0/E1 的 `lq_sha256` 不一致（8,400 行中 2,450 行 = 7×350），原因是 `noise_impulse`/`blur_glass` 使用私有 RNG；**GT 管线未变**，变的是同一 GT 的随机 realization。已用 `E1E2_fixed_lq` 同一份固定 LQ 重评，宏观结论不变；未修正前这 7 个场景的逐图 Δ 不能当严格配对。

---

## 8. 可直接写进论文/rebuttal 的段落

### 中文版

> rule-based controller 是一个确定性的 signal-to-pipeline 状态机，不是学习模型。它和 AutoIR 共享同一个 `run_full_analysis` 信号提取器、同一个 `verify_signals` 噪声验证器、同一批 5 张 support、同样的 Direct/Swin+SG 训练预算和同一个固定 LQ 测试集；它的 controller 层由冻结规则组成：逐对图阈值映射、5 对图 3/5 多数投票 + severity 中位数、固定候选枚举（near-miss、删步、severity±1/±2、子类型替换）、确定性退化用 PSNR、噪声用 signal 评分，以及 support PSNR 触发的 R0→R1→R2 轮次。所有阈值、severity 映射、架构规则和反思边界都在执行前冻结（`rules_frozen.json`, v1, 2026-10-24）。在 24 个场景上，该 controller 整体不优于 AutoIR（DIV2K −0.70 dB、LSDIR −0.82 dB，8/24 胜），但在 `blind_0003` 上以 +2.41/+1.86 dB 胜出，因为 `blur_glass` 就在它的固定子类型候选表中。这隔离了 controller 层：当正确候选被预先枚举时，规则控制器可以赢；LLM 控制器能补充候选表外的选项，但可能被保守的反思规则压制。

### English version

> The rule-based controller is a deterministic signal-to-pipeline state machine, not a learned model. It shares the same `run_full_analysis` signal extractor, the same `verify_signals` noise checker, the same five support pairs, the same Direct/Swin+SG training budget, and the same fixed-LQ test set as AutoIR. Its controller layer consists of frozen rules: per-pair threshold mapping, 3/5 majority aggregation with median severity, a fixed candidate enumeration (near-miss, remove-step, severity ±1/±2, subtype alternatives), PSNR scoring for deterministic degradations and signal scoring for noise, and support-PSNR-triggered R0→R1→R2 rounds. All thresholds, severity maps, architecture rules, and reflection margins were frozen before execution (`rules_frozen.json`, v1, 2026-10-24). On 24 scenarios, this controller is not better overall (−0.70 dB DIV2K, −0.82 dB LSDIR; 8/24 wins), but it wins `blind_0003` (+2.41 dB DIV2K, +1.86 dB LSDIR) because `blur_glass` is present in its fixed subtype candidate list. This isolates the controller layer: the rule controller can win when the correct candidate is pre-enumerated, while the LLM controller can add candidates outside the list but may be suppressed by conservative reflection rules.

---

## 9. 文件索引

```text
rebuttal_runs/scripts/rule_controller.py        ← 控制器实现（612 行）
rebuttal_runs/E1/rules_frozen.json              ← 冻结阈值/severity/架构/反思规则
rebuttal_runs/E1/decisions/blind_0003.json      ← 该场景候选打分与 R1 选择
rebuttal_runs/E1/selection.jsonl                ← 每场景最终选中的 round/checkpoint
rebuttal_runs/E1/selection_summary.json         ← 24 场景选择分布
rebuttal_runs/E1/e1_final_summary.json          ← rule vs AutoIR 固定 LQ 汇总
rebuttal_runs/scripts/merge_e1_selection.py     ← support PSNR 选模型
rebuttal_runs/scripts/aggregate_e1_final.py     ← 固定 LQ 聚合 + paired bootstrap
rebuttal_runs/E1_fix/                           ← 噪声映射修正后的重跑
```

---

## 10. 复现命令

```bash
# 查看冻结规则
python3 -m json.tool rebuttal_runs/E1/rules_frozen.json

# 查看 blind_0003 的规则控制器决策
python3 -m json.tool rebuttal_runs/E1/decisions/blind_0003.json

# 查看 24 场景最终选择分布
cat rebuttal_runs/E1/selection_summary.json

# 查看 rule vs AutoIR 全量对比
cat rebuttal_runs/E1/e1_final_summary.md

# 重新生成 E1 最终汇总（需要已有 test_metrics/E1 test_metrics）
.venv/bin/python3 rebuttal_runs/scripts/aggregate_e1_final.py
```
