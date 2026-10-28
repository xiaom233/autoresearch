# blind_0020：规则控制器与 LLM 的逐步决策对比

> 目的：回答 Reviewer 3「LLM controller 的收益没有被独立验证；需要证明 agent reasoning 相对规则/常规搜索的贡献」。
> 本文件逐轮记录 **规则控制器** 与 **LLM/AutoIR** 在同一 support、同一训练预算下的决策依据与结果。
> 这是 E1 24 场景中的**一个案例**，不是总体结论；总体结果见 `rebuttal_runs/E1/e1_final_summary.json`、
> `LLM_vs_rule_choices.md` 和 `E1_oneshot_no_reflection.md`。GT 只用于事后评估，盲识别/反思过程不读取 GT。

## 0. 结论先行

| 项 | 规则控制器 | LLM / AutoIR |
|---|---|---|
| R0 DC | `blur_gaussian:5 + contrast_weaken_scale:5 + saturate_weaken_HSV:5` | `blur_gaussian:4 + contrast_weaken_stretch:4` |
| R0 结果 | Swin+GCM，**loss NaN（step 11319 起）**，无可用模型 | direct_swin，support 15.809；fixed-LQ test **15.779 / 15.023** |
| R1 | fallback Swin+SG；`gaussian:5 + stretch:5 + sat:5`；support 15.736；test **15.991 / 15.227** | 同 DC，改 Swin+SG；support 16.490；test **16.975 / 16.421** |
| R2 | 预计算的第二候选 `gaussian:5 + scale:3 + sat:5`；support 14.246；test **14.118 / 14.04**（未选中） | R2 发现 `blur_glass:4`；`glass:4 + stretch:4` + ColorPre；support 17.858；test **18.084 / 17.409** |
| 最终选中 | R1（15.991 / 15.227） | R2（18.084 / 17.409） |
| 选定差值 | — | **+2.093 DIV2K / +2.182 LSDIR**（fixed-LQ） |

- 规则 R0 的 NaN 不是主要差距来源：R1 fallback 已恢复到 15.991。
- 规则的主要问题是 **DC 从未被质疑**：`blur_gaussian 5/5`、`saturation_weaken 5/5` 是错误的高置信信号，
  而规则的映射器和候选生成器都不会挑战它们。
- LLM 的贡献是 **证据驱动的假设检验**：R2 主动测试了 `glass_score=0.246`（刚好低于 0.25 阈值）的
  `blur_glass:4`，仿真 PSNR 从 37.73 升到 **39.56 dB（CI 10/10）**；同时明确拒绝了 saturation、noise、gamma、JPEG。
- 在**同架构（Swin+SG）**下，LLM 的 R1 DC 已经比规则 R1 高 **+0.984 dB DIV2K / +1.194 dB LSDIR**
  （16.975 vs 15.991；16.421 vs 15.227）。这是纯 DC 对比，不含 glass 和 ColorPre。
- 仍有限制：两边都漏掉 `noise_poisson:3`；LLM 最终也是 2/3 步；这是单场景，不代表普遍优势。
- 同架构控制（LLM final DC `glass:4+stretch:4` + Swin+SG，替换 ColorPre）正在训练，用于分离 glass 与 ColorPre 的贡献；
  结果出来后追加。现有同架构 R1 对比已能证明 **DC 本身有贡献**。

## 1. 数据与 GT（事后）

- 场景 `blind_0020`；support 为 `exp41/challenges/phase4/blind_0020/` 的 5 对图像；
  test 为 fixed-LQ 的 DIV2K 100 + LSDIR 250（`rebuttal_runs/E1E2_fixed_lq/blind_0020/`）。
- GT（仅事后）：`noise_poisson:3 → blur_glass:4 → contrast_weaken_stretch:4`。
- test 数值统一用 `E1E2_fixed_lq` 同一份 LQ；规则 R0 原始 LQ 与 A0/E1 不一致的问题不影响本案例的 fixed-LQ 结论。

## 2. 规则控制器：逐步决策

### 2.1 5 对 support 的原始信号（`rebuttal_runs/E1/cache/blind_0020_*.json`）

| pair | blur verdict | gm_ratio | glass_score | variance_ratio | saturation_ratio | noise sigma |
|---|---|---|---:|---:|---:|---:|
| 0 | GAUSSIAN_BLUR | 0.0953 | 0.2462 | 0.1008 | 0.4417 | 1.11 |
| 1 | GAUSSIAN_BLUR | 0.0895 | 0.2432 | 0.1817 | 0.3900 | 2.84 |
| 2 | GAUSSIAN_BLUR | 0.1414 | 0.2321 | 0.1370 | 0.3049 | 0.74 |
| 3 | MOTION_BLUR | 0.0844 | 0.2394 | 0.1516 | 0.4959 | 2.10 |
| 4 | GLASS_BLUR | 0.3529 | 0.5804 | 0.6141 | 0.4898 | 1.48 |

- compression：5 对均 `MILD_COMPRESSION_POSSIBLE`，但 `block_boundary≈1.0 < 1.05`，不加 compression。
- noise：只有 pair1/3 的 σ≥2（2/5），未达 3/5 共识 → 不加 noise。**漏掉 GT 的 poisson:3。**
- blur：分析器在 pair4 明确给 `GLASS_BLUR`，但下一步的映射器不读这个 subtype（见 2.2）。
- global：5 对都给 `CONTRAST_weaken_sev5 + SATURATION_weaken_sev5` 的原始信号；其中 saturation 是 contrast 的伪信号。

### 2.2 单对信号 → DC（`rule_controller.py:81-175`）

- blur 分支只按 `FFT_anisotropy` 选 `blur_motion`/`blur_gaussian`，**没有 glass 分支**；
  severity 用 `map_low(gm_ratio, [0.1,0.2,0.35,0.5])`，本案例 median 为 5。
- contrast 分支：`variance_ratio < 1` → `contrast_weaken_scale`，`map_dev` 给 sev 5。
- saturation 分支：`chroma_ratio < low` → `saturate_weaken_HSV`，sev 5。
- 5 对聚合（`aggregate`，`rule_controller.py:185-196`）：按类别投票，≥3/5 才纳入，severity 取中位数：

| 类别 | 函数 | votes | median severity |
|---|---|---:|---:|
| blur | `blur_gaussian` | 5/5 | 5 |
| contrast | `contrast_weaken_scale` | 4/5 | 5 |
| saturation | `saturate_weaken_HSV` | 5/5 | 5 |

→ **R0 DC = `blur_gaussian:5 + contrast_weaken_scale:5 + saturate_weaken_HSV:5`**，仿真 PSNR **19.83**。

### 2.3 R0 训练

- 架构规则 `rs_for` 因 contrast+structured 选择 **Swin+GCM**（`contrast_structured_gcm`）。
- 训练日志 `rebuttal_runs/E1/logs/blind_0020_R0.log`：从 step 11319 起 `loss: nan`，`train_R0.nan=true`，
  训练 5420.9s；规则把 R0 判为不可用，没有可用的 R0 checkpoint。
- NaN 的机制是 **GCM + 极端 contrast（variance_ratio≈0.10）** 的数值不稳定；不是盲识别本身。
  规则没有在设计阶段规避这个组合。

### 2.4 R1：fallback + 候选扰动

- 因为 R0 NaN 且原架构含 GCM，`decide` 把架构 fallback 为 **Swin+SimpleGate**（`arch_fallback_R1`）。
- R1 不是一个新假设，而是 **plan 阶段预计算的候选列表中的 best**。候选生成规则（`candidates_for`）：
  - `lowest = min(agg, key=(votes, -CAT_RANK))`；本案例 blur 5/5、saturation 5/5、contrast 4/5
    → **lowest = contrast**；
  - 只对 contrast 做 severity 扫描、subtype 替换、remove-step；**不会测试 blur 的 glass，也不会移除 blur/saturation**。
- 候选与分数（`E1/decisions/blind_0020.json`）：

| 候选 | 仿真 PSNR |
|---|---:|
| `blur_gaussian:5 + contrast_weaken_stretch:5 + saturate_weaken_HSV:5`（R1，选中） | **25.69** |
| `blur_gaussian:5 + contrast_weaken_scale:3 + saturate_weaken_HSV:5`（R2，预计算） | 20.77 |
| `blur_gaussian:5 + contrast_weaken_scale:4 + saturate_weaken_HSV:5` | 20.70 |
| `blur_gaussian:5 + saturate_weaken_HSV:5`（移除 contrast） | 15.40 |
| `blur_gaussian:5 + contrast_strengthen_stretch:5 + saturate_weaken_HSV:5` | 9.07 |
| `blur_gaussian:5 + contrast_strengthen_scale:5 + saturate_weaken_HSV:5` | 9.00 |

- R1 选 `stretch:5`（contrast 子类型修正），但 **blur 仍是 gaussian:5，saturation 仍在**。
- R1 训练：Swin+SG；support 15.736（selection；日志 best 15.63）；fixed-LQ test **15.991 / 15.227**。

### 2.5 R2：预计算第二候选

- R2 也不是基于 R1 训练结果重新分析，而是 plan 阶段预计算的 **second-best distinct candidate**：
  `blur_gaussian:5 + contrast_weaken_scale:3 + saturate_weaken_HSV:5`（仿真 20.77）。
- R2 训练：Swin+SG；support 14.246；fixed-LQ test **14.118 / 14.04**（低于 R1，因此最终选 R1）。
- 因此规则控制器的“反思”是 **预计算的 R1/R2 候选 + 训练反馈决定是否继续**，
  不是“根据上一轮结果生成新假设”。关键证据：
  - `rule_controller.py:484-520`：R1/R2 在 plan 阶段一次性写入；
  - `rule_controller.py:553-591`：decide 只读取 `pipeline_R1/R2` 并决定是否训练；
  - 候选生成只看最低票类别（contrast），高票的 blur/saturation 永远不会被证伪。

## 3. LLM / AutoIR：逐步决策

LLM 的完整记录在 `exp41/challenges/phase4/blind_0020/reflection.json`（28 iterations）；
下表摘录决定性的测试。

### 3.1 R0：先拒绝假阳性，再找 contrast 子类型

初始分析（`initial_analysis`）：

- noise：`sigma=1.11 < 2` + Qwen=NO；加 noise 使 PSNR 从 37.73 掉到 18.86（−18.9 dB）→ **无噪声**。
- compression：`unique_G=102<200` 但 `block_boundary=1.007`；判定为 extreme contrast 的假阳性 → **无 JPEG**。
- global：`variance_ratio=0.101` → contrast weaken；stretch vs scale **+10.7 dB**；saturation `chroma_ratio=0.442`
  但加入后 PSNR 掉 6.3 dB → **saturation 是 contrast 的假阳性**。
- blur：`gm_ratio=0.095` 高斯；anisotropy=1.93 提示 motion，但 gaussian 比 motion 高 8.5 dB。

R0 关键候选（17 次）：

| 候选 | PSNR | 决策 |
|---|---:|---|
| `compression_jpeg:3` | 15.41 | REJECT |
| `blur_gaussian:5 + contrast_weaken_scale:5` | 25.03 | TRY |
| `blur_gaussian:5 + contrast_weaken_stretch:4` | 36.96 | 关键：stretch 不是 scale |
| `blur_gaussian:5 + contrast_weaken_stretch:3` | 32.59 | severity 扫描 |
| `blur_gaussian:4 + contrast_weaken_stretch:4` | **37.73** | **R0 best，CI 8/10** |
| `blur_gaussian:4 + contrast_weaken_stretch:4 + saturate_weaken_HSV:4` | 31.45 | saturation 加入变差 → 拒绝 |
| `blur_gaussian:4 + contrast_weaken_stretch:4 + noise_gaussian_RGB:1` | 25.73 | 加 noise 变差 → 拒绝 |
| `blur_motion:4 + contrast_weaken_stretch:4` | 29.21 | motion 不如 gaussian |

→ R0 DC `gaussian:4 + stretch:4`，仿真 37.73；训练 `direct_swin`（无 GCM），support 15.809，
fixed-LQ test **15.779 / 15.023**。

### 3.2 R1：重验 + 训练稳定性诊断

R1 只有 3 次新测试：

- 重验 R0 best：37.73 / CI 8/10。
- 加 `noise_gaussian_RGB:1`：25.73 → 再次拒绝 noise。
- `blur_lens:4 + stretch:4`：37.40 → 保留为 alternative。

R1 的文字结论（`predicted_params_R1.json`）：**“训练 NaN 是 FiLM-GCM + 极端 contrast 的组合问题，不是盲识别错误；
建议 Swin+Direct，禁止 FiLM-GCM。”** 规则 R0 的 NaN 正好验证了这一点；LLM 在策略层做了正确的稳定性诊断。
R1 DC 不变，架构改为 **Swin+SG**；support 16.490；fixed-LQ test **16.975 / 16.421**。

> 历史记录说明：AutoIR 的 R1 reflection 记录了 NaN 事件，但 A0 实际使用的 R0 日志是 `direct_swin`（无 GCM）且正常收敛；
> 原始 NaN 日志未保留。因此这里只声称“LLM 在 R1 诊断并规避了 GCM+极端 contrast 的 NaN 风险”，
> 不声称“AutoIR R0 最终训练 NaN”。

### 3.3 R2：主动测试 near-threshold 的 glass

R2 的 8 次关键测试：

| 候选 | PSNR | 决策 |
|---|---:|---|
| `blur_gaussian:4 + brightness_darken_gamma_RGB:3` | 12.04 | REJECT（gamma 替代 contrast） |
| `blur_gaussian:4 + stretch:4 + compression_jpeg:1` | 35.69 | 变差 → 拒绝 JPEG |
| **`blur_glass:4 + contrast_weaken_stretch:4`** | **39.56 / CI 10/10** | **发现 glass（+1.83 dB vs gaussian）** |
| `blur_glass:3 + stretch:4` | 38.22 | glass severity 扫描 |
| `blur_glass:5 + stretch:4` | 38.62 | glass severity 扫描 |
| `blur_glass:4 + stretch:3` | 32.01 | contrast severity 扫描 |
| `blur_glass:4 + stretch:5` | 32.01 | contrast severity 扫描 |

关键依据：`glass_score=0.246`，**刚好低于 0.25 的决策阈值**；R0/R1 的 gaussian 判据把它截断，
R2 主动把边界假设拿出来做 PSNR 验证，得到 CI 10/10 的 **+1.83 dB**。这是 agent 在固定阈值之外生成/验证假设的直接证据。

→ R2 DC `glass:4 + stretch:4`，仿真 39.56；架构 Swin+SG+ColorPre；support 17.858；
fixed-LQ test **18.084 / 17.409**（A0 选中 R2）。

## 4. 逐步对照

| 维度 | 规则控制器 | LLM / AutoIR |
|---|---|---|
| noise | 2/5 票（σ≥2），不足 3/5 → 不加；漏 poisson | σ=1.11<2+Qwen NO；加 noise −18.9 dB → 拒绝；也漏 poisson |
| blur subtype | `dc_from_one` 无 glass 分支；pair4 的 GLASS_BLUR 被映射成 gaussian；5/5 高置信 | R0 gaussian:4；R2 主动测 glass:4（glass_score=0.246），+1.83 dB / CI 10/10 |
| blur severity | gaussian:5（median map_low） | gaussian:4 / glass:4（PSNR 扫描） |
| contrast subtype | R0 scale:5（错）；R1 stretch:5（对） | stretch:4（R0 即正确，+10.7 dB vs scale） |
| contrast severity | 5 | 4（:3/:5 都变差） |
| saturation | 5/5 高置信 `weaken:5`（假阳性）；候选策略从不移除 | R0 实测加入后 −6.3 dB → 拒绝 |
| 架构 | R0 Swin+GCM → NaN；R1 fallback Swin+SG；R2 Swin+SG | R0 direct_swin；R1 Swin+SG；R2 Swin+SG+ColorPre |
| 反思机制 | plan 阶段预计算 R1/R2；decide 只决定是否训练 | R1/R2 重新分析并生成新候选（noise 重验、glass 边界测试） |
| selected | R1（support 15.736） | R2（support 17.858） |
| fixed-LQ test | **15.991 / 15.227** | **18.084 / 17.409** |

## 5. agent 的贡献是什么（不是新架构）

1. **证伪高置信假阳性**：saturation 5/5、noise 2/5、JPEG mild 都是规则/分析器的假阳性；
   LLM 用“加上去 PSNR 变差”的反事实测试逐个拒绝，而规则的候选生成只扰动最低票类别，不会移除高票步骤。
2. **突破阈值边界**：glass_score=0.246 低于 0.25；LLM 把边界假设拿出来验证，得到 +1.83 dB / CI 10/10；
   规则的 `dc_from_one` 根本没有 glass 分支。
3. **训练稳定性诊断**：LLM R1 把 NaN 归因于 FiLM-GCM + 极端 contrast，并切换到 Swin；
   规则只是事后 fallback（R0 NaN → R1 Swin+SG）。
4. **不是架构发明**：两边都在 Swin 家族内；ColorPre 是策略规则选的，不是 agent 新设计的模块。
   贡献在 **假设生成、证伪和 DC 选择**。

## 6. 归因与混杂

### 6.1 fixed-LQ ablation（本次新增）

| 模型 | DC | 架构 | support | DIV2K | LSDIR |
|---|---|---|---:|---:|---:|
| AutoIR R0 | gaussian:4 + stretch:4 | direct_swin（无 SG） | 15.809 | 15.779 | 15.023 |
| AutoIR R1 | gaussian:4 + stretch:4 | Swin+SG | 16.490 | **16.975** | **16.421** |
| AutoIR R2 | glass:4 + stretch:4 | Swin+SG+ColorPre | 17.858 | **18.084** | **17.409** |
| 规则 R1 | gaussian:5 + stretch:5 + sat:5 | Swin+SG | 15.736 | **15.991** | **15.227** |
| 规则 R2 | gaussian:5 + scale:3 + sat:5 | Swin+SG | 14.246 | 14.118 | 14.040 |
| 规则 R0 | gaussian:5 + scale:5 + sat:5 | Swin+GCM | — | NaN | NaN |

- **同架构纯 DC 对比**：AutoIR R1 vs 规则 R1（均 Swin+SG）→ **+0.984 DIV2K / +1.194 LSDIR**。
  这时 LLM 还没有 glass，也没有 ColorPre；差异只来自 DC（gaussian:4 vs gaussian:5；无 saturation；stretch:4 vs stretch:5）。
- **AutoIR R0→R1**（同 DC，加 SimpleGate）→ +1.196 / +1.398；**AutoIR R1→R2**（glass + ColorPre）→ +1.109 / +0.988。
- **仿真层面**：规则 R1 DC 25.69 vs LLM R2 DC 39.56 → **+13.87 dB**，与架构无关。
- 同架构控制（LLM final DC `glass:4+stretch:4` + Swin+SG，替换 ColorPre）正在训练；
  完成后可把 R1→R2 的 +1.109 拆成 glass-DC 与 ColorPre 两部分。

### 6.2 混杂与限制

- 规则 R0 NaN 使 R0 不可比；但规则 R1 fallback 已经恢复到 15.991，NaN 本身不是 2.09 dB 差距的全部原因。
- 最终模型架构不同（AutoIR R2 ColorPre vs 规则 R1 Swin+SG）；同架构 R1 对比已排除该混杂并仍有 +0.98 dB。
- 两边都漏掉 GT 的 `noise_poisson:3`；LLM 最终 2/3 步，不是完整解。
- 只用 5 对 support 选 checkpoint；single scenario；没有 run-to-run 方差；fixed-LQ test 是同一份 LQ。
- 规则 R2 的 `train_R2=null` 是 decision 文件的旧字段，R2 实际训练/评估记录在 `E1/selection.jsonl` per_round 与 `E1/logs/blind_0020_R2.log`。

## 7. 可直接用于 Reviewer 3 的段落

**中文**：

> 在 `blind_0020` 上，规则控制器把 5/5 票的 Gaussian blur 与 saturation 当作高置信信号，但它的
> `dc_from_one` 没有 glass 分支、候选生成只扰动最低票类别（contrast 4/5），所以 R1/R2 只是在预计算的
> 候选里选 best/second-best，从未检验“glass”或“移除 saturation”。LLM 则用反事实 PSNR 测试拒绝了
> noise、saturation、gamma、JPEG，并主动测试了刚好低于 0.25 阈值的 `glass_score=0.246`，
> 把仿真 DC 从 37.73 提升到 39.56 dB（CI 10/10）。在**同一 Swin+SG 架构**下，LLM 的 R1 DC 已经比规则
> R1 高 +0.98 dB（16.975 vs 15.991，fixed-LQ DIV2K）；最终 AutoIR R2（glass + ColorPre）为 18.084，
> 规则为 15.991。该案例说明 agent 的贡献是**证据驱动的假设选择与假阳性剔除**，不是新架构；
> 它是单场景示例，不构成“LLM 普遍优于规则”的结论（E1-oneshot 总体的 LLM−rule 差异 CI 跨 0）。

**English**：

> On `blind_0020`, the rule controller treats its 5/5 Gaussian-blur and saturation votes as high-confidence
> evidence, but its mapper has no glass branch and its candidate generator perturbs only the lowest-vote category
> (contrast, 4/5); R1/R2 are therefore the best/second-best precomputed candidates, not new hypotheses.
> The agent instead ran counterfactual PSNR tests that rejected noise, saturation, gamma and JPEG, and explicitly
> tested a below-threshold glass hypothesis (`glass_score=0.246` vs the 0.25 cut), improving the simulated DC from
> 37.73 to 39.56 dB (CI 10/10). At the **same Swin+SG architecture**, the agent's R1 DC already beats the rule's
> selected R1 by +0.98 dB (16.975 vs 15.991, fixed-LQ DIV2K); the final agent run with glass+ColorPre reaches 18.084
> vs 15.991. This illustrates that the agent's contribution is evidence-driven hypothesis selection and false-positive
> rejection, not architecture invention. It is a single-scenario illustration, not evidence of uniform superiority
> (the overall E1-oneshot LLM−rule CI crosses zero).

## 8. 产物

```text
exp41/challenges/phase4/blind_0020/reflection.json          # LLM 28 iterations
exp41/challenges/phase4/blind_0020/predicted_params*.json   # R0/R1/R2
rebuttal_runs/E1/decisions/blind_0020.json                   # 规则 votes / candidates / R0-R2
rebuttal_runs/E1/cache/blind_0020_{0..4}.json                # 规则 5 对原始信号
rebuttal_runs/E1/logs/blind_0020_R0/R1/R2.log               # 规则训练日志（R0 NaN）
rebuttal_runs/E1/selection.jsonl                            # 规则选中 R1
rebuttal_runs/A0/support_selection.jsonl                    # LLM per-round support
rebuttal_runs/E1E2_fixed_lq_summary.json                    # fixed-LQ 15.991/18.084
rebuttal_runs/E1/blind_0020_ablation_metrics.jsonl          # R0/R1/R2 fixed-LQ ablation
rebuttal_runs/E1/blind_0020_ctrl_llmDC_swinsg/              # 同架构控制（进行中）
```
