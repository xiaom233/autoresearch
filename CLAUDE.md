# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**语言偏好：优先使用中文回答。所有与用户的沟通、代码注释、commit 信息均使用中文。**

## ⚠️ 关键规则

0. **Phase 切换时强制重读文档 🔴**：每个 Phase 开始前，必须重新阅读对应的协议文档。**禁止凭记忆操作**：
   - Phase 4 (盲识别): `Read` SKILL.md + CLAUDE.md §子 Agent 盲识别协议
   - Phase 5 (训练): `Read` CLAUDE.md §Phase 5 训练启动 + `Read` finetune_strategy.md
   - Phase 6 (DFPIR 对比): `Read` CLAUDE.md §DFPIR 基线评估 (仅在全部流程结束后)
   - GT 重评估: `Read` CLAUDE.md §GT 退化重评估
   - 反思: `Read` REFLECTION_MECHANISM.md
   - 违反后果: 架构遗漏(ColorPre/ColorMLP)、策略错误(Ft vs Direct)、FP问题未处理
1. **绝不接触 GT**：以下路径全程不可读（来源：exp18/exp19 泄露审计）：
   - `.ground_truth/` 和 `degradation_gt/` → 已移出 challenges/ 目录，chmod 700/600
   - `/tmp/expN_p4_mapping.json` → 已迁移至 `.gt_mappings/`（chmod 700）
   - Agent prompt 中必须明确列出所有禁止读取的路径
   - `.ground_truth.json` 不再存放于 Agent 可访问的 challenges/ 目录中
2. **VAL 锁定**：反思修正只改 PARAMS，VAL 永远指向 GT
3. **同图盲识别**：挑战生成用 `--same-image`
4. **不用 rebase**：只用 merge/push
5. **launcher 隔离**：训练用 `bash launcher.sh`，不写完整命令
6. **已完成的训练不重跑**
7. **执行前验证文件存在**：任何脚本或文件路径使用前，先 `ls` 确认存在。绝不假设已知路径有效：启动 M_blind 前先检查 checkpoint 是否存在
8. **全局退化不用 Ft**：管线含 contrast/brightness/saturation/gamma → 用 Direct（Ft 可能崩溃 -4.51 dB，来源：exp9）
9. **EPOCH_BUDGET 公平性**：所有修正训练必须 EPOCH=2，与基线一致，禁止通过增加训练步数获得 PSNR 提升
10. **架构选择参考**：严重 motion blur → OCAB + ws=16；全局退化 → 只用 Swin（避免 MDTA/OCAB）；contrast + 结构化 → FiLM-GCM（来源：exp10）
11. **禁止删除实验数据**：`expN/` 下的关键数据（predicted_params/、reflection/、logs/、results/、checkpoints/）绝对不可删除。除非用户明确允许。误删不可恢复（git 不追踪实验文件）。
12. **预测结果不覆盖**：save_prediction.py 自动版本化（_v2, _v3...），禁止手动覆盖已有预测文件。每次盲识别创建新文件，保留历史对比。
   ```bash
   ls expN/experiments/M_blind/checkpoints/*.pt && echo "已存在，跳过" || echo "需要训练"
   ```
   如果已有 checkpoint（含 val_psnr_db 的训练日志），绝不重新训练

### 子 Agent 盲识别协议 🔴 强制执行

**盲识别必须通过 Skill `image-degradation-simulator` 启动子 Agent 执行。禁止用脚本或 auto_pipeline 直接生成预测。**

**🔴 exp37 修正: verdict 分退化独立评估。pipeline 每步含自有 verdict，不再整体一刀切。**
- noise 步骤 UNCERTAIN 不阻止 contrast 步骤 LIKELY 触发 FiLM-GCM
- 组件触发看该退化自身的 verdict，不被其他步骤拖累
- `overall_verdict` 仅作统计参考，不影响策略选择

**🔴 Agent 命名和 prompt 禁止泄露退化信息（来源：exp36 审计）**：
- Agent 名称必须中性（如 "Blind ID batch A"），**禁止**包含 "single-deg"、"double-deg"、"triple-deg" 等退化步数提示
- Agent prompt **禁止**提及退化步数（"单退化"、"双退化"、"三退化"）
- Agent prompt **禁止**提及退化类型、严重度、或任何从 GT 推导的信息
- Agent 对退化完全一无所知——步数、类型、严重度均需从信号分析中自行判断
- blind_challenge.py 的 stdout 输出已确保不泄露 pipeline 信息（仅打印 challenge_id + 文件路径）

**🔴 子 Agent 每完成一个退化的识别后，必须重新读取 SKILL.md（来源：exp22 v12 Agent 未严格遵循 Step 1→4 顺序，退化到旧习惯）。每次开始分析新的 challenge 前，先 `Read` SKILL.md 确认当前协议，防止长时间运行后遗忘流程。**

#### 核心定位

```
盲识别的目标不是"找到唯一正确答案"——
而是输出一个带不确定性量化的候选列表，供训练后 PSNR 做最终仲裁。

原因:
  1. 确定性退化 (blur, compression, contrast): PSNR 可验证 (>40dB = 正确)
  2. 随机退化 (noise): PSNR 不可用 → 统计分布匹配
  3. 混合退化 (blur+noise+jpeg): 退化耦合，无法逐级剥离验证
     - noise→blur: 模糊改变了噪声统计
     - blur→noise: 噪声干扰了 PSNR
     - jpeg→contrast: 拉伸改变了块效应
  4. 唯一的可靠判据: 训练后 PSNR (Spec < M_blind - 3dB → 盲识别错误)
```

来源: exp17 — 16 Agent 盲识别，平均 CI=8.2/10，函数正确率仅 6%。

#### 分类型判断框架

```
退化分类:
  Tier 1 — 确定性单步 (blur, compression, contrast, brightness, quantize 单独):
    判据: PSNR >= 40dB → GOOD, 30-40dB → 调严重度, < 30dB → 换函数

  Tier 2 — 随机单步 (noise 单独):
    判据: 残差统计匹配 (SKILL.md §B 6步检查)
    - impulse: extreme_pct 误差 < 10%
    - gaussian: flat_region_var 误差 < 15%
    - speckle: var_mean_slope 误差 < 0.01
    - poisson: var_slope ~1.0 + vm_slope ~0
    - spatial_corr: spatial_correlation 误差 < 0.05

  Tier 3 — 混合退化 (多步, 含 noise, 退化耦合):
    判据: 多假设 PSNR 排名 + 统计匹配 + 证据一致性
    目标: 输出 ranked candidates → 训练 PSNR 做最终仲裁
    ⚠️ Tier 3 不接受 GOOD verdict — 必须是 UNCERTAIN 或 LIKELY
```

#### 强制工作流

```
1. analyze_degradation.py --target <d> --clean <c>
2. 分类:
   - 纯确定性? → Tier 1 (PSNR 验证)
   - 含 noise? → 先判 noise (残差统计), 再判其余 (PSNR)
   - 多步混合? → Tier 3 (候选排名)

3. 生成 3-5 个覆盖不同函数族的假设
   ⚠️ blur: gaussian/lens/zoom/glass 至少测 3 种
   ⚠️ noise: gaussian/speckle/poisson/impulse 至少测 2 种
   ⚠️ compression: jpeg/jpeg2000 至少测 2 种

4. 逐个模拟 + 评估:
   Tier 1/2: apply → PSNR 或统计匹配
   Tier 3: apply → PSNR + 统计 → 因耦合不可靠, 标记 UNCERTAIN

5. 输出 ranked candidates:
   | PSNR gap | verdict | 行动 |
   |:--:|------|------|
   | >= 10 dB | LIKELY | 主预测, 保留亚军为 alternative |
   | 3-10 dB | UNCERTAIN | 前 2 候选都保留 |
   | < 3 dB | UNCERTAIN | 前 3 候选都保留 |
   | max < 30 dB | POOR | 全部候选保留, 训练后决定 |

6. 保存:
   predicted_params.json: 主预测 + alternatives (仅保留有价值的)
   reflection.json: psnr_ranking(全量) + evidence_consistency + coupling_notes

#### Alternatives 质量要求 🔴

```
❌ 错误做法: 保存 PSNR 最高的 3 个, 即使 PSNR 都 < 30
   → exp18 验证: 0/16 GT命中, 全是垃圾数据

✅ 正确做法:
   1. 仅保存 PSNR >= 30 (确定性) 或 统计匹配 (噪声) 的候选
   2. 强制覆盖不同函数族: 即使 PSNR 更低, 至少保留1个不同 blur/noise/compression 类型的候选
   3. 所有候选 PSNR < 30 → alternatives 留空, verdict=POOR
      低PSNR的alternatives没有意义 (都是错误方向)
   4. PSNR >= 30 且 gap < 10dB → 至少保存 PSNR 最高的跨族候选
```

#### 强制交付物

| 文件 | 关键字段 |
|------|------|
| `predicted_params.json` | `{"pipeline":[...], "alternatives":[{pipeline, psnr, stats, function_family}], "analysis":{"verdict":"LIKELY|UNCERTAIN|POOR", "tier":1|2|3, "psnr":N, "psnr_gap":N}}` |
| `reflection.json` | `initial_analysis` + `psnr_ranking`(全量,含PSNR<30的也要记录) + `statistical_checks` + `coupling_analysis` + `iterations` + `final_decision` |

注意: reflection.json 记录全量 PSNR 排名 (含低分的), 但 predicted_params.json 的 alternatives 仅保留 >= 30 的高质量候选.

#### 判定标准

| verdict | 条件 | 含义 |
|------|------|------|
| LIKELY | Tier 1/2, PSNR >= 40 或统计匹配 | 高置信度, 可训练 |
| UNCERTAIN | Tier 3, 或多候选 PSNR 接近 | 不确定, 训练 PSNR 仲裁 |
| POOR | 所有候选 PSNR < 30 或统计不匹配 | 盲识别失败, 需重做 |

**🔴 禁止**：
- Tier 3 混合退化标 GOOD（退化耦合, 无法可靠验证）
- 用 CI 替代 PSNR/统计匹配做函数选择
- PSNR gap < 10 dB 声称确定
- 只测一种 blur/noise/compression 子类型
- 不记录全量 PSNR 排名表和耦合分析

### 🔴 Phase 5 前置守门：盲识别质量强制检查

**在启动任何 Phase 5 训练之前，必须先通过此检查。不通过 → 禁止训练 → 必须用 Agent 重做盲识别。**

#### 检查脚本

```bash
python3 -c "
import json, os, sys

exp = 'expN'  # 替换为当前实验
phase4 = f'{exp}/challenges/phase4'
errors = []

for bid in sorted(os.listdir(phase4)):
    d = os.path.join(phase4, bid)
    if not os.path.isdir(d): continue
    pf = os.path.join(d, 'predicted_params.json')
    rf = os.path.join(d, 'reflection.json')
    
    if not os.path.exists(pf):
        errors.append(f'{bid}: 缺少 predicted_params.json')
        continue
    
    pred = json.load(open(pf))
    analysis = pred.get('analysis', {})
    ci = analysis.get('ci_pass_rate', '0/10')
    verdict = analysis.get('verdict', 'N/A')
    ci_num = int(ci.split('/')[0])
    
    # 检查 1: reflection.json 必须存在
    if not os.path.exists(rf):
        errors.append(f'{bid}: 缺少 reflection.json → 未做 Agent 迭代推理')
    
    # 检查 2: CI=0 但 verdict=GOOD → 自相矛盾
    if ci_num == 0 and verdict == 'GOOD':
        errors.append(f'{bid}: CI=0/10 + GOOD → 脚本生成的无效结果, 必须 Agent 重做')
    
    # 检查 3: CI < 5 且 verdict != POOR
    if ci_num < 5 and verdict != 'POOR':
        errors.append(f'{bid}: CI={ci} + {verdict} → 严重不匹配, 必须重做')
    
    # 检查 4: reflection 内容完整性
    if os.path.exists(rf):
        ref = json.load(open(rf))
        if 'initial_analysis' not in ref:
            errors.append(f'{bid}: reflection 缺少 initial_analysis')
        if 'iterations' not in ref:
            errors.append(f'{bid}: reflection 缺少 iterations')
        if len(ref.get('iterations', [])) == 0 and ci_num < 7:
            errors.append(f'{bid}: CI={ci} < 7 但无迭代记录 → 未按协议执行')

if errors:
    print(f'❌ 守门失败 ({len(errors)} 个问题):')
    for e in errors: print(f'  - {e}')
    print()
    print('必须重新进行 Agent 盲识别: Skill(skill=\"image-degradation-simulator\", args=\"...\")')
    sys.exit(1)
else:
    print(f'✅ 守门通过: {len(os.listdir(phase4))} 挑战全部合格')
"
```

#### 守门规则汇总

| 检查项 | 条件 | 动作 |
|------|------|------|
| CI=0 | 脚本垃圾, 未做 Agent 分析 | ❌ 禁止训练, Agent 重做 |
| CI gap < 2 + verdict=GOOD | CI 无法可靠区分候选 | ❌ 禁止训练, 补全 CI 排名 |
| 缺少 hypothesis_ranking | 无全量 CI 对比表 | ❌ 禁止训练, Agent 重做 |
| verdict=CONFLICT | CI 胜者与证据不一致 | ⚠️ 可训练, PSNR 判定 |
| CI gap < 3 无 alternatives | 只有一个候选无备份 | ⚠️ 警告, 训练后需验证 |
| CI gap >= 3 + GOOD + evidence_consistency | 高质量识别 | ✅ 通过 |

**核心改变（vs 旧版）**:
- CI 不再作为绝对值阈值（>=7 通过）→ 改用 CI gap (胜者-亚军差距) 判断可靠性
- 新增 evidence_consistency 检查 → CI 胜者必须与初始分析一致
- 新增 alternatives 机制 → CI 接近的候选保留, 训练后 PSNR 做最终仲裁
- verdict 新增 UNCERTAIN (CI差距小) 和 CONFLICT (证据矛盾)

**来源**: exp17 教训 — 16 Agent 平均 CI=8.2/10, 函数正确率 6%. CI 衡量视觉相似性, 不衡量函数正确性.

### 🔴 训练后反思协议：Skill 子 Agent 强制执行 🔴

**来源**: exp17 R2 — 反思平均 +1.06 dB, 但对噪声假设无效 (0009 -1.41, 0014 -2.31).
**来源**: exp27 R1 — 3/7 反思有害 (-1.02, -3.51, -6.96 dB). 根因: 过度依赖 R0 模型残差，而非回归 SKILL.md 信号.
**来源**: exp28 R1 — 反思未通过 Skill 子 Agent 执行，手动调整参数，违反了隔离和可审计性原则.

#### 🔴 反思必须通过 Skill 子 Agent 执行（来源: exp28 审计）

```
❌ 禁止: 主 Agent 手动修改 predicted_params.json 后直接训练
❌ 禁止: 主 Agent 批量脚本处理多个挑战的反思
❌ 禁止: 跳过 PSNR 预验证直接生成 R1 训练管线

✅ 正确: 对每个需反思的挑战，Skill(skill="image-degradation-simulator", args="反思 blind_XXXX...")
         → Skill 加载 → 子 Agent 执行反思工作流:
         1. Read SKILL.md 确认当前协议
         2. 重新运行 run_full_analysis.py 获取完整信号报告
         3. 信号重释: 找被忽略/误判/边界的信号
         4. 生成修正假设 (注明信号来源)
         5. apply_multi.py 模拟 + PSNR 验证 (≤5 候选)
         6. 保存 R1 predicted_params.json + reflection.json (更新)
         7. 仅在 PSNR 验证通过后 → 保存 R1 训练 params
```

**为什么必须用 Skill 子 Agent**:
1. 反思面临比盲识别更大的 GT 泄露风险（已知 R0 不完美）
2. 子 Agent 强制遵守隔离规则（禁止读取 GT 文件）
3. 子 Agent 强制可审计性（每个修正必须注明信号来源）
4. 主 Agent 手动操作会跳过 PSNR 预验证 → 直接训练 → 浪费 GPU
5. exp28 教训: 主 Agent 手动反思未做 PSNR 测试, 修正方向未经验证

**反思并发**: 与盲识别相同, 按挑战数分组并行子 Agent
| 挑战数 | Agent 数 | 说明 |
|------|:--:|------|
| 1-4 | 1 | 少量挑战, 1 Agent 顺序处理 |
| 5-8 | 4 | 中等数量, 4 Agent 各处理 2 个 |
| 9-16 | 8 | 大量挑战, 8 Agent 各处理 1-2 个 |

#### 🔴 核心原则修正 (exp27 教训)

```
❌ 旧原则: "失败模型是诊断工具" — 它学会了修什么, 就说明训练管线里有什么是对的
   → exp27 证明: R0 模型训练于错误退化 → residual = target - pred 被模型错误污染
   → 从被污染的 residual 诊断 → 修正方向随机 → 3/7 有害

✅ 新原则: R0 模型 = 验证器, 不是诊断器
   1. R0 GT PSNR < 阈值 → 确认盲识别有误 → 触发 SKILL.md 信号复检
   2. 修正方向来自原始信号 (Step 1→4), 不是模型 residual
   3. 信号复检重点:
      a. 被忽略的信号 (存在但未纳入决策, 如 block_boundary 已检出但被覆盖)
      b. 被错误归因的信号 (如 mean_shift 归因于 brightness 而非 contrast, 
         盲点: variance_ratio 未检查)
      c. 阈值边界的信号 (略低于阈值的也应考虑)
      d. 遗漏的二阶/高阶统计量 (方差比、偏度、残差-强度相关性)
   4. R0 模型 residual 仅作为辅助 (如: residual 中明显 8×8 块 → 提醒检查 compression)
   5. 严禁从 R0 模型 residual 反推"缺少哪种退化" — 解耦不可靠
```

#### 🔴 反思验证方法强制规则 (exp37 审计)

```
反思阶段验证候选管线时，必须根据退化类型选择正确的验证方法：

✅ 确定性退化 (blur/compression/contrast/brightness/saturation):
   - 主判据: PSNR (>40dB = 正确)
   - test_candidate.py --mode psnr

✅ 随机退化 (含 noise, sigma >= 2):
   - 主判据: verify_signals.py (MATCH/PARTIAL/WEAK/MISMATCH)
   - test_candidate.py --mode signal
   - PSNR 对 noise 无效 (seed 失配导致像素级对比失真)

🔴 禁止:
   - 对含 noise 的候选仅用 PSNR 做判据
   - 因 PSNR < 20 就拒绝信号 MATCH 的噪声候选
   - 反思时忽略 verify_signals 的 PARTIAL/MATCH 结果

判定优先级:
   1. verify_signals MATCH + 信号改善 → 即使 PSNR 不高也要保存 R1
   2. verify_signals PARTIAL (改善) → 保存为 alternative
   3. CI 改善 (如 7/10 → 10/10) → 即使 PSNR 相似也可能是正确方向
   4. PSNR 改善 > 2dB → 仅确定性退化可靠

来源: exp37 blind_0005/0012 — R1 反思仅用 PSNR 错误地拒绝了
     所有噪声候选, 实际上噪声存在时 PSNR 是随机数。
```

#### 🔴 反思 Agent 强制隔离规则 (来源: exp18/exp19 泄露审计)

```
反思 Agent 比盲识别 Agent 面临更大的泄露风险（已知 R0 不完美，有动机寻找 GT）。
以下为反思 Agent 的额外强制约束：

1. 禁止访问的文件（即使同用户可读，Agent 也必须主动拒绝）:
   - .gt_mappings/ 目录下的任何文件
   - expN/.ground_truth/ 目录下的任何文件
   - expN/degradation_gt/ 目录下的任何文件
   - /tmp/expN_*_mapping*.json
   - expN/logs/ 中的训练日志（含 EXP_META，可能泄露训练退化信息）
   - expN/results/ 中的 GT 重评估结果

2. 反思只能基于以下数据源:
   ✅ 原始目标图像 (expN/challenges/phase4/blind_XXXX/degraded.png)
   ✅ clean 图像 (expN/challenges/phase4/blind_XXXX/clean.png)
   ✅ R0 盲识别结果 (predicted_params.json, alternatives, reflection.json, thinking_process.json)
   ✅ Skill 工具脚本 — 主要用于重新运行信号分析:
      - run_full_analysis.py (重新跑 Step 1→4, 获取完整信号报告)
      - analyze_degradation.py (单独分析特定退化信号)
      - noise_prior.py (噪声 σ 估计)
      - apply_multi.py, compare_degradation.py (PSNR 测试候选)
   ⚠️ R0 模型 checkpoint — 仅用于 GT PSNR 评估 (判断 R0 是否有误)
   ❌ 禁止: model_diagnosis.py 对 R0 模型 residual 做退化类型诊断

3. 反思 Prompt 约束:
   🔴 不能说"修正错误预测" —— 暗示了已知 R0 是错误的
   ✅ 应该说"重新审视信号分析，探索被忽略或误判的信号"
   🔴 不能说"找到正确的退化" —— 暗示了存在已知正确答案
   ✅ 应该说"提出可能更匹配目标图像退化特征的候选管线"

4. 强制可审计性 (reflection.json 必须包含):
   - 每个修正假设的 SIGNAL 来源追踪 (不是 residual 来源):
     "variance_ratio=0.72 → 方差显著降低但 mean_shift 微弱 → 假设 contrast_weaken 替代 brightness"
     "residual_intensity_corr=0.35 → 残差与强度相关 → 假设 gamma 替代 YCrCb noise"
   - 绝不能出现无法解释信号来源的修正
   - 如果 PSNR 测试发现某个候选 PSNR > 60dB：必须记录测试的函数+severity+PSNR值

5. 反思终止条件:
   - 所有信号都已重新审视且无新发现 → 标记 BEYOND_CAPABILITY
   - 连续 2 轮信号重释后 PSNR 无改善 → 终止
   - 已经测试 ≥ 15 个新候选且无显著改善 → 终止（防止变相枚举）
```

#### 触发条件

```
触发条件 (自参照，全程不对比外部模型):
  首次反思 (R0→R1):
    - verdict = POOR → 盲识别失败, 必须反思
    - verdict = UNCERTAIN → 盲识别不确定, 必须反思 (无论 Spec vs M_blind 如何)
    - verdict = LIKELY/GOOD + GT重评估 PSNR < 35 → 退化严重度可能误判, 反思
    - verdict = LIKELY/GOOD + GT重评估 PSNR >= 35 → 仅此情况跳过反思 (高置信+高PSNR)

  迭代反思 (Rn→Rn+1):
    - 上次修正带来 PSNR 提升 > 2dB → 可继续 (可能还有改善空间)
    - 提升 1-2dB 且未耗尽 alternatives → 最后尝试1次
    - 提升 < 1dB 且已 ≥ 2 轮 → 终止 (收益递减)
    - alternatives 耗尽 → 终止

  不启动反思的唯一条件:
    - 盲识别 verdict=GOOD/LIKELY + GT PSNR >= 35 (高置信 + 高PSNR, 大概率正确)
    - 所有候选 PSNR < 20 (噪声主导, 无可靠信号)
    - 已修正 ≥ 2 轮且提升 < 1dB (收益递减)

  🔴 exp37 修正: 原先 "UNCERTAIN + Spec > M_blind" 也可跳过, 但这依赖 GT 评估结果.
     修正后仅高置信 (GOOD/LIKELY + PSNR>=35) 跳过, UNCERTAIN 一律反思.
```

#### 反思工作流 (修正版)

```
主 Agent 职责 (仅触发 + 汇总, 不做具体反思):
  Step 0: GT PSNR 评估 R0 模型 → PSNR < 30? → 触发反思
  Step 1: 对每个需反思的挑战, 启动 Skill 子 Agent:
          Skill(skill="image-degradation-simulator", args="反思 exp28...blind_XXXX...")
  Step 2: 等待所有子 Agent 完成
  Step 3: 汇总反思结果, 更新 FINAL_REPORT.md

子 Agent 职责 (由 Skill 加载, 执行完整反思工作流):
  Step A: Read SKILL.md 确认当前协议
  Step B: 重新运行 run_full_analysis.py → 获取完整信号报告
          ⚠️ 重点关注: variance_ratio, residual_intensity_corr, clipped_fraction
  Step C: 信号重释 — 找出所有被忽略/误判/边界的信号
     a. 检查每个 Step 的 verdict vs signals: 信号指向 A 但 verdict 选了 B?
        典型: block_boundary>1.0 但 verdict=NO_COMPRESSION
        典型: variance_ratio≠1.0 但 verdict=BRIGHTNESS
        典型: gamma_suspect=True 但 verdict=YCrCb
     b. 检查阈值边界: 信号接近但未达阈值 → 降低阈值重检
     c. 检查遗漏统计量: 只用了一阶? 漏了方差/偏度/残差-强度相关?
  Step D: 形成新假设 (基于信号, 非 residual)
     每个假设必须注明信号来源:
     ✅ "variance_ratio=0.72 + mean_shift 微弱 → contrast_weaken, 不是 brightness"
     ❌ "R0 模型 residual 显示高频噪声 → 换 noise type" (不可审计)
  Step E: 🔴 PSNR 预验证 (强制, 不可跳过)
     apply_multi.py 模拟每个候选 → PSNR vs target
     仅当 PSNR 提升 > 2dB 或 PSNR > 35dB → 进入 Step F
     全部无改善 → BEYOND_CAPABILITY, 不生成 R1 训练 params
  Step F: 保存 R1 预测 + 训练 params
     - exp28/challenges/phase4/blind_XXXX/predicted_params_R1.json
     - exp28/degradation/blind_XXXX_params_R1.json
     - 更新 exp28/challenges/phase4/blind_XXXX/reflection.json (追加 R1 记录)
  Step G (可选): 如果信号指向漏检了一步退化
     用已识别的高置信步骤训练 specialist → 消除 → residual 重检
     ⚠️ 仅当已识别步骤信号置信度高; deterministic > random
```

**关键约束**:
- 🔴 反思必须由 Skill 子 Agent 执行, 禁止主 Agent 直接修改预测
- 🔴 每个修正假设必须通过 PSNR 预验证才能进入训练
- 反思主路径 = 信号重释, 不做模型 residual 诊断
- 如果所有信号已穷尽且 PSNR 无改善 → BEYOND_CAPABILITY
- 最多 2 轮信号重释
- 总 PSNR 测试 ≤ 15 次/轮

#### 判据分离规则

| 退化类型 | 主判据 | 禁用判据 | 原因 |
|------|------|------|------|
| blur/compression/contrast | PSNR (>40dB=正确) | — | 确定性, PSNR可靠 |
| noise 类型 | 统计匹配 (6步检查) | PSNR | 随机seed, PSNR无意义 |
| noise 严重度 | 残差 std 校准表 | PSNR | 同上 |
| **gamma vs YCrCb** | **残差-强度相关性** (同图模式) | cross_ch_corr alone | gamma 确定性变换, 残差随强度变化 |
| **contrast vs brightness** | **variance_ratio + mean_shift** (两者都查) | mean_shift alone | contrast 变方差不变均值 |

## 环境安装

```bash
# 主环境 (train.py / prepare.py)
uv sync                          # 安装所有依赖 (含 torch, webdataset, einops 等)
uv run prepare.py                # 验证安装: 裁切 DIV2K + 打包 WebDataset

# DFPIR 基线 (独立 conda 环境)
conda activate dfpir             # torch 2.5.1+cu124
# DFPIR checkpoint: resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt

# 新增依赖时
uv add <package>                 # 自动写入 pyproject.toml
# ⚠️ 不要用 uv pip install — uv run 使用项目虚拟环境，非 conda 环境
```

## Commands

```bash
uv sync                          # install dependencies
uv run prepare.py                # crop DIV2K + package WebDataset shards (one-time)
uv run prepare.py --demo         # test the degradation dataloader
uv run train.py                  # train restoration model
uv run train.py > run.log 2>&1   # training with log capture

# DFPIR all-in-one baseline (conda env: dfpir)
/home/zyli/anaconda3/envs/dfpir/bin/python resource/.../test_degradation.py --params <params.json> --gpus 0,1,...,7
```

## Architecture

详见 [program.md](program.md)。核心要点：

| File | Role | Mutable |
|------|------|---------|
| `program.md` | 项目完整规范、pipeline、盲识别协议 | Read-only |
| `prepare.py` | Data prep, WebDataset, degradation dataloader | **Read-only** |
| `train.py` | Model (`RestoreNet`), training loop, evaluation | **Agent edits** |
| `x_distortion/` | 35 退化函数 × 5 严重度 | Read-only |
| `resource/.../net/model.py` | DFPIR 大模型 (31M, CVPR'25) | Read-only |
| `resource/.../test_degradation.py` | DFPIR 特定退化测试 | Agent invokes |

### Key design decisions

- **Training degradation**: per-sample random when `PARAMS_PATH=None`. `generate_random_pipeline()` per sample (blur/noise/compression, 1-3 steps, severity 1-5).
- **Validation**: full images, not patches. Per-image evaluation due to variable resolutions.
- **Time budget**: training stops at `TIME_BUDGET`, first 10 steps excluded (compilation warmup).
- **`prepare.py` import**: use `from prepare import make_dataloader_restoration` directly.

## 盲识别挑战（Phase 4 数据隔离）⚠️

详见 [program.md](program.md) 完整协议。**盲识别必须通过 Skill + 子 Agent 执行**（见上方「子 Agent 盲识别协议」），禁止脚本直接生成。

### Phase 执行顺序 ⚠️ 最优先

```
正确: Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6
错误: Phase 1 → Phase 2 → Phase 3 → Phase 4 → ...
       train.py 会打印 pipeline 到日志 → 泄露！
```

### 防泄露规则

```bash
# ✅ 正确
bash setup_challenge.sh --exp exp7 --seed 42
bash setup_challenge.sh --exp exp7 --num-degs 2
bash setup_challenge.sh --exp exp7 --target-only

# ❌ 绝对禁止
cat .ground_truth.json
python3 -c "import json; json.load(open('.ground_truth.json'))"
读取 degradation/params.json（Phase 4 完成前）
```

### 盲识别必须用 Skill

```
✅ Skill(skill="image-degradation-simulator", args="分析 degraded.png...")
❌ 手动跑脚本 → 扫参数网格 → 挑最高
```

### 指标纪律

跨图模式下 content-dependent 指标不可用于退化 TYPE 判断。详见 program.md。

### 退化管线规则

blur、noise、compression 各最多出现一次，最多 3 步。

## 实验执行

**不预设固定实验矩阵。** 每个实验方案由 LLM 根据当前任务特点自主设计，追求有效和创新而非暴力枚举。

实验执行逻辑（保留基础设施）：

```python
# 通过环境变量覆盖 train.py 参数，灵活运行任意实验
AR_LOSS_FN=mse AR_LR_SCHEDULE=constant AR_EMBED_DIM=96 \
AR_PARAMS_PATH=params.json AR_VAL_PARAMS_PATH=params.json \
AR_EPOCH_BUDGET=1 AR_CKPT_PREFIX=expN/experiments/exp_XXX \
AR_CHALLENGE_DIR=expN/challenges/phase4/blind_XXXX \
uv run train.py
```

**🔴 AR_CHALLENGE_DIR (exp40 新增)**: 设置为挑战目录时，验证集使用挑战自带的 `clean_*.png` + `degraded_*.png` 成对图像（无需施加退化参数）。val PSNR 直接反映 GT 退化恢复质量，比 Set14 + 预测退化更可靠。
- 设置后不再使用 Set14 作为 quick val
- 每个挑战独立训练时必须设置，确保 val PSNR 监控 GT 性能
- 批量训练示例: `AR_CHALLENGE_DIR=exp40/challenges/phase4/blind_0001`

动态调度器（`expN/scripts/phase5_scheduler.py`）：维护待执行实验队列，GPU 空闲（< 15%）立即分配下一个。

### GPU 分配规则 ⚠️ 必须遵守

**每张 GPU 最多同时运行 1 个实验。** 违反会导致 OOM 崩溃。

正确做法：使用 GPU 脚本队列（每次 1 个顺序执行），或轮询分配时确保总数 ≤ 8。
分配示例：
```bash
# 每个 GPU 脚本内部串行：
CUDA_VISIBLE_DEVICES=0 exp1 ; CUDA_VISIBLE_DEVICES=0 exp2  # 顺序执行
# 不是：
CUDA_VISIBLE_DEVICES=0 exp1 & CUDA_VISIBLE_DEVICES=0 exp2 &  # 并行! OOM!
```

### DFPIR 大模型基线对比 ⚠️ Phase 6 — 全部流程结束后的最后一步

**🔴 关键原则：DFPIR 是外部模型，仅在实验全部结束后用于最终对比。全程不接触、不对比 DFPIR 或其他外部模型。反思、盲识别、架构选择均不得参考 DFPIR 结果。**

每个实验计划中，在 Phase 5 训练 + GT 重评估全部完成后，用 DFPIR (31M, CVPR'25) 做最终对比。

#### ✅ 推荐：串行多 GPU 模式（每个 task 用所有可用 GPU，逐个接力）

```bash
# 生成串行队列脚本（每个 task 用 7 GPU 并行，GPU7 留给训练）
.venv/bin/python3 << 'PYEOF'
import os
DFPIR_PY = "resource/.../test_degradation.py"
DFPIR_PYTHON = "/home/zyli/anaconda3/envs/dfpir/bin/python"
bids = sorted([...])  # 已完成训练的 challenge ID 列表

with open("expN/scripts/dfpir_serial.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for bid in bids:
        params = f"expN/degradation_gt/{bid}_params.json"
        output = f"expN/results/dfpir_{bid}.json"
        logf = f"expN/logs/dfpir_{bid}.log"
        cmd = f"{DFPIR_PYTHON} {DFPIR_PY} --params {params} --gpus 0,1,2,3,4,5,6 --output {output}"
        f.write(f'echo "[$(date +%H:%M)] {bid} START"\n')
        f.write(f'{cmd} > {logf} 2>&1\n')
        f.write(f'echo "[$(date +%H:%M)] {bid} DONE"\n')
os.chmod("expN/scripts/dfpir_serial.sh", 0o755)
PYEOF

# 启动
bash expN/scripts/dfpir_serial.sh &
```

**性能**：每退化 ~1 分钟（7 GPU 分担 737 张图），15 退化 ≈ 15 分钟。

#### 训练和 DFPIR 并行

GPU7 留给训练，GPU0-6 跑 DFPIR，互不干扰。

```bash
# 训练：GPU7 单独跑最后一个 task
# DFPIR：GPU0-6 串行多 GPU 模式（--gpus 0,1,2,3,4,5,6）
```

#### ❌ 错误方法

| 错误 | 原因 |
|------|------|
| 8 GPU 模式 + GPU 被占用 | spawn 死锁 |
| 每个 task 单独单 GPU 进程 | 每个进程重复加载全部 737 张图，I/O 冗余 |
| 修改 test_degradation.py | 外部代码，改动难追踪 |

**关键参数**：
- Checkpoint: `resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt`
- 模型: `ChannelShuffle_skip_textguaid` (31.1M 参数)
- 环境: `/home/zyli/anaconda3/envs/dfpir/bin/python` (torch 2.5.1+cu124)
- 验证集: 737 张图片，自动 tiled inference (tile=512, overlap=64)

**对比格式**：实验报告中必须包含 DFPIR PSNR 作为参考上界。

#### DFPIR-ft：同 GPU 预算 fine-tune 对比 🔴 用预测退化训练 + 梯度累积公平对齐

与 RestoreNet Specialist 在完全相同条件下对比：**同预测退化训练，同 1.5 GPU-h 预算，同 effective batch=16，GT 仅用于最终评估**。

##### 🔴 公平性设计 (exp37)

DFPIR 31M 参数量是 RestoreNet 0.46M 的 67 倍，单 GPU 只能跑 batch=4。必须用梯度累积对齐 effective batch：

| 维度 | RestoreNet | DFPIR-ft | 公平性 |
|------|:--:|:--:|:--:|
| 参数量 | 0.46M | 31.1M | DFPIR 先天优势 |
| Batch size | 16 | 4 | GPU 显存硬限制 |
| 梯度累积 | 1 | **4** | ← 对齐 |
| Effective batch | 16 | **16** | ✅ 一致 |
| GPU 预算 | 1.5h | 1.5h | ✅ 一致 |
| 优化器步数 | ~15094 | ~3500 | DFPIR 67×参数的代价 |

**关键：不比较"谁步数多"，比较"同预算下谁效果更好"**。

```bash
# 1. 训练 (8 GPU 并行, 每 GPU 3 任务串行, ~4.5h 墙钟)
# 脚本: exp35/scripts/train_dfpir_ft.py
# 关键参数: --params (预测退化, R1优先R0兜底) --time-budget 5400 --batch-size 4 --accum-steps 4
# 环境: /home/zyli/anaconda3/envs/dfpir/bin/python

# 2. GT 重评估 (用 test_degradation.py --checkpoint 加载 fine-tune 权重)
for bid in blind_*; do
  /home/zyli/anaconda3/envs/dfpir/bin/python resource/.../test_degradation.py \
    --params expN/degradation_gt/${bid}_params.json \
    --checkpoint expN/experiments/dfpir_ft_${bid}/dfpir_ft_step*.pt \
    --gpus 0,1,2,3,4,5,6,7 --output expN/results/dfpir_ft_${bid}.json
done
```

**exp35 结论**: 1.5 GPU-h fine-tune 对 31M DFPIR 几乎无效（≈ zero-shot）。同预算下 0.46M RestoreNet 从头训练效率是 DFPIR 的 68 倍。这不是 DFPIR 能力问题——如果给足够的 fine-tune 预算（如 10 GPU-h），DFPIR 应能大幅超越小模型。

**exp37 改进**: 引入梯度累积 (batch=4 + accum=4 → effective=16) 对齐优化条件，排除梯噪差异干扰。

**对比矩阵**:
| 模型 | 训练退化 | GPU 预算 | Effective Batch | GT 用于 |
|------|------|:--:|:--:|------|
| RestoreNet R0 | 预测 | 1.5h | 16 | 最终评估 |
| RestoreNet R1 | 预测 | 1.5h | 16 | 最终评估 |
| DFPIR-ft | 预测 | 1.5h | 16 | 最终评估 |
| DFPIR zero-shot | N/A | 0 | — | 最终评估 |

**效率指标**: GPU-h per dB = 训练总 GPU 小时 / (Spec PSNR - M_blind PSNR)。越低越好。```
| 退化 | Direct | Ft | Curric | DFPIR(31M) |
|------|--------|----|--------|------------|
| D1   | 22.92  | 23.04 | 21.05 | 22.19      |
```

## 目录结构与归档规则

**所有实验产物归入 `expN/`，临时脚本归入 `expN/scripts/`，禁止散落根目录。**

```
expN/
├── degradation/          ← 退化管线
├── experiments/          ← 子实验 (checkpoints + results)
├── scripts/              ← 临时脚本（phase 评估、调度器等）
├── logs/                 ← 训练日志
├── model/                ← train.py 快照
├── results/              ← results.tsv
└── summarize/            ← 总结文档 + CSV
```

根目录仅保留核心文件：`train.py`、`prepare.py`、`blind_challenge.py`、`evaluate_blind_challenge.py`、`setup_challenge.sh`。

### 🔴 实验总结报告规范 (summarize/)

每个实验完成后，`expN/summarize/` 下必须包含以下 **3 份**报告，缺一不可：

#### 1. 盲识别与反思报告 (`blindID_reflection.md`) — R0+R1 合并

```markdown
## R0 盲识别
- Agent 分配、工作流、验证模式
- 逐挑战 R0 预测 vs GT 对比 (函数+严重度+顺序)
- 准确度: 完全匹配率, 类别匹配率, 按退化步数分别统计
- 主要失败模式 (漏检/过预测/子类型混淆/假阳性)

## R1 反思
- 触发条件: 哪些挑战触发/跳过, 原因
- 逐挑战 R0→R1 修正对比:
  | Challenge | R0预测 | R1修正 | 信号来源 | 仿真PSNR Δ | 实际训练PSNR Δ | 效果 |
- 反思成功率: 有效/有害, 平均PSNR改善
- 成功案例: 信号重释如何找到被忽略/误判的信号
- 失败案例: 为什么修正方向错误

## 关键结论
- 盲识别主要瓶颈 (步数/类型/严重度)
- 反思有效模式 (擅长什么, 不擅长什么)
```

#### 2. 训练策略与资源报告 (`training_config.md`)

```markdown
- 逐挑战 R0/R1 训练配置:
  | Challenge | 策略 | 架构 | 参数量 | EPOCH | GPU时间 |
- 架构选择理由 (引用 finetune_strategy.md 决策规则)
- GPU 资源统计: 总 GPU 数, 任务分配, 墙钟时间
- 对比基线: M_blind, DFPIR zero-shot, DFPIR-ft
```

#### 3. 最终性能对比报告 (`final_performance.md`)

```markdown
- 逐挑战对比表:
  | Challenge | R0 | R1 | DFPIR | DFPIR-ft | Best | ΔvsDFPIR | Winner |
  (DIV2K 和 LSDIR 分别列出)
- Specialist vs DFPIR 胜率统计
- 按退化步数分层: 1步/2步/3步 退化的表现差异
- 训练效率: GPU-h per dB
- 关键结论 (≤5条)
```

#### 注意事项

- 所有 summarize 文件写入后立即 chmod 600 (含 GT 对比数据)
- 性能对比必须包含 DIV2K 和 LSDIR 分别列 + Delta vs DFPIR
- 盲识别对比报告合并 R0+R1 为一份 (不再分两文件)
- 架构选择理由追溯到 finetune_strategy.md 决策规则

## Phase 5 训练启动 ⚠️ 实操

### 训练前必读 🔴

启动 Phase 5 前**必须按顺序执行**：

1. **Read `finetune_strategy.md`** — 读完整文档，不是凭记忆
2. **对每个挑战，基于 predicted_params.json 的 per-step verdict，查表选择策略和架构**
3. **确认 VAL 参数指向预测管线**（非 GT）— 防泄露
4. **确认 EPOCH_BUDGET=2, LR=5e-4**

### 🔴 策略选择协议（每次盲识别后通过子 Agent 强制执行）

**策略选择必须通过子 Agent 执行，禁止主 Agent 写脚本批量处理。**

```
对每个挑战，启动子 Agent:
  Agent prompt:
    1. Read finetune_strategy.md（完整阅读，不可跳过）
    2. 读取该挑战的 predicted_params.json（per-step verdict）
    3. 基于 per-step verdict + full_analysis.json 原始信号，人工分析：
       a. 每步退化匹配 finetune_strategy.md Section 七的信号门控规则
       b. Tier S/A 信号可覆盖 UNCERTAIN verdict
       c. 考虑组件与退化的匹配度、风险、参数公平性
    4. 输出: (strategy, architecture, reasoning)
    5. 保存训练决策到 {challenge_dir}/training_strategy.json
  
  子 Agent 必须:
    - 逐个挑战分析，不批量处理
    - 注明每个决策的信号来源和 finetune_strategy.md 引用
    - 不确定时保守（纯 Swin + Direct）
```

> 🔴 禁止: 主 Agent 用 Python 脚本批量选策略（exp40 教训: 遗漏 signal-tier 门控）
> 🔴 禁止: 凭记忆选策略（必须 Read finetune_strategy.md）
> 🔴 每个挑战独立分析，不套模板

### 策略选择速查

| 退化类型 | 策略 | 架构 | LR |
|------|------|------|:--:|
| contrast 型全局 | **Direct** | Swin | 5e-4 |
| UNCERTAIN 盲识别 | Direct | **DualBranch** | 5e-4 |
| LIKELY + 全局退化 | Ft (如有ckpt) | Swin+ColorPre | 5e-4 |
| LIKELY + 局部退化 | Ft (如有ckpt) | Swin | 5e-4 |
| motion blur sev≥5 | Direct/Ft | OCAB+ws16 | 5e-4 |

**预训练 ckpt**：从历史实验复制 `M_blind` 的 `step15092.pt` → `resource/blind_pretrain/checkpoints/`。
注意：ColorPre/DualBranch 等改架构的不能 Ft（ckpt 不兼容），必须 Direct。

### 任务生成和启动

**🔴 统一使用 `scripts/exp_launcher.sh`，禁止自行编写调度器。**

```bash
# 1. 生成 JSONL 任务文件
.venv/bin/python3 << 'PYEOF'
import json
tasks = []
for bid in ["blind_0001", ...]:
    tasks.append({"id": bid, "cmd": f"CUDA_VISIBLE_DEVICES=GPU_ID AR_PARAMS_PATH=... .venv/bin/python3 train.py > expN/logs/{bid}.log 2>&1"})
with open("expN/scripts/tasks.jsonl", "w") as f:
    for t in tasks: f.write(json.dumps(t) + "\n")
PYEOF

# 2. 启动排队
bash scripts/exp_launcher.sh start expN/scripts/tasks.jsonl 0,1,2,3,4,5,6,7

# 3. 监控
bash scripts/exp_launcher.sh status
```

**排队机制**：`scripts/gpu_runner.sh` 每 GPU 一个进程，`flock` 原子取任务，串行执行。来源：exp10/exp12 验证。

**🔴 exp37 教训**：自行编写的 `queue_scheduler.sh` 使用 Python subprocess 启动训练，exit 144 无声失败，R1 实验全部未执行。禁止重复此错误。

旧方案（直接生成 bash 脚本）备选：

```bash
# 1. 用 Python 脚本生成 per-GPU 任务文件和 bash 启动脚本
.venv/bin/python3 << 'PYEOF'
import json, os

LOAD_CKPT = "resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt"
LR = "0.0005"

# 按 finetune_strategy.md 分配每个挑战的策略和架构
# 示例配置（需根据实际 verdict 调整）
configs = {
    "blind_0001": "AR_USE_DUAL_BRANCH=1",           # UNCERTAIN → DualBranch
    "blind_0005": f"AR_LOAD_CKPT={LOAD_CKPT}",      # LIKELY 局部 → Ft
    "blind_0008": "",                                 # contrast → Swin Direct
    "blind_0009": "AR_USE_COLOR_PRE=1",              # 全局 LIKELY → ColorPre
}

tasks = []
for i in range(1, 17):
    bid = f"blind_{i:04d}"
    extra = configs.get(bid, "")
    params = f"expN/degradation/{bid}_params.json"     # 预测管线 params
    ckpt = f"expN/experiments/expN_v1_{bid}"
    log = f"expN/logs/expN_v1_{bid}.log"
    env = f"AR_PARAMS_PATH={params} AR_VAL_PARAMS_PATH={params} AR_EPOCH_BUDGET=2 AR_CKPT_PREFIX={ckpt} AR_LEARNING_RATE={LR}"
    if extra: env += f" {extra}"
    cmd = f"CUDA_VISIBLE_DEVICES=GPU_ID {env} .venv/bin/python3 train.py > {log} 2>&1"
    tasks.append({"id": bid, "cmd": cmd})

# 分配任务到 GPU（每个 GPU 2 个任务，顺序执行）
gpu_queues = {i: [] for i in range(8)}
for i, t in enumerate(tasks):
    gpu_queues[i % 8].append((t['id'], t['cmd'].replace('GPU_ID', str(i % 8))))

# 生成 per-GPU bash 脚本
for gpu in range(8):
    with open(f'expN/scripts/gpu{gpu}_run.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        for task_id, cmd in gpu_queues[gpu]:
            ckpt_dir = f"expN/experiments/expN_v1_{task_id}/checkpoints"
            # 删除旧 LR=1e-3 的 ckpt
            f.write(f'if grep -sq \'"lr": 0.001\' expN/logs/expN_v1_{task_id}.log 2>/dev/null; then rm -rf {ckpt_dir}; fi\n')
            # 跳过已完成的
            f.write(f'if [ ! -f {ckpt_dir}/*.pt ] 2>/dev/null; then\n')
            f.write(f'  echo "[$(date +%H:%M)] GPU{gpu}: {task_id} START"\n')
            f.write(f'  {cmd}\n')
            f.write(f'  echo "[$(date +%H:%M)] GPU{gpu}: {task_id} DONE"\n')
            f.write(f'else\n')
            f.write(f'  echo "[$(date +%H:%M)] GPU{gpu}: {task_id} SKIP"\n')
            f.write(f'fi\n')
    os.chmod(f'expN/scripts/gpu{gpu}_run.sh', 0o755)

# 2. 并行启动所有 GPU
import subprocess
for gpu in range(8):
    subprocess.Popen(["bash", f"expN/scripts/gpu{gpu}_run.sh"],
                     stdout=open(f"expN/logs/gpu{gpu}_runner.log", "w"),
                     stderr=subprocess.STDOUT)
PYEOF

# 3. 监控进度
watch -n 30 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

### 训练故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| OOM | DualBranch+大图 | 减少 BATCH_SIZE 或换 Swin |
| ColorPre+Ft 崩溃 | ckpt 缺少 color_pre 层 | 必须用 Direct |
| 重复进程 | 旧 bash 脚本未杀 | `pkill -f train.py` 后重启 |
| LR=1e-3 | 默认值未覆盖 | 确认 `AR_LEARNING_RATE=0.0005` |
| gpu_scheduler "no pending" | `.claim.*` 锁文件残留 | `rm -f expN/scripts/phase5_tasks.jsonl.claim.*` |

⚠️ **auto_pipeline 限制**: `auto_pipeline.py` 只能处理 Phase 3 (params 导出)、Phase 5 (训练)、DFPIR 评估。**绝对禁止** auto_pipeline 做 Phase 4 盲识别——必须由 Agent 通过 Skill 执行。
（来源: exp17 — 脚本生成的盲识别 75% 函数错误，全部 CI=0/10，0 reflection 文件）

### GT 退化重评估（Phase 5 后）

⚠️ **Phase 5 Specialist 训练的 VAL_PARAMS 使用盲识别预测（不泄露 GT），但评估对比时必须以 GT 退化为准。**

🔴 **必须 8 卡并行**：16 个挑战评估不能单卡串行（~30 分钟），必须按排队机制分配到 8 张 GPU，每卡 2 个挑战并行评估（~5 分钟）。

#### 1. 生成 per-GPU 并行评估脚本

```bash
# 生成 8 GPU 并行评估脚本（每 GPU 处理 total/8 个挑战）
cat > expN/scripts/reeval_parallel.sh << 'SCRIPT'
#!/bin/bash
EXP="expN"
cd /data/zyli/projects/autoresearch
GPU=$1

export CUDA_VISIBLE_DEVICES=$GPU

.venv/bin/python3 << PYEOF
import json, os, sys, glob, torch
sys.path.insert(0, '.')
from train import evaluate_all, ValDataset, RestoreNet

EXP = 'expN'
MAPPING = f'.gt_mappings/{EXP}_mapping.json'
gt_map = json.load(open(MAPPING))
device = torch.device('cuda:0')
amp = torch.amp.autocast('cuda', dtype=torch.bfloat16)

D = ['datasets/Set5/GTmod4','datasets/Set14/GTmod4','datasets/B100/GTmod4',
     'datasets/Urban100/GTmod4','datasets/Manga109/GTmod4','datasets/DIV2K/DIV2K_valid_HR']

all_bids = sorted(gt_map.keys())
gpu_id = $GPU
my_bids = [all_bids[i] for i in range(len(all_bids)) if i % 8 == gpu_id]

results = {}
for bid in my_bids:
    # Find checkpoint (supports _R1, _v1 suffixes)
    ckpt_dir = f'{EXP}/experiments/{EXP}_v1_{bid}/checkpoints'
    if not os.path.isdir(ckpt_dir):
        # Try R1 or other variants
        for suffix in ['_R1', '']:
            alt = f'{EXP}/experiments/{EXP}{suffix}_{bid}/checkpoints'
            if os.path.isdir(alt): ckpt_dir = alt; break
    pts = sorted(glob.glob(f'{ckpt_dir}/*.pt'))
    if not pts:
        print(f'{bid}: NO CHECKPOINT')
        continue
    
    # Extract architecture from EXP_META in training log
    logf = f'{EXP}/logs/{EXP}_v1_{bid}.log'
    arch = {'attention_type': 'swin', 'window_size': 8, 'dual_branch': 0, 'color_pre': 0}
    if os.path.exists(logf):
        with open(logf, errors='ignore') as f:
            for line in f:
                if 'EXP_META:' in line:
                    try:
                        mo = json.loads(line.split('EXP_META: ')[1].split(' ===')[0])['model']
                        arch = {'attention_type': mo.get('attention_type','swin'),
                                'window_size': mo.get('window_size',8),
                                'dual_branch': int(bool(mo.get('dual_branch',0))),
                                'color_pre': int(bool(mo.get('color_pre',0)))}
                    except: pass
                    break
    
    model = RestoreNet(in_ch=3, embed_dim=64,
                       attention_type=arch['attention_type'],
                       window_size=arch['window_size'],
                       use_dual_branch=arch['dual_branch'],
                       use_color_pre=arch['color_pre']).to(device)
    ck = torch.load(pts[-1], map_location=device, weights_only=True)
    model.load_state_dict(ck['model'])
    model.eval()
    
    tmp_path = f'/tmp/{EXP}_gt_{bid}_gpu{gpu_id}.json'
    json.dump({'pipeline': gt_map[bid]['pipeline']}, open(tmp_path, 'w'))
    vs = [(ds.split('/')[-2], ValDataset([ds], count=0, params_path=tmp_path))
          for ds in D if os.path.isdir(ds)]
    _, ov = evaluate_all(model, vs, device, amp)
    os.remove(tmp_path)
    
    results[bid] = {'psnr_rgb': round(ov['psnr_rgb'],2), 'psnr_y': round(ov.get('psnr_y',0),2)}
    print(f'[{gpu_id}] {bid}: GT PSNR={ov["psnr_rgb"]:.2f} dB')

json.dump(results, open(f'{EXP}/results/gt_reeval_gpu{gpu_id}.json', 'w'), indent=2)
PYEOF
SCRIPT
chmod +x expN/scripts/reeval_parallel.sh
```

#### 2. 并行启动

```bash
# 8 GPU 并行评估
for gpu in 0 1 2 3 4 5 6 7; do
  nohup bash expN/scripts/reeval_parallel.sh $gpu > expN/logs/reeval_gpu${gpu}.log 2>&1 &
done
```

#### 3. 合并结果 + 对比分析

```bash
# 全部 GPU 完成后合并
.venv/bin/python3 << 'PYEOF'
import json, glob
EXP = 'expN'
merged = {}
for f in sorted(glob.glob(f'{EXP}/results/gt_reeval_gpu*.json')):
    merged.update(json.load(open(f)))

json.dump(merged, open(f'{EXP}/results/gt_reeval.json', 'w'), indent=2)

# 对比 M_blind 基线
m_blind = json.load(open(f'{EXP}/results/m_blind_baseline.json'))
print(f"{'Challenge':<14} {'M_blind':>8} {'Specialist':>10} {'Delta':>8}")
for bid in sorted(merged.keys()):
    mb = m_blind[bid]['psnr_rgb']
    ms = merged[bid]['psnr_rgb']
    print(f'{bid:<14} {mb:>8.2f} {ms:>10.2f} {ms-mb:>+8.2f}')
PYEOF
```

**关键参数**：
- 验证集：DIV2K_valid_HR (100张) + LSDIR val1 (250张)，分别报告
- 模型架构：从训练日志 `EXP_META` 提取（`model.attention_type`, `model.color_pre`, `model.dual_branch`）
- Checkpoint：取每个实验目录下 step 最大的 .pt 文件
- 评估退化：`.gt_mappings/expN_mapping.json` 中的 GT pipeline（仅用于 VAL，不用于训练）
- **并行策略**：8 GPU 各评估 total/8 个挑战，每 GPU ~5 分钟（vs 单卡 ~30 分钟）

## The experiment loop

1. Read `train.py` for full context
2. Design an experimental idea (LLM-driven, not grid search)
3. Modify `train.py`
4. `git commit -m "experiment: <description>"`
5. `uv run train.py > run.log 2>&1`
6. `grep "^val_psnr_db:" run.log`
7. If crash → fix if trivial, else log and move on
8. If improved → keep commit. If not → `git reset --hard HEAD~1`

**Never stop**: Do not ask "should I keep going?". Run indefinitely until interrupted.
