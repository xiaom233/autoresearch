# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

**语言偏好：优先使用中文回答。所有与用户的沟通、代码注释、commit 信息均使用中文。**

## ⚠️ 关键规则

0. **Phase 切换时重读文档**：每个 Phase 开始前，必须重新阅读对应的协议文档：
   - Phase 4 (盲识别): 重读 `SKILL.md` + `CLAUDE.md` §子 Agent 盲识别协议
   - Phase 5 (训练): 重读 `CLAUDE.md` §GPU 并行调度 + `finetune_strategy.md`
   - Phase 6 (DFPIR 对比): 重读 `CLAUDE.md` §DFPIR 基线评估 (仅在全部流程结束后)
   - GT 重评估: 重读 `CLAUDE.md` §GT 退化重评估
   - 反思: 重读 `REFLECTION_MECHANISM.md`
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

### 🔴 训练后反思协议：用失败模型诊断盲识别错误

**来源**: exp17 R2 — 反思平均 +1.06 dB, 但对噪声假设无效 (0009 -1.41, 0014 -2.31).
核心发现: 确定性退化和噪声必须用不同判据, 且失败模型本身就是最好的诊断工具.

**来源**: exp18/exp19 反思泄露 — 反思 Agent 直接读取 `/tmp/expN_p4_mapping.json` 获取 GT。反思必须有比盲识别更严格的约束。

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
   ✅ 失败模型 checkpoint (expN/experiments/...)
   ✅ clean 图像 (expN/challenges/phase4/blind_XXXX/clean.png)
   ✅ R0 盲识别结果 (predicted_params.json, alternatives, reflection.json, thinking_process.json)
   ✅ Skill 工具脚本 (analyze_degradation.py, noise_prior.py, model_diagnosis.py, global_degradation_analyzer.py, apply_multi.py, compare_degradation.py)

3. 反思 Prompt 约束:
   🔴 不能说"修正错误预测" —— 暗示了已知 R0 是错误的
   ✅ 应该说"分析模型行为，基于残差诊断提出改进假设"
   🔴 不能说"找到正确的退化" —— 暗示了存在已知正确答案
   ✅ 应该说"提出可能更匹配目标图像退化特征的候选管线"

4. 强制可审计性 (reflection.json 必须包含):
   - 每个修正假设的来源追踪：
     "残差中检测到 8×8 块效应 → 假设漏检 compression_jpeg"
     "noise_prior 小波 HH 子带 σ 估计与 noise_impulse 校准表匹配 → 假设噪声类型为 impulse"
   - 绝不能出现无法解释来源的"完美匹配"修正
   - 如果 PSNR 测试发现某个候选 PSNR > 60dB：必须记录测试的函数+severity+PSNR值

5. 反思终止条件:
   - 修正假设无法在 alternatives 或残差诊断中找到支撑 → 标记 BEYOND_CAPABILITY
   - 连续 2 轮修正后 PSNR 无改善 → 终止
   - 已经测试 ≥ 20 个新候选且无显著改善 → 终止（防止变相枚举）
```

#### 核心原则

```
1. PSNR 对确定性退化可靠, 对噪声完全不可靠
2. 失败模型是诊断工具: 它学会了修什么, 就说明训练管线里有什么是对的
3. 残差分析区分"确定性部分错" vs "噪声部分错"
```

#### 触发条件

```
触发条件 (自参照，全程不对比外部模型):
  首次反思 (R0→R1):
    - verdict = POOR → 盲识别失败, 必须反思
    - verdict = UNCERTAIN → 盲识别不确定, 优先反思
    - verdict = LIKELY/GOOD + GT重评估 PSNR < 35 → 退化严重度可能误判, 反思
    - verdict = LIKELY/GOOD + GT重评估 PSNR >= 35 → 跳过反思 (高置信+高PSNR)

  迭代反思 (Rn→Rn+1):
    - 上次修正带来 PSNR 提升 > 2dB → 可继续 (可能还有改善空间)
    - 提升 1-2dB 且未耗尽 alternatives → 最后尝试1次
    - 提升 < 1dB 且已 ≥ 2 轮 → 终止 (收益递减)
    - alternatives 耗尽 → 终止

  不启动反思:
    - 盲识别 verdict=GOOD/LIKELY + GT PSNR >= 35 (高置信 + 高PSNR, 大概率正确)
    - 所有候选 PSNR < 20 (噪声主导, 无可靠信号)
    - 已修正 ≥ 3 轮且提升 < 1dB (收益递减)
    - 退化超出小模型容量 (即使正确识别也难以恢复)
```

#### 反思工作流

**单步退化反思**（步数 < 2）:
```
Step 1: 加载失败模型 → clean推理 → 残差诊断
Step 2: 仅以下情况修正:
        - 残差明确指向某类错误 (8×8→漏JPEG, 高频→noise类型错)
        - alternatives 中有跨函数族候选
Step 3: 测试 ≤5 个 alternatives swap, PSNR gap > 3dB → 替换
Step 4: 全部无改善 → BEYOND_CAPABILITY, 结束
最多 1 轮, 不发明新候选, 不反复迭代
```

**双步退化反思**（步数 ≥ 2 — 训练驱动两阶段）:

来源: exp21 — 残差诊断无法解耦双步错误, 需要先消除一个退化再检测另一个。

```
前提: R0 verdict = POOR/UNCERTAIN, R0 步数 ≥ 2

Stage 1: 用最自信的那一步训练 specialist, 消除该退化
  1a. 从 R0 管线中选择置信度最高的 1 步 (优先级: compression > global > blur > noise)
      或从逐级剥离检测结果中选择信号最强的退化
  1b. Train(model_A, 仅这一步退化, EPOCH=2)
  1c. model_A(target) → pred_A  # specialist 修复了退化A
  1d. residual_A = target - pred_A  # 消除退化A后的残差

Stage 2: 在 residual_A 上重跑检测, 暴露第二步
  2a. 将 pred_A 视为"部分修复的目标", run_full_analysis(pred_A, clean)
      或直接在 residual_A 上分析:
      - 8×8 块效应 → 漏了 compression_jpeg
      - 高频随机 → 漏了 noise
      - 边缘残留模糊 → blur 类型/severity 不对
      - 全局色偏/亮度 → 漏了 global
  2b. 选出候选 B → Train(model_AB, A+B, EPOCH=2)
  2c. GT 重评估 → PSNR 对比 R0:
      提升 > 2dB → 修正成功 ✅
      提升 < 1dB → 换 B 的候选, 最多 3 次尝试
      全部失败 → BEYOND_CAPABILITY

Stage 3 (可选): 如果 B 也正确识别了, 可以反向修正 A
  3a. 类似 Stage 2, 但用 model_AB 消除 B, 在 residual_B 上重检 A
  3b. 如果 A 修正 → 最终管线 ← 交叉验证

特殊情况: R0 只识别了 1 步 (漏检)
  → 那 1 步就是 A, Stage 1 照常
  → Stage 2 在 residual_A 上找缺失的 B
```

**关键约束**:
- Stage 1 选择的 A 必须是逐级剥离或信号检测给出高置信度的退化
- 如果 R0 所有步骤置信度都低 → 跳过本流程, BEYOND_CAPABILITY
- 每阶段额外训练最多触发 2 次 (Stage 1 + Stage 2)
- 总 PSNR 测试 ≤ 10 次/阶段

#### 判据分离规则

| 退化类型 | 主判据 | 禁用判据 | 原因 |
|------|------|------|------|
| blur/compression/contrast | PSNR (>40dB=正确) | — | 确定性, PSNR可靠 |
| noise 类型 | 统计匹配 (6步检查) | PSNR | 随机seed, PSNR无意义 |
| noise 严重度 | 残差 std 校准表 | PSNR | 同上 |

#### 不启动反思的情况

- 所有候选 PSNR < 20dB (噪声主导, 无可靠信号)
- PSNR gap < 3dB 且无统计差异 (参数简并, 无法区分)
- 退化超出小模型容量 (即使正确识别也难以恢复)
- verdict=GOOD/LIKELY + GT重评估 PSNR >= 35 (高置信 + 高PSNR)

**来源**: exp17 R2 — 0001(确定性修正 +0.56), 0016(去噪修正 +7.41) 成功;
0009/0014(噪声假设用PSNR调整) 退步 -1.41/-2.31.

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
uv run train.py
```

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

#### ✅ 正确方法：8 GPU 串行模式（推荐）

```bash
# 1. 创建队列（每个退化用全部 8 GPU，串行执行）
> dfpir_queue.txt
for f in expN/degradation/*.json; do
    name=$(basename $f .json)
    [[ "$name" =~ _R[12]$|_fixed$|challenge_ids ]] && continue
    echo "/home/zyli/anaconda3/envs/dfpir/bin/python \
      resource/.../test_degradation.py --params $f --gpus 0,1,2,3,4,5,6,7 \
      --output expN/results/dfpir_${name}.json \
      > expN/logs/dfpir_${name}.log 2>&1|DFP_${name}" >> dfpir_queue.txt
done

# 2. 启动专用 runner（串行取任务，每个任务自动用 8 GPU）
bash exp12/scripts/dfpir_runner.sh dfpir_queue.txt &
```

**前提条件**：⚠️ **全部 8 GPU 必须完全空闲**。spawn 在有任何 CUDA 进程时死锁。

**性能**：每退化 ~4.2 分钟，56 退化 ≈ 4 小时。

#### ✅ 备选：单 GPU direct 模式（可与训练并行）

```bash
# 队列中每个任务用单 GPU（--gpus 0 + CUDA_VISIBLE_DEVICES），每个任务前加 sleep 10 错峰
echo "sleep 10 && CUDA_VISIBLE_DEVICES=GPU_ID ... --gpus 0 ..." >> dfpir_queue.txt
# 启动 8 个 runner
for gpu in 0..7; do bash exp10/scripts/gpu_runner.sh $gpu dfpir_queue.txt & done
```

#### ❌ 错误方法

| 错误 | 原因 |
|------|------|
| 8 GPU 模式 + GPU 被占用 | spawn 死锁，进程永久挂起 |
| 8 GPU 模式 + `CUDA_VISIBLE_DEVICES` | spawn 需要看到全部 GPU |
| 单 GPU 模式 + 无 `sleep` 错峰 | 8 进程同时预加载 → CPU 饱和 → GPU 饿死 |
| `fork` 替代 `spawn` | CUDA 拒绝：`Cannot re-initialize CUDA in forked subprocess` |

#### 优化记录

- `test_degradation.py` worker：`ThreadPoolExecutor(2)` 多线程预加载 + DataLoader pin_memory
- 单 GPU 模式（`n_gpus==1`）direct 调用 worker，不走 spawn
- `train.py` evaluate() 同样加入预加载 + DataLoader

**关键参数**：
- Checkpoint: `resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt`
- 模型: `ChannelShuffle_skip_textguaid` (31.1M 参数)
- 环境: `/home/zyli/anaconda3/envs/dfpir/bin/python` (torch 2.5.1+cu124)
- 验证集: 737 张图片，自动 tiled inference (tile=512, overlap=64)

**对比格式**：实验报告中必须包含 DFPIR PSNR 作为参考上界。

```
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

## GPU 并行调度

基于 Python 原子锁调度器，替代旧的 bash gpu_runner.sh（flock 竞态问题）。

### 核心工具

| 工具 | 路径 | 用途 |
|------|------|------|
| GPU 调度器 | `scripts/gpu_scheduler.py` | 通用任务生成 + 多 GPU 并行执行 |
| DFPIR 串行 | `expN/scripts/dfpir_serial.sh` | DFPIR 8 GPU 串行模式 |

### 完整实验流程

```bash
# === Phase 5: Specialist 训练 ===

# 1. 生成任务文件 (自动跳过已完成, 导出 params)
.venv/bin/python3 scripts/gpu_scheduler.py gen \
  --exp exp18 \
  --task-file exp18/scripts/phase5_tasks.jsonl \
  --ckpt-prefix exp18/experiments/exp18_v2 \
  --epoch-budget 2
  # 可选: --extra-env "AR_ATTENTION_TYPE=ocab AR_WINDOW_SIZE=16 AR_USE_COLOR_PRE=1"

# 2. 启动 8 GPU worker 并行执行
.venv/bin/python3 scripts/gpu_scheduler.py run \
  --task-file exp18/scripts/phase5_tasks.jsonl \
  --gpus 0,1,2,3,4,5,6,7

# 3. 查看状态
.venv/bin/python3 scripts/gpu_scheduler.py status \
  --task-file exp18/scripts/phase5_tasks.jsonl

# 4. 停止 (软停: 完成当前任务后停)
touch /tmp/stop_exp18 && .venv/bin/python3 scripts/gpu_scheduler.py run \
  --task-file exp18/scripts/phase5_tasks.jsonl --stop-file /tmp/stop_exp18

# 5. 紧急停止
pkill -f "gpu_scheduler.py"
```

### 通用任务格式 (JSON Lines)

每行一个任务, 支持任意命令:
```jsonl
{"id": "blind_0001", "cmd": "CUDA_VISIBLE_DEVICES=GPU_ID AR_... .venv/bin/python3 train.py > log 2>&1", "status": "pending", "claimed_by": null}
```

`GPU_ID` 占位符由 scheduler 自动替换为实际 GPU 编号。

### 关键设计

| 特性 | 实现 |
|------|------|
| **原子取任务** | `O_CREAT|O_EXCL` 创建 .claim 锁文件, 无竞态 |
| **GPU 绑定** | `GPU_ID` 占位符自动替换 |
| **断点续跑** | `gen` 命令检查 checkpoint + val_psnr_db, 自动跳过 |
| **幂等** | 同一队列多次运行不重复执行已完成任务 |
| **状态追踪** | pending → claimed → running → done/failed |
| **通用性** | 支持任意命令, 不限于 train.py |

⚠️ **auto_pipeline 限制**: `auto_pipeline.py` 只能处理 Phase 3 (params 导出)、Phase 5 (训练)、DFPIR 评估。**绝对禁止** auto_pipeline 做 Phase 4 盲识别——必须由 Agent 通过 Skill 执行。
（来源: exp17 — 脚本生成的盲识别 75% 函数错误，全部 CI=0/10，0 reflection 文件）

### GT 退化重评估（Phase 5 后）

⚠️ **Phase 5 Specialist 训练的 VAL_PARAMS 使用盲识别预测（不泄露 GT），但评估对比时必须以 GT 退化为准。**

```bash
# === 重评估所有 specialist checkpoint（用 GT 退化） ===

# 1. 创建 GT 退化 params（从 mapping 文件提取，仅用于 VAL）
python3 -c "
import json, os
m = json.load(open('/tmp/expN_p4_mapping.json'))
os.makedirs('expN/degradation_gt', exist_ok=True)
for bid, info in m.items():
    json.dump({'pipeline': info['pipeline']}, open(f'expN/degradation_gt/{bid}_params.json', 'w'))
"

# 2. 评估脚本: expN/scripts/reeval_with_gt.py
#   - 遍历所有 checkpoint（含 _v2 等变体）
#   - 从 EXP_META 日志提取模型架构（attention_type, use_color_pre 等）
#   - 用 GT params 构建 ValDataset，调用 evaluate_all()
#   - 输出: expN/results/reeval_gt.json（每个 checkpoint 的 PSNR_RGB/Y + SSIM）
#   - 验证集: Set5, Set14, B100, Urban100, Manga109, DIV2K_valid_HR

# 3. 训练完成后立即执行 GT 重评估 (不等待 DFPIR)
echo "CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 expN/scripts/reeval_with_gt.py > expN/logs/reeval_gt.log 2>&1|REEVAL_GT" >> expN/scripts/reeval_tasks.txt
# GT 重评估独立运行，不依赖 DFPIR 完成。DFPIR 在所有流程结束后单独执行。
```

**评估脚本关键参数**：
- 模型架构：从训练日志 `EXP_META` 提取 `model.attention_type`, `model.color_pre`, `model.pcp` 等
- Checkpoint：取 `experiments/expN_{bid}{suffix}/checkpoints/` 下 step 最大的 .pt 文件
- 评估退化：`/tmp/expN_p4_mapping.json` 中的 GT pipeline（仅用于 VAL，不用于训练）
- 验证集：与 train.py 一致的 6 个标准 benchmark

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
