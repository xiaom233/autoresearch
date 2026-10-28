# exp41 多轮反思实验

## 目标

在盲识别 R0→R1 之后继续下一轮→再下一轮...，观察反思机制在多少轮后开始失效，产生震荡。

## 目录结构（已就绪）

```
exp41/
├── challenges/phase4/blind_XXXX/    # 24 挑战 (copied from exp40)
│   ├── degraded_0~4.png, clean_0~4.png
│   ├── predicted_params.json       # R0 盲识别
│   ├── predicted_params_R1.json    # R1 反思 (22/24)
│   └── reflection.json             # R0+R1 历史
├── degradation_gt/                  # GT 退化参数 (训练+评估用, Agent 禁止访问)
└── random_degradation/              # Random 基线
```

## 🔴 核心要求：worktree 隔离

**主会话已讨论过盲识别准确率（含退化类型信息），必须用 `isolation: "worktree"` 避免 GT 泄露。**

```
❌ 错误: Agent(description="下一轮反思", subagent_type="general-purpose", prompt="...")
   → Agent 共享主会话上下文，能读到盲识别准确率讨论记录

✅ 正确: Agent(description="下一轮反思", subagent_type="general-purpose",
              isolation="worktree", prompt="...")
   → Agent 在独立 worktree 中运行，上下文干净，只能看到磁盘文件
```

## 执行流程（循环，直到收敛）

### Step 1: 启动当前轮反思（8 Agent 并行 🔴）

**🔴 关键：所有 Agent 必须在一条消息中并行启动（单次 8 个 Agent 工具调用），不分批顺序执行。**
每个 Agent 独立在 worktree 中运行，互不依赖，并行可节省 4-6 倍墙钟时间。

```
# 正确: 一条消息中同时启动 8 个 Agent
Agent A: isolation="worktree", prompt="...blind_0001-0003..."
Agent B: isolation="worktree", prompt="...blind_0004-0006..."
Agent C: isolation="worktree", prompt="...blind_0007,0009..."  (跳过 0008)
Agent D: isolation="worktree", prompt="...blind_0010-0012..."
Agent E: isolation="worktree", prompt="...blind_0013-0015..."
Agent F: isolation="worktree", prompt="...blind_0016-0018..."
Agent G: isolation="worktree", prompt="...blind_0019-0021..."
Agent H: isolation="worktree", prompt="...blind_0022,0023..."  (跳过 0024)

# ❌ 错误: 逐个启动 Agent A → 等待完成 → Agent B → ...
#   浪费 8× 墙钟时间
```

**Agent 分组**:

| Agent | 挑战 | 备注 |
|------|------|------|
| A | blind_0001-0003 | |
| B | blind_0004-0006 | |
| C | blind_0007, 0009 | blind_0008 跳过 (PSNR=Inf 完美匹配) |
| D | blind_0010-0012 | |
| E | blind_0013-0015 | |
| F | blind_0016-0018 | |
| G | blind_0019-0021 | |
| H | blind_0022, 0023 | blind_0024 跳过 (BEYOND_CAPABILITY) |

**跳过条件**（需在每轮开始前检查）:
- PSNR=Inf（像素级完美匹配）→ 永久跳过
- 上一轮 BEYOND_CAPABILITY + 已测试 ≥ 20 候选 → 永久跳过
- 上一轮 PSNR >= 40 + LIKELY → 可选跳过（由人工判断）

**Agent prompt 模板**:

```
执行 exp41 多轮反思。此前已完成 R0→R1→...→R{N-1} 反思，当前轮为 R{N}。

处理的挑战: blind_{X}, blind_{Y}, blind_{Z}

先 Read SKILL.md，对每个挑战:
1. Read 所有已存在的 predicted_params*.json + reflection.json
2. 重跑 run_full_analysis.py
3. 信号重释: 比较全部修正历史，找仍被忽略/误判的信号
   - 绝对不能重复之前失败的方向
   - 如果之前某轮退化于更早轮，以最优轮为起点
   - 重点关注: glass_score 边界值 (~0.25)、peeled_noise_check extreme_pct (impulse 漏检)
4. test_candidate.py PSNR/信号预验证
5. 通过验证则保存 predicted_params_R{N}.json + 更新 reflection.json

规则: PSNR>2dB 或信号改善才保存; 含noise禁用纯PSNR; ≤15候选;
     无改善→BEYOND_CAPABILITY，不保存
```

### Step 2: 训练当前轮

```
# 新 params 全部保存后，8 GPU 并行训练
# AR_PARAMS_PATH=exp41/degradation/blind_XXXX_params_R{N}.json
# EPOCH_BUDGET=2, LR=5e-4, batch=16
```

### Step 3: GT 重评估

```
# 对比 R0 vs R1 vs R2 vs ... vs 当前轮 GT PSNR
# 记录每个 challenge 的 Δ(当前轮 - 上一轮)
```

### Step 4: 下一轮触发判定

```
for each challenge:
    if 当前轮_PSNR - 上一轮_PSNR > 0.5 dB:
        进入下一轮
    elif 连续两轮下降:
        标记 OSCILLATION, 取最优轮次，停止
    elif 当前轮保存了 params 但 PSNR 改善 ≤ 0.5 dB:
        收敛，停止
    else:
        BEYOND_CAPABILITY, 停止
```

### 终止条件

| 条件 | 动作 |
|------|------|
| Δ < 0.5 dB vs 上轮 | 收敛，终止 |
| 连续 2 轮无信号新发现 | BEYOND_CAPABILITY |
| PSNR 连续下降且回不到历史最优 | 震荡，取最优轮 |
| 当前轮无任何 challenge 保存了 params | 全局终止 |
| 达到第 5 轮（累计，含 R0/R1） | 强制终止 |

## Agent Prompt 模板

```
执行 exp41 多轮反思。此前已完成 R0→R1→...→R{N-1} 反思。

处理以下 3 个挑战: blind_{A}, blind_{B}, blind_{C}

**必须先 Read SKILL.md 确认当前协议**，然后对每个挑战执行:

1. Read 全部已有 predicted_params*.json + reflection.json（了解完整修正历史）
2. 重跑 run_full_analysis.py --target <degraded_0.png> --clean <clean_0.png>
3. 信号重释:
   - 比较全部历史轮次的修正轨迹
   - 找仍被忽略/误判/边界的信号
   - 绝对不能重复历史上试过且失败的修正方向
   - 如果某轮修正退化于更早轮次，分析原因，以最优轮为起点
4. 生成 ≤5 个新修正假设（每个注明信号来源）
5. test_candidate.py PSNR/信号预验证
6. 保存 predicted_params_R{N}.json + 更新 reflection.json

关键规则:
- 基于 R0→...→R{N-1} 全部历史，当前轮为 R{N}
- PSNR 改善 > 2dB 或 verify_signals 改善才保存
- 噪声存在时禁止仅用 PSNR 拒绝候选
- ≤ 15 个新候选测试
- 所有候选都不改善 → BEYOND_CAPABILITY，不保存
```

## 最终输出

`exp41/summarize/reflection_multiround.md`:

```markdown
| Challenge | R0 | R1 | R2 | R3 | R4 | R5 | 最优轮 | 震荡? | 终止原因 |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|
| blind_0001 | 25.9 | 25.9 | 25.9 | - | - | - | R1 | N | 收敛 |
| ...

## 关键图表
1. 每轮平均 PSNR 改善量 (R0→R1, R1→R2, R2→R3...)
2. 收敛轮次分布 (不含 R0)
3. 震荡比例 (PSNR 上升后连续下降的挑战数)
4. 按退化步数分层的各轮 PSNR 均值
```
