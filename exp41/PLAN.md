# exp41 多轮反思实验

## 目标

在盲识别 R0→R1 之后继续 R2→R3→... 多轮反思，观察反思机制在多少轮后开始失效，产生震荡。

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
❌ 错误: Agent(description="R2 反思", subagent_type="general-purpose", prompt="...")
   → Agent 共享主会话上下文，能读到盲识别准确率讨论记录

✅ 正确: Agent(description="R2 反思", subagent_type="general-purpose",
              isolation="worktree", prompt="...")
   → Agent 在独立 worktree 中运行，上下文干净，只能看到磁盘文件
```

## 执行步骤

### Phase 1: 启动 R2 反思（24 挑战，8 Agent × 3 挑战）

```bash
# 在新会话中执行以下 Agent 调用，8 个并行

Agent A: isolation="worktree", prompt="执行 exp41 多轮反思 R2，处理 blind_0001, blind_0002, blind_0003。先 Read SKILL.md，然后每个挑战：(1) Read R0 predicted_params.json + reflection.json (2) Read R1 predicted_params_R1.json (3) 重跑 run_full_analysis.py (4) 信号重释：比较 R0→R1 历史，找仍被忽略/误判的信号，不重复失败方向 (5) test_candidate.py PSNR/信号预验证 (6) 保存 predicted_params_R2.json + 更新 reflection.json"

Agent B: 同上，处理 blind_0004-0006
Agent C: 同上，处理 blind_0007-0009
Agent D: 同上，处理 blind_0010-0012
Agent E: 同上，处理 blind_0013-0015
Agent F: 同上，处理 blind_0016-0018
Agent G: 同上，处理 blind_0019-0021
Agent H: 同上，处理 blind_0022-0024
```

### Phase 2: 训练 R2

```
# R2 params 全部保存后，8 GPU 并行训练
# 对每个 challenge: AR_PARAMS_PATH=exp41/degradation/blind_XXXX_params_R2.json ...
# EPOCH_BUDGET=2, LR=5e-4, batch=16
```

### Phase 3: GT 重评估 R2

```
# 对比 R0 vs R1 vs R2 GT PSNR
# 记录每个 challenge 的 Δ(R2-R1)
```

### Phase 4: R3 反思（条件触发）

```
for each challenge:
    if R2_PSNR - R1_PSNR > 0.5 dB:
        启动 R3 反思 (isolation="worktree")
    elif R2_PSNR < R1_PSNR < R0_PSNR:
        震荡检测 → 标记 OSCILLATION, 取最优轮次
    else:
        收敛 → 结束
```

### 终止条件

| 条件 | 动作 |
|------|------|
| Δ < 0.5 dB vs 上轮 | 收敛，终止 |
| 连续 2 轮无信号新发现 | BEYOND_CAPABILITY |
| R_n < R_{n-1} 且 R_{n-2} > R_{n-1} | 震荡，取最优 |
| 达到第 5 轮 | 强制终止 |

## Agent Prompt 模板

```
执行 exp41 多轮反思 R{N}，处理 blind_{A}, blind_{B}, blind_{C}。

**必须先 Read SKILL.md 确认当前协议**，然后对每个挑战执行反思工作流：

1. Read R0 predicted_params.json + reflection.json
2. Read 所有之前轮次的 predicted_params_R*.json（在 exp41/challenges/phase4/<bid>/ 下）
3. 重跑 run_full_analysis.py --target <degraded_0.png> --clean <clean_0.png>
4. 信号重释：比较 R0→R1→...→R{N-1} 完整修正历史
   - 找仍被忽略/误判/边界的信号
   - 绝对不能重复之前试过且失败的修正方向
   - 如果某轮修正退化了，分析原因
5. 生成 ≤5 个新修正假设（每个注明信号来源，非 residual 来源）
6. test_candidate.py PSNR/信号预验证
7. 保存 predicted_params_R{N}.json + 更新 reflection.json

关键规则：
- R{N-1} 退化于更早轮次 → 以最优轮次为起点
- PSNR 改善 > 2dB 或 verify_signals 改善 (WEAK→PARTIAL→MATCH) 才保存
- 噪声存在时禁止仅用 PSNR 拒绝候选
- ≤ 15 个新候选测试
- 所有候选都不改善 → BEYOND_CAPABILITY，不保存 R{N} params
```

## 最终输出

`exp41/summarize/reflection_multiround.md`:

```markdown
| Challenge | R0 | R1 | R2 | R3 | R4 | R5 | 最优轮 | 震荡? |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| blind_0001 | x | x | x | - | - | - | 1 | N |
...

## 关键图表
1. 每轮平均 PSNR 改善量 (ΔR1, ΔR2, ΔR3...)
2. 收敛轮次分布 (1轮/2轮/3轮/4轮/5轮)
3. 震荡比例 (R_n < R_{n-1} 的挑战数)
4. 按退化步数分层的收敛速度 (1步/2步/3步)
```
