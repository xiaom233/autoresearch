# E4 执行说明：仅 R0，无训练后 R1/R2 反思

日期：2026-10-26。适用于 `E4/sidd` 与 `E4/uhdm`。

## 实际执行范围

- 每个任务只预登记 **1 组 support**（SIDD 5 对、UHDM 5 对），只训练 **1 次 R0 specialist**：
  - SIDD Skill：`skill_R0`，best step 7546，2 epoch / 15,094 steps。
  - SIDD 规则：`rule_R0`，best step 3773，2 epoch / 15,094 steps。
  - UHDM 规则 fallback：`rule_R0`，best step 11319，2 epoch / 15,094 steps。
- 没有生成任何 R1/R2 训练任务、params、checkpoint 或 `predicted_params_R1.json`。
- 任务文件只有 R0：
  - `rebuttal_runs/E4/sidd/tasks_skill.jsonl`
  - `rebuttal_runs/E4/sidd/tasks_rule.jsonl`
  - `rebuttal_runs/E4/uhdm/tasks_rule.jsonl`
- 训练日志只有 `skill_R0.log` / `rule_R0.log`，三者均正常跑到 15,094 steps 并保存 checkpoint；无 NaN、无异常退出。

## “反思”到底有没有发生

必须区分两层：

1. **训练前候选搜索反思**：发生了。
   - `E4/sidd/skill/reflection.json`：15 个候选迭代（`iterations` 长度 15），最终 `overall_verdict = UNCERTAIN`，备选 `noise_gaussian_RGB:2`、`noise_poisson:1`。
   - `E4/uhdm/skill/reflection.json`：7 个候选迭代，最终 `overall_verdict = POOR`、`out_of_library = true`。
   - 这是 Skill 在 R0 训练前对 support 信号做的候选模拟/验证，不是对 R0 训练结果的 R1 反思。

2. **训练后 R1/R2 反思**：没有发生。
   - 没有 R1 任务、R1 日志、R1 predicted_params、R1 checkpoint 或 R1 reflection 记录。
   - 没有任何“停止反思/NO_CHANGE/BEYOND_CAPABILITY”的决策文件。
   - 因此不能声称 agent 主动判断“不需要反思”；按项目 `CLAUDE.md` 的触发条件，SIDD `UNCERTAIN`、UHDM `POOR` 本应触发 R1。

## 原因判断

- **不是运行崩溃**：R0 训练全部正常完成，日志和 checkpoint 完整；没有错误堆栈或异常退出。
- **不是 agent 基于 R0 结果主动判定“不需要反思”**：没有这样的记录；Skill 的最终 verdict 反而是会触发反思的 `UNCERTAIN`/`POOR`。
- **是执行范围/流程遗漏**：E4 按任务卡的最小真实实验排了“单组 support + 单轮 R0”，没有把训练后反思 R1/R2 排入任务队列。更准确地说，是“反思流程没有被调度”，而不是“agent 评估后决定跳过”。

## 对结论的影响

- E4 的所有数值应标为 **R0-only specialist / direct-use** 结果，不是完整 R0–R2 系统结果。
- 对 SIDD：Skill 的最终 verdict 是 UNCERTAIN，且备选仍存在；未做 R1 意味着 E4 没有测试“反思是否能把真实噪声 specialist 调得更好”。
- 对 UHDM：Skill 判定 out-of-library/POOR；即使做 R1，按协议也可能以 BEYOND_CAPABILITY 终止，但该终止记录同样不存在；因此不能把“未反思”解释成“已确认无候选”。
- 后续若要补 E4 反思：应仅用 support 做 R1 决策，重新训练 R1 specialist，再在独立 test 上评估；不能在 test 上选轮次。
