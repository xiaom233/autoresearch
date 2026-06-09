# IRAgent 工作流程规范

## 核心原则

1. **Agent 绝不接触 GT**。`.ground_truth/` 和 `degradation/` 目录全程不可读
2. **VAL_PARAMS_PATH 由 launcher 锁定**。反思修正只改 PARAMS，VAL 永远指向 GT
3. **盲识别用同图模式**。`blind_challenge.py --same-image`，clean 为 degraded 原图
4. **不用 rebase**。只用 merge/push。实验数据和 checkpoint 只读

## 子 Agent 处理策略

### 何时用子 Agent

| 任务类型 | 策略 | 并行数 |
|----------|------|:--:|
| 盲识别（同图模式） | 每 6 退化 1 个 Agent | 4 并行 |
| 单退化报告 | 每 7-8 退化 1 个 Agent | 8 并行 |
| 汇总报告 | 1 个 Agent | 1 |
| 反思分析 | 每 4 退化 1 个 Agent | 4-6 并行 |

### 子 Agent 隔离

- 子 Agent 不接触 `.ground_truth/`、`degradation/`
- 只通过 Skill 脚本分析图像
- 只写入 `predicted_params/` 目录

### 避免的陷阱

| 陷阱 | 后果 | 预防 |
|------|------|------|
| 挑战生成后立即重生成 | Agent 分析结果作废 | 生成前确认参数，一次性完成 |
| 跨图模式用于盲识别 | 准确率极低 (12%) | 始终用 `--same-image` |
| 暴力枚举假设 | 违反 Skill 原则，质量差 | LLM 视觉推理 + 迭代 |
| 子 Agent 结果未检查就使用 | CI 格式不一致 | 完成后方差检查 |

## 盲识别流程

### 挑战生成（一次性）

```bash
# 同图模式 — Agent 不接触 GT
for seed in ...; do
    uv run blind_challenge.py --seed $seed --num-degs N --same-image \
        --output-dir "expN/challenge_${seed}" \
        --params-output "expN/degradation/blind_${seed}.json" --quiet
    cp expN/challenge_${seed}/.ground_truth.json expN/.ground_truth/blind_${seed}.json
done
```

### 盲识别（子 Agent 并行）

每个 Agent 负责 6 个退化，使用同图模式：
- `analyze_degradation.py --target <d> --clean <c>` → ratios_vs_clean
- 假设 → 模拟 → 比较 → 迭代（最多 5 轮）
- 输出 `predicted_params/{id}.json`

## 训练流程

### Launcher 隔离

```bash
# Agent 不写完整命令，只用 launcher
bash expN/scripts/launcher.sh {challenge_id}          # Spec
bash expN/scripts/launcher.sh {challenge_id} --ft     # Ft
bash expN/scripts/launcher.sh {challenge_id} -c {id}_R1  # R1
```

### 反思修正

**只改 PARAMS，VAL 锁定 GT**：
- 创建 `predicted_params/{id}_R1.json`（修正后退化）
- launcher 始终将 VAL 指向 `.ground_truth/{base_id}.json`
- Spec 和 R1 在同一 GT 上评估 → 对比有意义

## DFPIR 基线

| 模式 | 何时用 | 命令 |
|------|--------|------|
| 单 GPU direct | 可与训练并行 | `CUDA_VISIBLE_DEVICES=X python test_degradation.py --gpus 0` |
| 8 GPU spawn | 全部 GPU 空闲 | `python test_degradation.py --gpus 0,1,...,7` |

⚠️ spawn 在 GPU 有进程时死锁。用 `dfpir_runner.sh` 串行执行。

## 常见错误速查

| 错误 | 症状 | 修复 |
|------|------|------|
| VAL_PARAMS_PATH 被修改 | Spec vs R1 对比无效 | 用 launcher 锁定 |
| 跨图模式盲识别 | CI pass < 5/10 | 用 --same-image |
| 暴力枚举 12+ 假设 | 大量无意义测试 | LLM 视觉推理 + 迭代 |
| rebase | 历史丢失 | git merge/push 只 |
| DFPIR spawn + GPU 忙 | 进程挂起 | GPU 空闲后跑 |
| 挑战重生成 | Agent 结果作废 | 一次性生成完毕 |
