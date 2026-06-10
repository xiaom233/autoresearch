# exp12: 全流程端到端验证

## 目标

验证完整 IRAgent 流程：盲识别 → 训练 → 反思修正 → 超越盲基线和大模型。

## 退化配置

| 类型 | 数量 | 说明 |
|------|:--:|------|
| 单退化 | 8 | 1 步随机退化 |
| 双退化 | 24 | 2 步随机退化 |
| 三退化 | 24 | 3 步随机退化 |
| **总计** | **56** | 随机从 37 种函数采样 |

⛔ 退化内容通过 `setup_challenge.sh` 生成，Agent 全程不接触 GT。

## Pipeline

### Step 1: 生成 56 组退化 + 训练 M_blind (并行, ~3h)

- 随机生成 56 个 params.json (8 single, 24 dual, 24 triple)
- 训练 M_blind: `PARAMS_PATH=None, EPOCH_BUDGET=2` (~86 min)
- 同时测试 DFPIR 基线

### Step 2: 基线测试 (8 GPU, ~2h)

- M_blind 在 56 组上测试
- DFPIR 在 56 组上测试
- 建立 baseline PSNR

### Step 3: 盲识别 (Skill, ~2h)

- 对每组退化，用 image-degradation-simulator skill 识别
- 输出 predicted_params.json

### Step 4: 专家训练 (8 GPU, ~10h)

- 56 组 expert 训练 (EPOCH_BUDGET=2)
- 每组在 predicted_params.json 上训练

### Step 5: 反思修正（子 Agent 异步执行）

**机制**：每个退化训练完成后，自动检查是否需要反思。如果需要，**启动子 Agent** 独立执行反思分析，不影响主 GPU 训练队列。

**基线定位**：
- M_blind：盲预训练（无任何优化）
- M_Ft：盲预训练→微调（简单策略，非智能优化）
- DFPIR：All-in-one 大模型
- M_spec：本系统产出（识别→训练→反思）

**反思触发条件**：每轮训练完成后**始终尝试反思**，追求无上限的性能提升。

**停止条件**（非触发条件）：
- 本轮反思后 PSNR 提升 < 0.5 dB（边际收益不足）
- 或已达 3 轮上限

**子 Agent 工作流**：
```
训练完成 → 启动反思子 Agent
  ├─ 读取 REFLECTION_MECHANISM.md 规则
  ├─ 分析诊断特征（edge/chroma/hflf/图像）
  ├─ 即使 Spec 已优于 Blind，也检查是否可进一步改善
  ├─ 提出修正建议 → 提交重训练
  ├─ 对比修正前后 PSNR
  └─ Δ < 0.5dB 或 3 轮 → 停止
```

**优势**：
- GPU 持续满载（子 Agent 分析期间，其他退化训练不中断）
- 反思决策可审计（每个子 Agent 独立记录分析过程）
- 多退化并行反思（多个子 Agent 可同时分析不同退化）

### Step 6: 文档生成

每组退化自动生成报告 `exp12/reports/{name}.json`，含：
- 退化管线（训练完成后公开）
- Blind / Ft / Specialist 三个基线 PSNR
- 反思轮次和修正记录
- 最终结论

### Step 7: 最终对比

### Step 6: 最终对比

- M_spec_final vs M_blind vs DFPIR
- 统计超越率

## 时间估算

| Step | 时间 |
|------|:--:|
| M_blind 训练 | 1.5h |
| 基线测试 | 2h |
| 盲识别 | 2h |
| 56 组专家训练 | 10h |
| 反思修正 | 3h |
| **总计** | **~18h** |

## 防泄露协议

1. 退化文件通过脚本批量生成，Agent 不接触文件内容
2. Skill 调用在子进程中执行
3. 只通过 challenge_id 引用退化
4. 遵守 program.md 五层隔离
