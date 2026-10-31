# E6-C 实验审计（2026-10-31）

审计对象：`rebuttal_runs/E6C_support_size/`（24 场景 × {size1,size3}，嵌套于 E6-B setA）。
结论：**训练、选点、test 协议通过；两处 support LQ 限制必须披露**。主结论在 17 个确定性场景的敏感性分析下不变。

## 1. 通过项

| 检查 | 结果 |
|---|---|
| challenge 目录 | 24 个 size1 + 24 个 size3；size1 每目录 1 对、size3 每目录 3 对（含 alias） |
| support 干净图嵌套 | 24/24 场景满足 `size1={π[0]}`、`size3={π[0..2]}`、`size5=setA` |
| support 目录 GT 隔离 | challenge 目录内无 `degradation_gt`/GT 文件；`support_plan_gt.json` 在实验根目录，权限 **600** |
| R0 训练 | 48/48 有 `TRAIN END`、best checkpoint、`training_seconds`；**0 NaN** |
| R0 架构 | 46 条 Swin+SG（dim64/ws8），2 条 OCAB（0016_size1/size3，dim30/ws16），与策略一致 |
| R1 训练 | 5/5 CHANGE 链完成（0001_size3、0010_size3、0013_size3、0020_size3、0024_size3）；45 份 R1 决策；0 NaN |
| R2 | 0010_size3 训练 1 次并被选中；0013_size3/0020_size3 R2 NO_CHANGE；0001_size3 R1 29.97 未触发 |
| 选点 | 48/48 行；每行 `selected_round` 确为候选 support PSNR 最大者（并列取早轮）；checkpoint 全部存在；R0 45 / R1 2 / R2 1 |
| test 覆盖 | 16,800 行 = 48 runs × 350 图；无重复 `(run,dataset,image)`；DIV2K 4,800 + LSDIR 12,000 |
| test LQ 一致性 | 与 E6-B setA 同一 (scenario,dataset,image) 的 `lq_sha256` **16,800/16,800 全部一致，0 mismatch** |
| 结果复算 | 由逐图 metrics 重算 DIV2K 27.2414/27.2609、LSDIR 24.8438/24.8200，与 `e6c_summary.json` 一致 |
| Agent 禁读 | 输出中 `test_metrics`/`exp41` 的 5 次出现均为“未读取…”的自述，无实际访问证据；未见 GT/其他挑战路径内容 |

## 2. 必须披露的限制

### 2.1 7 个私有 RNG 场景的 support LQ 不可复现

- 场景：`0003, 0005, 0017, 0018, 0020, 0022, 0023`（与 E6-B/E1 的私有 RNG 集合一致）。
- 实测：同一 clean 图、同一 `make_lq` 种子连续调用两次，输出 sha256 **不同**；因此 size1 与 size3 共享图的 `degraded_sha256` 在这 7 个场景不一致。
- 其余 **17/24** 场景的 size1/size3 共享 support 图 LQ **逐字节一致**。
- 含义：嵌套设计在 17 个场景严格成立；7 个场景同时改变了“数量”与“support 退化 realization”。这是底层退化函数不可播种的已知问题（E6-B 也同样存在，test 侧已用 fixed-LQ 规避）。

### 2.2 size5 基线的 support LQ salt 与 E6-C 不同

- E6-C 使用 `e6c-size-v1`，E6-B setA 使用 `e6b-full-v1`；两者 clean image 完全相同、pipeline 相同，但随机退化的 realization 不同（共享图 72 对中 54 对不同）。
- support 图仅用于 **agent 诊断与 checkpoint 选择**（`AR_CHALLENGE_DIR` 验证集）；训练梯度仍来自 DIV2K clean + 预测 DC 的合成数据。因此影响限于选点/诊断，不改变训练数据。
- 结论表述应写“同一批 clean support images 的嵌套”，不能写“完全相同的 support 对”。

## 3. 敏感性：排除 7 个私有 RNG 场景（17 场景）

| 数据集 | size1 | size3 | size5 | size5−size1 | size5−size3 |
|---|---:|---:|---:|---:|---:|
| DIV2K | 27.925 | 27.863 | 28.455 | +0.530 [−0.140,+1.357]（8/17） | +0.592 [−0.241,+1.914]（9/17） |
| LSDIR | 25.526 | 25.412 | 25.934 | +0.408 [−0.172,+1.082]（7/17） | +0.523 [−0.216,+1.694]（9/17） |

- 与全 24 场景一致：1→3 ≈0，5 张相对 1 张仅 +0.4–0.5 dB，CI 仍跨 0。
- 因此 §10 的结论不是由 7 个 LQ 不可控场景驱动。

## 4. 其他

- 0018_size1 的原 NaN 训练目录/日志已移至 `experiments/blind_0018_size1_nan_backup`、`logs/blind_0018_size1.log.nan_backup`；重训日志无 NaN，选点使用重训 checkpoint。
- 0001_size3 R1 的 wrapper 未写 `TRAIN END`，按日志 mtime 与 train.py 完成状态补记（best step 15092，support 29.97；train.py 已正常结束）；checkpoint 与日志均在。
- 汇总脚本 `scripts/aggregate_e6c_support_size.py`；本次审计检查为只读脚本化核对，未修改训练/预测文件（除上述 0001_size3 日志补记与 7 个 R1 pipeline 格式规范化）。
