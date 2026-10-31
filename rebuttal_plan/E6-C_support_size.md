# E6-C：support 数量消融（1 张 vs 3 张 vs 5 张）

> 状态：**已完成（2026-10-31）**。结果为：DIV2K size1/3/5 27.241/27.261/27.633，LSDIR 24.844/24.820/25.177；配对差 CI 均跨 0。详见 §10。
> 目的：回应“系统需要 5 对 support、实践中难获得”（Weakness 5）以及 support 数量的成本/收益问题。现有实验固定 5 张 support；本实验在 exp41 全部 24 个场景上分别测试 **1 张**和 **3 张** support。

## 1. 问题

AutoIR 在 5 对 support 上工作。若只有 1 对或 3 对：
- 盲识别/反思/策略的输出如何变化？
- 最终 test PSNR 下降多少？哪些场景下降最大？
- 1→3→5 的收益是否单调？是否在部分场景已饱和？

## 2. 设计（预注册）

| 项 | 设置 |
|---|---|
| 场景 | exp41 `blind_0001–blind_0024`（24 个；1/2/3 步各 8） |
| support 来源 | `datasets/DIV2K/DIV2K_train_HR`（与 E6-B 相同 pool；与 test 的 DIV2K_valid 不重叠） |
| support 基数 | 每个场景 2 条链：`size1`（1 对）、`size3`（3 对） |
| 嵌套设计 | 以 E6-B `setA` 的 5 张 support 为全集，用固定种子做确定性排列 `π(sid)`，`size1={π[0]}`、`size3={π[0],π[1],π[2]}`、`size5=setA`（即 size1 ⊂ size3 ⊂ setA）。**5 张基线直接复用 E6-B setA 结果**，保证 1/3/5 图像身份受控 |
| 排列种子 | `stable_seed(20261030, sid, "size_order")`；记录 π 与 image_id |
| support 退化 | 用该场景 GT pipeline 生成；`make_lq` 种子由 `(sid, image_id)` 派生（**与 size 无关**，保证同一图像在 size1/size3/setA 下 LQ 完全相同）；保存 PNG + sha256 |
| 流程 | 完整 LLM：Skill 盲识别 → 策略子 Agent → R0 训练 → R1/R2 support-only 反思 → 跨轮 support PSNR 选点 → 固定 GT-LQ test |
| R0 训练 | 同 E6-B：EPOCH 2 / 15,094 steps、batch 16、LR 5e-4、seed 42、`AR_CHALLENGE_DIR` = size-k support |
| R1 触发 | verdict ∈ {UNCERTAIN,POOR} 或 support PSNR <35 或 NaN；LIKELY+support≥35 跳过 |
| R1/R2 | 仅 pipeline 改变（CHANGE）才追加训练；R2 触发为 R1 support PSNR 比 R0 高 >2.0 dB |
| 最终选择 | R0/R1/R2 中 best support PSNR（并列取较早轮次） |
| test | 固定 GT-LQ DIV2K 100 + LSDIR 250；7 个私有 RNG 场景用 `E1E2_fixed_lq` 共享物化 LQ，其余 17 个确定性 on-the-fly；RGB PSNR（另报 SSIM） |
| GT 用途 | 仅 support 构造、test LQ、事后评估；不进入盲识别/策略/反思 Agent |

## 3. Agent 协议差异（必须显式处理）

- **盲识别**：pair0 仍是主分析对象；可用 support 对为 k=1 或 3。候选验证只在 k 对上做，**不得**补充其他 support 或 test 数据。
- **反思**：验证集同样是 k 对；CHANGE 判据仍为“确定性 +2 dB / >40 dB”或“噪声 verify_signals 多数对改善”，但 k=1 时只有单对，必须记录该限制。候选预算 ≤15 次 `test_candidate.py`。
- **策略**：与 E6-B 相同门控；不得因为 support 少而放宽阈值。
- **禁止**：读取 E6-B/E6-A 其他挑战的 support 或结果；不得用 5 张 setA 的其余图像“补”给 size1/size3（否则不再是 1/3 张实验）。
- 三个子 Agent 的指令文件在 `E6C_support_size/` 下单独提供，明确 k 值和可用 pair 列表。

## 4. 指标与分析

- 每场景每 size：test PSNR（DIV2K、LSDIR 分列）、support PSNR、选中轮次、盲识别 pipeline。
- **主分析**：按场景配对的 `size3 − size1`、`size5 − size3`、`size5 − size1`；均值、中位、95% bootstrap CI、胜场（24 场景）。
- 分层：GT 1/2/3 步；盲识别 exact match（size1/size3 内部及与 size5）；R1/R2 触发率与 CHANGE 率。
- support→test 迁移：k=1/3/5 的 support PSNR 与 test PSNR 的秩相关。
- 披露：k=1 的 support PSNR 噪声大；本设计是“同一 setA 内的嵌套缩减”，结论限定在 setA 条件下，不外推到任意随机 1/3 张。
- 可选敏感性（不属主结论）：若预算允许，另做一组独立随机 1 张抽取，估计“1 张”的抽样方差；未预注册前不纳入主表。

## 5. 触发与执行顺序

1. 等待 E6-B full24 的 R2 判定/训练、跨轮选点、hybrid fixed-LQ test 全部完成。
2. 运行 `E6C_support_size` setup：从 E6-B `support_plan_gt.json` / 各挑战 `support_manifest.json` 读取 setA，生成 48 个 challenge 目录（size1/size3）。
3. 启动 48 条 R0：8 卡队列（`scripts/exp_launcher.sh` 或 E6-B 的 gpu_runner 参数化版本）。
4. R0 完成 → collect → 按 support-only 触发 R1；CHANGE 生成策略并训练 R1；R1 +>2 dB 触发 R2。
5. 跨轮选点 → hybrid fixed-LQ test（DIV2K+LSDIR 350 图）→ 三组（1/3/5）配对统计。
6. 更新 `02_结果与逐条回应`、`05_剩余实验状态`、`STATUS.md`，再提交。

## 6. 成本估计

- 盲识别：48 × ~19 min（agent 墙钟，可并行）。
- R0：48 × 1.48 GPU-h ≈ **71 GPU-h**。
- R1/R2：按 CHANGE 率 15–25% 估 8–12 次训练 ≈ 12–17 GPU-h；R2 若少量触发另加。
- test：48 模型 ×350 图 ≈ 0.5–1 GPU-h。
- 合计约 **85–95 GPU-h**；8 卡串行约 11–13 h 墙钟（含 agent 批次）。

## 7. 失败处理

- Agent 未产出完整文件 → 该链重跑一次；仍失败标记缺失，不插补。
- 训练 NaN/无有效 checkpoint → 固定配置重训一次；仍失败标记缺失。
- 不因 test 结果挑选/删除场景或 size。

## 8. 产物与脚本

```text
rebuttal_runs/E6C_support_size/
├── challenges/blind_XXXX_size1|size3/   # clean/degraded + 预测/反思/策略
├── experiments/                          # R0/R1/R2 checkpoints
├── logs/                                 # 训练/评测日志
├── selection_final.jsonl                 # 跨轮选点
├── test_metrics*/                        # 逐图 PSNR/SSIM
├── e6c_summary.json/.md                  # 三组统计
└── support_plan_nested.json              # π、image_id、sha256（chmod 600）
scripts/e6c_support_size_setup.py         # 从 E6-B setA 生成 size1/size3
scripts/e6c_train_one.sh / e6c_train_reflect.sh / e6c_collect.py / e6c_select_rounds.py
```

实现方式：复制 E6-B 的脚本并将 `OUT=rebuttal_runs/E6B_full24` 参数化为 `E6C_support_size`；训练/反思/收集/选点逻辑不变，仅 support 数量和 challenge 路径不同。

## 9. 风险与限制

- 嵌套设计使 1/3/5 共享同一组图像，控制了图像身份，但结论条件于 setA；不能直接说“任意 1 张 support 的平均表现”。
- k=1 时 support PSNR 只有单图，选点和 R2 触发都更噪声；需在结果中标注。
- Agent 随机性未重复（每个 size 一次运行）；如需 run-to-run 方差，另做重复（不在本计划）。
- 本实验不重新训练 5 张基线：直接使用 E6-B setA 的已选 checkpoint 与 test；必须确认 E6-B test 协议与本计划完全一致。

## 10. 结果（2026-10-31 完成）

**覆盖与成本**：48 R0 + 5 R1 + 1 R2 = **54 次训练 / 77.85 GPU-h**（R0 69.32、R1 7.11、R2 1.42）。R1 定稿 45/48；5 条 CHANGE（0001_size3、0010_size3、0013_size3、0020_size3、0024_size3）；R2 仅 0010_size3 训练（support 31.9，选点选中 R2）。0018_size1 曾出现训练 NaN，按计划重训一次后纳入。

**主表（24 场景等权；固定 GT-LQ test）**

| 数据集 | size1 | size3 | size5（E6-B setA） |
|---|---:|---:|---:|
| DIV2K | **27.241** | **27.261** | **27.633** |
| LSDIR | **24.844** | **24.820** | **25.177** |

**配对差（场景级 10k bootstrap）**

| 数据集 | size3−size1 | size5−size3 | size5−size1 |
|---|---:|---:|---:|
| DIV2K | +0.019 [−1.173,+0.976]（13/24） | +0.372 [−0.246,+1.389]（14/24） | +0.391 [−0.086,+1.008]（11/24） |
| LSDIR | −0.024 [−1.072,+0.785]（12/24） | +0.357 [−0.171,+1.233]（14/24） | +0.333 [−0.089,+0.847]（12/24） |

**结论**

- support 数量从 1→3 的宏平均变化 ≈0（DIV2K +0.02 dB、LSDIR −0.02 dB）；3→5 的增益 +0.37/+0.36 dB，1→5 的增益 +0.39/+0.33 dB，**三个 CI 全部跨 0**。
- 逐场景差异有正有负：5 张相对 1 张最好的是 blind_0001（+5.79）、0014（+2.36）、0013（+1.87）；反过来 5 张更差的是 0015（−1.35）、0024（−1.05）。这与 support-set 抽样的重尾一致，不是“越多越稳”。
- 分层（DIV2K size1/3/5）：1 步 34.075/33.542/34.666；2 步 26.101/26.539/26.800；3 步 21.548/21.702/21.431。2 步场景从更多 support 中受益最明显，3 步基本持平。
- **可支持的结论**：在宏平均意义上，1 张 support 即可接近 5 张（差异不显著）；5 张的边际收益小且场景相关。**不能支持**“support 越多越稳定”——005/0024 等场景 5 张反而更差。
- **限制**：嵌套设计条件于 E6-B setA；k=1 的 support PSNR 噪声大；每个 size 只跑 1 次，未做 run-to-run 重复；未做独立随机 1 张抽样（预注册的可选敏感性未执行）。

产物：`rebuttal_runs/E6C_support_size/e6c_summary.json/.md`、`selection_final.jsonl`、`test_metrics_final_gpu*.jsonl`（16,800 行）；汇总脚本 `scripts/aggregate_e6c_support_size.py`。

## 11. 审计与限制（2026-10-31）

完整审计见 `rebuttal_runs/E6C_support_size/AUDIT.md`。通过项：48/48 R0（0 NaN；46 Swin+SG、2 OCAB）、5/5 R1、1 R2；选点 48/48 且 checkpoint 齐全；test 16,800 行无重复，`lq_sha256` 与 E6-B setA **16,800/16,800 完全一致**；宏平均可复算。

**必须披露的两点**：

1. **7 个私有 RNG 场景**（0003/0005/0017/0018/0020/0022/0023）的 support LQ 本身不可复现（同一图同种子两次生成不同），因此 size1 与 size3 共享图的 LQ 在这 7 个场景不一致；其余 17/24 场景逐字节一致。
2. **size5 基线的 support LQ salt 不同**（E6-B `e6b-full-v1` vs E6-C `e6c-size-v1`）：clean image 相同、pipeline 相同，但随机退化 realization 不同。support 仅用于诊断与 checkpoint 选择，训练梯度来自合成数据，故影响限于选点/诊断。

**17 个确定性场景敏感性**（排除 7 个私有 RNG 场景）：

| 数据集 | size1 | size3 | size5 | size5−size1 | size5−size3 |
|---|---:|---:|---:|---:|---:|
| DIV2K | 27.925 | 27.863 | 28.455 | +0.530 [−0.140,+1.357]（8/17） | +0.592 [−0.241,+1.914]（9/17） |
| LSDIR | 25.526 | 25.412 | 25.934 | +0.408 [−0.172,+1.082]（7/17） | +0.523 [−0.216,+1.694]（9/17） |

与全 24 场景结论一致：1→3 ≈0，5 张相对 1 张仅 +0.4–0.5 dB，CI 跨 0。**结论不由 7 个 LQ 不可控场景驱动。**
