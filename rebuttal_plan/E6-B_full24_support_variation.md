# E6-B full24：全 24 场景 × 2 组随机 support 的稳定性实验

> 日期：2026-10-30。产物目录：`rebuttal_runs/E6B_full24/`（新目录，不覆盖 `E6` / `E6_LLM` / `E6A_full24`）。
> 用户确认：48 条链 = 24 场景 × 2 组 support；每组直接随机采样 5 张 DIV2K_train 图像。

## 1. 问题

更换 support set 会让完整 AutoIR 流程的输出与最终 test PSNR 变化多大？
E6-A full24 测的是**同一 support** 的 agent run-to-run 方差；本实验测的是**support 变化**下的系统级稳定性（含 agent 随机性），两者互补。

## 2. 设计（预注册）

| 项 | 设置 |
|---|---|
| 场景 | exp41 `blind_0001–blind_0024`（24 个；1/2/3 步各 8） |
| support | 每场景 2 组：`setA`、`setB`；每组**独立随机采样 5 张** `datasets/DIV2K/DIV2K_train_HR/*.png`（不强制两组不重叠；重叠会在结果中统计并披露） |
| 采样种子 | 固定 `20261030`；`random.Random(stable_seed(seed, sid, set))`，记录 image_id（采样前固定，不按结果调整） |
| support 退化 | 用该场景 GT pipeline 生成 degraded；`make_lq` 以 `(sid, set, image_id)` 派生固定 seed，生成后保存 PNG 并记录 sha256；GT 参数不进入 challenge 目录 |
| 流程 | 完整 LLM：Skill 盲识别 R0 → 策略子 Agent（`training_strategy.json`）→ R0 训练 → R1/R2 反思（support-only 触发）→ 跨轮 support PSNR 选点 → 固定 GT-LQ test |
| R0 训练 | 策略派生架构；EPOCH 2 / ~15,094 steps，batch 16，LR 5e-4（策略可覆盖），seed 42，`AR_CHALLENGE_DIR` = 5 对 support |
| R1 触发 | R0 verdict ∈ {UNCERTAIN, POOR} 或 support PSNR <35 或 NaN；`LIKELY + support≥35` 跳过 |
| R1/R2 策略 | pipeline 改变（CHANGE）时，新增策略子 Agent 生成 `training_strategy_R1/R2.json`；仅 CHANGE 才训练 |
| R2 触发 | R1 CHANGE 且 R1 support PSNR 比 R0 高 >2.0 dB |
| 最终选择 | 在 R0/R1/R2 中按 best support PSNR 选一个 checkpoint（并列取较早轮次） |
| test | 固定 GT-LQ：DIV2K 100 + LSDIR 250；7 个私有 RNG 场景用 `E1E2_fixed_lq` 共享物化 LQ，其余 17 个用确定性 on-the-fly LQ；统一 PSNR/SSIM |
| GT 用途 | 仅 support 构造、test LQ 与事后匹配；不进入盲识别/策略/反思 Agent |

## 3. 预注册指标

- 每场景 setA / setB test PSNR（DIV2K、LSDIR），within-scenario SD / range。
- macro：24 场景均值、场景间 SD、mean within-SD、max range。
- 按 GT 步数 1/2/3 分层。
- 盲识别输出一致性（setA vs setB；不是同 support 的 run-to-run，包含 support 与 agent 双重差异）。
- support 重叠统计：每场景 setA∩setB 的图像数；跨场景图像复用率。
- 与 E6-A 参考对比：E6-A 同 pipeline 的 agent/选点抖动 mean |Δ| 0.27 dB（max 1.38），全场景 within-SD 1.491 dB；本实验的 support 方差应与此对照解释。
- support→test 迁移：setA/setB 的 support PSNR 差与 test PSNR 差的关系。

## 4. 失败处理

- Agent 未产出完整输出 → 该链无效，重跑一次；仍失败标记缺失，不插补。
- 训练 NaN / 无有效 checkpoint → 固定配置重训一次；仍失败标记缺失。
- 不因 test 结果挑选/删除场景或 support。

## 5. 成本

- support 生成：240 张 degraded（CPU）。
- R0：48 盲识别 + 48 策略 + 48 训练（~68 GPU-h）；R1/R2：按触发与 CHANGE 追加（预计 ~10–20 次训练，~15–30 GPU-h）。
- test：48 模型 × 350 图，8 卡约 30–45 min。
- 预计总训练 85–100 GPU-h，8 卡约 11–13 h 墙钟（含 agent 批次）。

## 6. 运行状态（2026-10-30 10:43）

- **R0 训练**：31/48 完成（8 卡运行中，9 个 R0 任务 + R1 队列排队中）。
- **R1 触发**：已完成 R0 的 31 条链中 29 条需要 R1（2 条 LIKELY 且 support PSNR ≥35 直接跳过）。
- **R1 反思**：29/29 完成；**25 NO_CHANGE、4 CHANGE**（`blind_0001_setA`、`blind_0012_setB`、`blind_0013_setB`、`blind_0015_setB`）。
  - CHANGE 示例：0001_setA 由 `poisson5+qhist2+brightness2` 收敛为 `noise_poisson:5`；0012_setB 由 `motion3+jpeg4` 改为 `motion5+jpeg3`；0013_setB 由 R0 管线收敛为 `speckle3+jpeg2`（5 对 support 全 MATCH）；0015_setB 改为 `motion1+brightness_darken3`（5 对 36.7–58.9 dB）。
  - NO_CHANGE 多为 R0 管线在 5 对 support 上已 MATCH 或逐像素复现（如 0010/0014 的 `brightness+jpeg` 组合 5/5 PSNR=inf，低 support PSNR 归因于裁剪/训练上限而非识别错误）。
- **R1 训练**：4 条 CHANGE 的 `training_strategy_R1.json` 已由策略子 Agent 生成并入队；25 条 NO_CHANGE 仅记录 skip（不占 GPU）。0012_setB R1 策略为 Direct+OCAB（ws16/dim30 参数对齐，anisotropy 1.78 + PCA 4.06 判 motion）。
- **blind_0023_setB**：盲识别最终定稿 `blur_glass:5 + contrast_weaken_scale:1 + compression_jpeg:4`（UNCERTAIN，5 对 support 25.1–25.8 dB，加入 contrast 步骤 5/5 提升 0.19–2.43 dB）；R0 策略已按新管线重新生成（Swin+SG+FiLM-GCM，contrast Tier A `variance_ratio` 0.59–0.70 + 结构化 JPEG 门控），R0 待训练。
- 所有 R1/R2 反思均通过 worktree 隔离子 Agent 执行，禁用 GT 路径；NO_CHANGE 链的 `training_strategy_R1.json` 为 R0 策略的机械复制（pipeline 未变）。

## 7. 最终结果（2026-10-30 完成）

**覆盖与成本**：48/48 R0；46/48 触发 R1（2 条 LIKELY+support≥35 跳过），其中 7 条 CHANGE 训练 R1；R2 触发 3 条（0001_setA +10.19、0015_setB +13.75、0020_setA +6.46），仅 0020_setA 训练 R2（support 18.25 < R1 20.07，选点回退 R1），另 2 条 NO_CHANGE。总 **56 次训练 / 82.57 GPU-h**（R0 70.58 + R1 10.55 + R2 1.43）。Agent 墙钟：盲识别 48×19.4 min（中位 14.0，max 100.9；合计 15.5 h）；R1 反思 46×11.5 min（中位 9.0）；R2 反思 3×4.0 min；API token 未记录。

**主表（24 场景等权；hybrid fixed-LQ test：DIV2K 100 + LSDIR 250）**

| 数据集 | setA | setB | 合并 mean | 场景间 SD |
|---|---:|---:|---:|---:|
| DIV2K | **27.633** | **27.540** | **27.586** | 6.813 |
| LSDIR | **25.177** | **25.178** | **25.177** | 6.494 |

**within-scenario（support-set 差异）**

| 数据集 | mean SD | median SD | max SD | mean range | median range | max range |
|---|---:|---:|---:|---:|---:|---:|
| DIV2K | 0.790 | 0.258 | 6.240 | 1.118 | **0.365** | **8.825** |
| LSDIR | 0.724 | 0.142 | 6.396 | 1.023 | **0.200** | **9.046** |

- **setB − setA**：DIV2K −0.093 [−0.879,+0.929]（setB 胜 10/24）；LSDIR +0.001 [−0.706,+1.002]（11/24）。无系统性方向：support 抽样是方差源，不是偏差。
- **重尾**：`blind_0015` setA 20.91 vs setB 29.73（range 8.83/9.05）；`blind_0009` 31.07 vs 26.32（4.75/3.41）；`blind_0018` 19.31 vs 16.29（3.02/2.25）。其余多数场景 range <0.5 dB（中位 0.365/0.200）。
- **选点**：setA 21 R0 / 3 R1 / 0 R2；setB 20 R0 / 4 R1 / 0 R2。R1 被选中的 7 条：0001_setA、0012_setB、0013_setB、0015_setB、0018_setA、0019_setB、0020_setA。
- **按 GT 步数（DIV2K）**：1 步 34.677（range 0.160）；2 步 27.009（1.942）；3 步 21.073（1.251）。LSDIR：31.870 / 24.512 / 19.149。
- **与 E6-A 参考**：E6-A 使用 DIV2K_valid support（不同来源）为 27.654/25.240；E6-B 合并 27.586/25.177，差约 0.07 dB。5 张 support 的抽样不改变 macro，但尾部场景可差 ~9 dB。
- **结论**：support 变化对中位场景约 0.2–0.4 dB，但存在 2/24 的灾难性尾部（0015 8.8 dB、0009 4.75 dB）；不能声称普遍稳定，也不能只报 macro 均值。
- 产物：`E6B_full24/e6b_full24_newonly_summary.json/.md`、`E6B_full24_support_variation.md`；逐图 16,800 行在 `E6B_full24/test_metrics_final_gpu*.jsonl`；选点 `selection_final.jsonl`；汇总脚本 `scripts/aggregate_e6b_full24_newonly.py`。
