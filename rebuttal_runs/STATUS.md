# Rebuttal 执行状态（2026-10-26）

## 已完成

| 项 | 结果 | 路径 |
|---|---|---|
| 远程同步 | `dev` = `origin/dev` (`3e57eb9` + 本地 merge) | — |
| A0 exp41 R0–R2 清单 | 72 行、288 checkpoint、71 次真实训练、101.95 GPU-h | `rebuttal_runs/A0/` |
| support 选择 | R0=5 / R1=8 / R2=11；support 25.06 vs last-step 24.25 dB | `A0/support_selection.jsonl` |
| AutoIR 固定 LQ 测试 | DIV2K **24.968** / LSDIR **22.956** dB；排除校准重叠图 0801–0810 后 24.963/22.956 | `A0/test_metrics/`、`A0/autoair_metrics.json` |
| E3 组合空间 | 1–3 步 **144,615** 有序配置；校准脚本 128 + 1,320 唯一配置 | `E3/` |
| E3 校准重叠 | blur 用 0801、JPEG 用 0801–0810；**10/100 DIV2K 测试图重叠**；exp41 链重叠 3/24 | `E3/E3_calibration_overlap.json` |
| E5a DC 复算 | exp41 R2：位置 Type 43.75%、整链 Type 37.50%、整链 Spec 29.17%；R0：25.00/25.00/20.83% | `E5a/` |
| AR_SEED | `AR_SEED` 覆盖 Python/NumPy/Torch/CUDA/workers，默认 42 | `train.py` |
| 显式评估器 | 逐图 PSNR/SSIM(VGG)/LPIPS + LQ/checkpoint 哈希，无回退 | `scripts/eval_checkpoint.py` |
| 安全启动器 | 无全局 pkill；成功标志+哈希；断点续跑 | `scripts/safe_launcher.py` |
| E2 DFPIR 脚本 | support 验证、1/3+2/3+T 快照、last/best、取数计入预算 | `scripts/train_dfpir_budget.py` |

## 主要结果（最新）

| 任务 | 状态 | 预计 |
|---|---|---|
| E2 24 场景 DFPIR-ft | ✅ 24/24 完成；同 T_s、同 DC* | `E2/*/*_summary.json` |
| E2 固定 LQ 测试 | ✅ 24×3 checkpoint（best/last/original）；**DFPIR-ft best +0.682 DIV2K / +0.659 LSDIR vs AutoIR**；last +0.398/+0.407；original −0.974/−0.926 | `E2/e2_test_summary.json` |
| E1 R0 | ✅ 24/24 完成（96 ckpt）；support PSNR 均值 21.96（9.91–35.11） | `E1/experiments/*_R0` |
| E1 R1 | ✅ 19/19 完成；无 NaN；R1 support 有明显提升（如 0004 +9.6、0005 +8.0） | `E1/stage.log` |
| E1 R2 | ✅ 9/9 完成（GPU0 两个任务）；R2 无 NaN | `E1/stage.log` |
| E1 修正 | ✅ 7/7 完成：0002 R0/R1、0009/0012/0013/0021 R0、0018 回退；无 NaN | `E1_fix/logs/`、`E1/logs/blind_0018_R1.log` |
| E1 最终（旧 AutoIR） | 规则 24.272/22.132 vs **旧 AutoIR（A0/exp41 六月链）** 24.968/22.956；Δ rule−旧AutoIR −0.697/−0.825 dB，CI 跨 0，规则胜 8/24 | `E1/e1_final_summary.json` |
| **E1 新 LLM（2026-10-30）** | 用 **E6-A full24 final checkpoints**（新 Skill agents，R0/R1）替换 LLM 侧：DIV2K **28.030 / 27.277 / 27.654**（rep1/rep2/mean），LSDIR **25.554 / 24.928 / 25.241**；vs rule **+3.758 [+1.624,+6.257] / +3.006 [+1.274,+5.134] / +3.382 [+1.520,+5.619]**（18/24 胜），LSDIR +3.422 / +2.796 / +3.109（19/24） | `E1/e1_newllm_summary.json`、`e1_newllm_summary.md` |
| E1 结论 | **旧 A0/exp41 链**下 LLM 无统计显著独立优势（CI 跨 0）；**新 LLM（E6-A full24 final）** 显著优于规则（DIV2K +3.382，18/24；LSDIR +3.109，19/24），但新 LLM 用固定 Direct、规则用规则架构+R0–R2，不是单一架构变量；场景互补仍然存在（0005 规则 +23.0；0008 AutoIR +20.5 等旧链现象需按版本区分） | T2/逐条回应 |
| PSNR 口径 | 全部对比使用 **RGB joint-MSE PSNR**（`psnr_rgb`，非 YCbCr、非逐通道平均）；E1/E6-A/E6-B/SIDD 逐图字段均无 `psnr_y`；`train.py` 的 checkpoint 选择与 `val_psnr_db` 也用 `psnr_rgb` | `E1/PSNR_PROTOCOL.md` |
| E1 结论 | LLM 无统计显著独立优势；场景互补（0005 规则 +23.0；0008 AutoIR +20.5） | T2/逐条回应 |
| E1 成本 | 训练 59 次 / 83.99 GPU-h；support+test 评估 ~2.6 GPU-h | `E1/manifest.jsonl` |
| E1 0018 NaN 回退 | ✅ 完成；support 17.10，最终选中 R1 | `E1/logs/blind_0018_R1.log` |
| E2 固定 LQ 测试 | ✅ 完成；DFPIR-ft best +0.682 DIV2K / +0.659 LSDIR | `E2/e2_test_summary.json` |
| E4 数据 | ✅ SIDD validation lmdb（5 support / **1,280 test 块 = 40 场景×32，官方全量**；含 5 个 support 块 0.39%；旧 1,120 = 35 场景敏感性）；UHDM test.tar.gz（5 support / 495-of-500，待补 500）；Rain100H（5 support / 95-of-100，待补 100）；未下载训练集 | `datasets/E4/` |
| E4 SIDD 基线（35 均衡子集） | M_blind 30.763；DFPIR official release 5D 26.972（CLIP σ=25；random text 23.678 非官方）；blinded DFPIR 31.986（项目参考） | `E4/sidd/baselines/`、`E4/sidd/baselines_official/` |
| E4 SIDD full 1,280 | **1,280 块（40 场景×32，官方 validation 全量）**：LQ 23.662 / M_blind 30.634 / **DFPIR official release 5D 26.424（CLIP σ=25）** / blinded 31.395（参考）/ 规则 21.312 / AutoIR 30.588；AutoIR−M_blind −0.045（CI 跨 0；旧 1,120 +0.030），规则 −9.322，官方 −4.210；5 个 support 块重叠，no_overlap1275 差异 ≤0.01 | `E4/sidd/full_eval_1280_summary.json`、`full_eval_1280_bootstrap.json` |
| E4 SIDD X-Restormer | ✅ 官方 `denoise_300k.pth` zero-shot（单退化 Gaussian σ=50，非 all-in-one）；full 1,280 **31.954 / 0.7163 / 0.4213**；旧 1,120 32.014；比 M_blind +1.320（CI [+0.154,+2.558]，19/40）、比 AutoIR +1.366 | `E4/sidd/full_eval_1280_summary.json` |
| E4 SIDD 其他 4D/5D | full 1,280：PromptIR 24.325 / AirNet 23.660 / DCPT-5D 23.161 / R2R-5D 22.772 / DATPRL-5Task 16.057 / DATPRL-CDD11 13.729；全部低于 M_blind 30.634 | `E4/sidd/full_eval_1280_summary.json` |
| E4 SIDD 盲识别 | 规则：`YCrCb:5+contrast:4+sat:4` UNCERTAIN；Skill：`YCrCb:1` UNCERTAIN（拒绝全局 FP） | `E4/sidd/rule/`、`E4/sidd/skill/` |
| E4 UHDM 盲识别 | 规则：无共识 → 单对 fallback `compression_jpeg:2`，**POOR**；AutoIR：空 pipeline + POOR + out_of_library | `E4/uhdm/rule/`、`E4/uhdm/skill/` |
| E4 最终 specialist | SIDD full 1,280：规则 21.312 / AutoIR 30.588；规则过度预测造成 −9.276 dB vs AutoIR；AutoIR 相对 M_blind −0.045（不显著） | `E4/sidd/full_eval/*.jsonl`、`full_eval_1280_summary.json` |
| E4 UHDM 最终（495 对全量） | M_blind 17.307 / **官方 DFPIR 15.529** / blinded 17.254（参考）/ 规则 fallback 17.373；AutoIR 空 pipeline/POOR；相对不恢复 17.100 提升 ≤0.27 dB | `E4/uhdm/full_eval_summary.json` |
| E4 结论 | 真实数据可运行，AutoIR 避免规则式伤害并拒绝库外；但 SIDD full 1,280 上 AutoIR 未显著超 M_blind（−0.045，CI 跨 0），且低于单退化 X-Restormer 1.366 dB；官方 DFPIR 分布外失败，UHDM/Rain 无库外恢复（两者仍各排除 5 对 support，待补全） | `02_结果与逐条回应_已填写.md` E4/T6 |
| E4 Rain100H 全量 test | ✅ 95 对（100 对排除 support 1–5）：恒等 **12.104 / 0.3640 / 0.5352**；M_blind 11.840；blinded DFPIR 12.148；DFPIR official 5D 14.647；规则 16.392；PromptIR 15.470；AirNet 14.516。AutoIR 空 pipeline + POOR/out_of_library。规则 +4.288 为全局亮度补偿（ΔPSNR 与 degraded−clean 均值差 corr 0.807，LPIPS 比恒等差 0.017），不是去雨 | `E4/rain/full_eval/summary.json`、`E4/rain/README.md` |
| E4 Rain 结论 | ✅ 真实库外雨：系统可保守拒绝、库内模型不迁移；含 derain 训练的模型 +2.4~+3.4 dB；规则高 PSNR 是光度捷径；不支持库外去雨 | `02_..._已填写.md` E4/T6、`E4/rain/README.md` |
| 审计报告 | ✅ 任务卡/数据规模/训练来源/评测公平性逐项审计；发现 SIDD/UHDM support 划分、UHDM 50→495、baseline LQ 未固定、official DFPIR 随机 text 协议错误等问题 | `rebuttal_runs/AUDIT_2026-10-26.md`、`AUDIT_lq_evidence.json` |
| 固定 LQ 重评 | ✅ 7 场景非确定 LQ 已物化并重评 E1/E2；E1 差异 ≤0.02 dB，结论不变；E2 DFPIR 结果写入 `E1E2_fixed_lq_summary.json` | `E1E2_fixed_lq/`、`scripts/prepare_fixed_lq.py`、`eval_fixed_lq.py` |
| UHDM 全量 test | ✅ 495 对（500 排除 support 0–4）4 方法重评完成；M_blind 17.307 / 官方 15.529 / blinded 17.254 / 规则 17.373；不恢复 17.100 | `E4/uhdm/full_eval_summary.json` |
| P0 状态 | ✅ E1/E2/E3/E4/E5a 主结果全部完成；T0–T10 与逐条回应已填主结果 | `02_结果与逐条回应_已填写.md` |
| DFPIR-from-scratch | ✅ 24/24 训练完成；support best 23.893 / last 23.633。**Full test best 23.833/21.811（DIV2K/LSDIR）、last 23.638/21.667**；vs AutoIR −1.135/−1.145（best）、−1.330/−1.289（last）；vs ft-best −1.817/−1.804（best）、−2.012/−1.949（last）；vs original −0.161/−0.219（best）、−0.357/−0.363（last）。随机初始化 3× 预算仍低于 ft/AutoIR，接近 blinded zero-shot | `E2_scratch/scratch_full_summary.json`、`scratch_full_summary.md` |
| E2 预算曲线 | ✅ T1/T2/T3 全 24 snapshot 完成：25.251 / 25.286 / 25.366 DIV2K；23.241 / 23.295 / 23.363 LSDIR；**T/3 已超 AutoIR** | `E2/budget_curve.json` |
| T4 修正 | 旧 T1-best 子集 21.58 是选择偏差，已废弃；全 24 曲线为准 | `02_..._已填写.md` T4 |
| E6-B support-set 稳定性 | ✅ 6 场景×3 组 support×R0–R2（规则控制器，33 次训练 / 64.93 GPU-h）：DIV2K 25.000/25.304/26.064；LSDIR 23.197/23.540/24.171；range 0.13–2.66 dB，场景相关 | `E6/e6_summary.json`、`E6/e6_summary.md` |
| E6-B（LLM）support-set variation | ✅ 6 场景×3 组，完整 LLM 流程（Skill 盲识别→策略→R0→R1/R2→support 选择）：DIV2K macro **30.194** vs rule 25.456（**+4.737**，6 场景 CI [+0.341,+9.789]，4/6 胜）；LSDIR 27.662 vs 23.636（+4.026，CI [+0.183,+8.725]）；23 次训练 / 32.82 GPU-h；R1 4 次 CHANGE、R2 1 次训练 | `E6_LLM/e6llm_summary.json`、`E6-B_LLM_support_variation.md` |
| E6-A full24 run-to-run | ✅ 24 场景×2 次独立 agent（exp41 全部；同 support sha256 / 训练 seed 42）：R0 48/48、R1 9/9、R2 0；R0-only DIV2K 26.817 / within-SD **1.491** / max range 10.398；同 pipeline 10 场景 mean \|Δ\| **0.27 dB** vs 不同 pipeline 14 场景 **3.42 dB**；final **+0.837**（DIV2K）/+0.692（LSDIR），within-SD 0.856/0.724，max range 13.043（0009 单侧 R1 生效）；训练 81.12 GPU-h | `E6A_full24/e6a_full24_summary.json`、`rebuttal_plan/E6-A_full24_repeat2.md` |
| E7 exp41 两次重复 | ⏹ 已停止（用户要求停止所有实验）；Rep A 只有部分中间文件、无最终预测；**E6-A full24（24 场景×2 次）已覆盖本计划的 Phase 1/2 需求**，E7 不再重启 | `rebuttal_plan/12_exp41_repeat_randomness.md`、`13_end_to_end_cost_and_timing.md`、`E6A_full24/` |
| E7 计时/成本审计 | ✅ 现有 log 覆盖审计：训练 4.248 GPU-h/场景、support eval ~9.4 min；反思仅日期、候选/Token/API UNKNOWN；5 GPU-h 非端到端 | `E7/existing_timing_coverage.json`、`13_end_to_end_cost_and_timing.md` |
| E5c blind_0005 4-cell | ✅ 完成：no-fix 21.915、DC-only 44.908、RS-only 21.976、both 41.620（DIV2K fixed-LQ 0005）；DC +22.99，RS-only +0.06，RS at correct DC −3.29 | `E5c/`、`rebuttal_plan/E5c_blind_0005_4cell.md` |
| DFPIR-scratch test | ✅ full test 完成：best 23.833/21.811、last 23.638/21.667；见 `scratch_full_summary.md` | `E2_scratch/scratch_full_summary.*` |
| 进行中/未执行 | **E6-A full24 已完成**；E7 只有部分 Rep A 中间文件、不作结果；未执行：E5b/E2-RS 完整四格 24 场景、E3-G 留出 | E5c 只能作为 blind_0005 的 targeted RS 诊断，不替代完整 E2-RS；E4 rain、E6-B（LLM）、E6-A full24 已完成；E7 已被 E6-A full24 覆盖 |
| 旧无人值守链 | 已完成 E2 评估；E1 段因 `E1/logs` 目录缺失失败，已创建目录并另起 `run_e1_stage.sh` | `chain.log`、`E1/stage.log` |
| E1 规则控制器决策 | ✅ 24/24 完成；R1 可用 19、R2 可用 10；候选测试 196 次；arch=20 Swin+SG / 4 GCM | `E1/decisions/` |
| E1 R0/R1/R2 训练 | ✅ 24 R0 / 19 R1 / 9 R2 完成；support 选择与固定 LQ test 完成 | `E1/stage.log` |
| 无人值守链 | 历史记录；E2/E1 已由后续单独阶段完成；无活动 chain | `rebuttal_runs/chain.log`、`CHAIN_DONE` |

### E1 规则决策摘要

- 24/24 场景有效；R0 步数分布：1 步 8、2 步 7、3 步 9。
- 所有场景 `low_confidence_R0=True`；R1 候选 19 个，R2 候选 10 个。
- 架构：20 Swin+SimpleGate，4 FiLM-GCM（contrast+structured 信号）。无 OCAB（R0 未出现 anisotropy>1.7 的 motion）。
- 正确方向示例：0001 poisson 4→5；0003 gaussian→glass；0004 gaussian→motion；0005 impulse 3→5；0008 去掉假 brightness。
- 失败示例：0009/0014/0017/0018/0024 无改进候选；0006 JPEG severity 5→3 仍偏离 GT 1；0013 删除 noise 但 GT 含 speckle。

## 关键文件

- 结果填写稿：`rebuttal_plan/02_结果与逐条回应_已填写.md`
- E1 规则冻结：`rebuttal_runs/E1/rules_frozen.json`
- E1 控制器：`rebuttal_runs/scripts/rule_controller.py`
- E1 决策：`rebuttal_runs/E1/decisions/`（含 candidate_scores 与信号来源）
- E2 任务：`rebuttal_runs/E2/tasks.jsonl`
- DFPIR 评估器：`rebuttal_runs/scripts/eval_dfpir.py`
- E6-A full24 结果：`rebuttal_runs/E6A_full24/e6a_full24_summary.json` / `.md`；计划与解释：`rebuttal_plan/E6-A_full24_repeat2.md`

## 已知限制（必须在最终回应中保留）

1. **校准与测试图像重叠**：0801–0810 在 DIV2K 测试集内；不能声称独立。排除后 AutoIR DIV2K 仅 −0.005 dB，说明重叠不是性能来源，但仍需披露。
2. **Ours 31.25/27.08 是按位置微平均**，不是整链准确率；Ours 的 R0/R1 选择依据 UNKNOWN。
3. **E4 的 SIDD/UHDM 已执行（官方 DFPIR 为主基线）**：SIDD **full 1,280 块（40 场景×32，官方 validation 全量）**上 AutoIR 与 M_blind 持平（−0.045 dB，CI 跨 0；旧 1,120 为 +0.030）、规则 specialist 灾难（−9.322 dB）、DFPIR official release 5D 分布外失败（CLIP σ=25 仅 26.424，低于 M_blind 4.210）。SIDD support 取自 validation 场景 0–4（不是官方 train）；full test 含 5 个 support 块（0.39%），no_overlap1275 差异 ≤0.01 dB；旧 1,120 = 35 场景 sensitivity。UHDM support 取自 test 0–4（不是 train）；已改 full 495 对：50 子集 LQ 均值 17.752 vs 全量 17.100，M_blind 子集 17.917 vs 全量 17.307，子集高估约 0.61 dB。blinded DFPIR 31.395（旧 1,120 为 31.218）仅为项目参考，不是官方权重。**E4 specialist 只训练了 R0，没有训练后 R1/R2 反思**：训练前 Skill 候选反思有 15（SIDD）/7（UHDM）次，但任务库只有 `*_R0`，无 R1 任务、日志、params、checkpoint 或停止决策记录；R0 均正常完成、无 NaN。不是 agent 判定不需要，也不是运行崩溃，而是 E4 按最小真实实验只排了单轮 R0，训练后反思流程未调度；SIDD UNCERTAIN、UHDM POOR 按协议本应触发 R1。E4 只能标为 **R0-only**，不能写成完整三轮系统。详见 `E4/EXECUTION_NOTE.md`。**E4 rain 已执行**（Rain100H 95 对，R0-only）：不能声称库外去雨或真实数据显著提升；该雨集存在强全局亮度偏移，PSNR 会被规则的光度修正放大（见 `E4/rain/README.md`）。
4. **E1/E2 support 与 DIV2K test 重叠**：24 场景共 120 次抽取、74 张唯一 DIV2K 图，每场景 5/100；support-excluded 敏感性显示影响 <0.01 dB（E1）/0.007 dB（E2 best），但不能声称严格独立 test。
5. **LQ 协议**：7/24 场景的 `make_lq` 不可复现（impulse/glass 私有 RNG）。2026-10-27 复核：A0/E1 主指标 8,400 个共同图像行中 **2,450 行 `lq_sha256` 不一致**，正好是 **7 场景 × 350 张**（0003/0005/0017/0018/0020/0022/0023）；其余 17/24 hash 一致。**GT 管线没有变，变的是同一 GT 的随机 realization**。已用 `E1E2_fixed_lq` 同一份固定 LQ 重评：24 场景 macro/paired 变化 <0.001 dB（逐场景最大约 0.01 dB）。`LLM_vs_rule_choices.md` 的 7 个场景已改用 fixed-LQ；one-shot 的 6 个共同 fixed 场景（不含 0020）用 `eval_oneshot_fixed_lq.py` 同一份固定 LQ 重评，23 场景结论不变（rule 反思 +2.426/18 胜，LLM +1.279/15 胜）。未修正前这 7 个场景的逐图 Δ 不能当严格配对。历史 baseline 的 LQ 也未跨方法固定（18/24 随机场景）；AirNet 敏感度实测 mean +0.003/−0.009 dB。严格主表仍应统一重跑。
6. **E6-B support-set 稳定性已完成**（规则控制器，6 场景×3 组）：DIV2K macro 25.0–26.1 dB，support-set range 0.13–2.66 dB；稳定性场景相关，不能声称普遍稳定。**E6-B（LLM）完整流程已完成**：DIV2K macro 30.194 vs rule 25.456（+4.737，6 场景 CI [+0.341,+9.789]，4/6 胜）；support 稳定性同样场景相关（LLM mean range 1.269 vs rule 1.326，但 0008/0017/0024 的 range 更大）。**E6-A full24 已完成（2026-10-29）**：24 场景×2 次独立 agent（同 support sha256、同训练 seed）；R0-only DIV2K macro 26.817、within-SD 1.491 dB、max range 10.398；同 pipeline 10 场景 mean |Δ| 0.27 dB（训练/选点抖动）vs 不同 pipeline 14 场景 3.42 dB（DC 差异主导）；全流程 R1 9 次 CHANGE，final +0.837/+0.692 dB（DIV2K/LSDIR）、within-SD 0.856/0.724，但 max range 升到 13.04 dB（0009 单侧 R1 生效）。不能声称 LLM 方差普遍更小；R2 6 条全部 NO_CHANGE。**LQ 敏感性**：7 个私有 RNG 场景（0003/0005/0017/0018/0020/0022/0023）主评估 LQ 进程间不同；用 E1E2_fixed_lq 共享 LQ 重评后 mean Δ≤0.002 dB、max 0.018 dB，修正 macro 与主表一致到 3 位小数，结论不变。
7. **E2-RS 四格未执行**；RS 架构贡献未单独隔离。
8. **API model ID/版本/费用 UNKNOWN**；文档中的 DeepSeek/Qwen 只是声明。
9. **E1-oneshot 的 NaN 失败必须如实报告**：E1-oneshot（无反思规则）为 **23/24 有效、1/24 NaN**（`blind_0020` 的 GCM R0 训练 NaN）。主表是 23 个有效场景 macro，不做 0 dB 插补/静默删除；E1-full 通过 R1 fallback 恢复到 15.98，AutoIR-oneshot 15.78、AutoIR-full 18.09。Swin+SG 稳定重训只作可选 sensitivity（fixed-LQ 0020 = 14.05，仍低于 AutoIR-oneshot 15.78 / E1-full 15.98），不冒充原始 one-shot。
10. **RS 规则来源与 24 场景独立性**：核心映射 contrast→GCM / motion→OCAB 来自 exp10/38/39 的 GT-known 合成实验，不使用 24 个盲场景；但盲场景门控/回退规则来自 exp37 的 24 挑战消融（与 exp40/41 的 24 个 `degraded.png`/GT **24/24 相同**）和 exp40 的 4 个案例，因此 **24 个场景全部参与过 RS 规则开发**，不是 RS 泛化的独立测试集。GT 校准侧：校准脚本早于 24 个 GT 生成，GT 由随机生成器产生，无按性能挑选记录；但 **3/24 GT 链是校准配置集成员**（0006/0011/0013）、10/100 DIV2K test 图为校准图、3840 manifest 缺失，不能声称严格独立。723+ 训练去重清单缺失（raw 日志 1,137 条；组数相加 724）。详见 `rebuttal_plan/RS_rule_provenance_audit.md`、`03` 2026-10-28 补充。

12. **E6-A full24 泄露/口径审计（2026-10-30）**：E6-A support 全部来自 exp41（DIV2K_valid test；74/100 唯一图），E1/AutoIR 同源；排除本场景 5 张 support 后 R0 26.817→26.819、final 27.654→27.659，重叠不抬高 test。E6-B 98 个 Agent 0 次禁读；E6-A 100 个中仅 blind_0001_rep2 读到其他场景 GT 提示（非自身 GT），剔除影响约 −0.06 dB。E6-A 高于 rule 的 mean +2.54 但 median +0.65，增益集中在规则 DC 灾难场景；rep2 GT 整链 15/24 vs 规则 5/24。详见 `E6A_full24/AUDIT.md`。

11. **端到端成本与 timing（Weakness 4）**：5 GPU-h/场景只是 R0–R2 训练预算的近似；实测训练 4.248 GPU-h/场景（71 次/101.95 GPU-h），support eval ~9.4 min。现有 exp41 日志缺 R1/R2 反思时刻、DC 候选评估计时、策略选择时间、API token/费用、最终 test start/end；只能恢复训练与部分评测，其余标 UNKNOWN，不回溯编造。离线开发（723+ 规则/校准/DFPIR 预训练）必须单列，不能摊入 5h。E7 起强制 `timing.json`（DC/RS/CLR 每轮），模板 `E7/timing_schema.json`、wrapper `scripts/timing_wrapper.sh`；规范见 `13_end_to_end_cost_and_timing.md`。

## GPU 效率结论

- 之前“功耗不满”的原因是：0.45M RestoreNet + batch=1 逐图全分辨率评估 + CPU 在线退化；RFA/规划阶段是纯 CPU，GPU 空闲属正常。
- 现在 E2（31M DFPIR，8 卡并行）实测 **95–100% 利用率、292–299W / 300W**，说明 GPU 本身可被喂满；小模型训练/逐图评估的瓶颈在 CPU 数据管线与串行评估，不在 GPU 算力。
