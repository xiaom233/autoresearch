# E6-B full24 R1/R2 反思子 Agent 指令

你的挑战目录、轮次（R1/R2）、上一轮 pipeline 与触发原因由 prompt 给出。退化步数、类型、严重度仍未知，只能依据信号与 support 验证自行判断。**本次反思的对象是信号重释，不是 R0 模型残差诊断。**

## 强制流程

1. 开始前完整 Read `/data/zyli/projects/autoresearch/.claude/skills/image-degradation-simulator/SKILL.md`，严格按协议执行；噪声/模糊判定前按 SKILL 要求重读对应行段。
2. 只允许使用以下脚本（解释器 `/data/zyli/projects/autoresearch/.venv/bin/python3`）：
   - `.claude/skills/image-degradation-simulator/scripts/run_full_analysis.py`
   - `.../peeled_noise_check.py`
   - `.../test_candidate.py`（`--mode signal` 用于含噪声候选，`--mode psnr` 用于纯确定性候选）
   禁止其他分析脚本；禁止写 Python 脚本/子进程脚本；禁止 `run_in_background`。
3. 允许读取的输入：
   - 本挑战目录的 `degraded_*.png` / `clean_*.png`（5 对 support，`*_0` 为主分析对）
   - 本挑战目录的 `full_analysis.json`、`peeled_noise.json`、`predicted_params.json`、`reflection.json`、`r0_result.json`（`r0_result.json` 只提供 support PSNR，不含 GT）
   - 上一轮 `predicted_params_{R1}.json`（做 R2 时）
   - 本指令文件
4. 信号重释（不是 residual 诊断）：
   - 重跑 pair0：`run_full_analysis.py --target degraded.png --clean clean.png --output full_analysis_R1.json`
   - 逐项检查被忽略/误判/阈值边界的信号（variance_ratio、contrast_spread_ratio、mean_shift、clipped_fraction、residual_intensity_corr、block_boundary、unique_G、gm_ratio、spectral_slope、anisotropy、glass_score、noise sigma/§B 六项）
   - 每条修正假设必须注明信号来源与数值，例如 “variance_ratio=0.72 + mean_shift≈0 → contrast_weaken 而非 brightness”
   - 禁止用 R0 模型 residual 反推“缺少哪种退化”
5. 候选生成与验证（≤5 个新候选，≤15 次 `test_candidate.py`）：
   - 纯确定性候选：在 **5 对 support** 上做 PSNR 验证；判定 `>40 dB` 或相对上一轮 `+2 dB` 才可 CHANGE
   - 含 noise 候选：用 `--mode signal`，在 **5 对 support** 上比较 verify_signals；多数对由 WEAK→PARTIAL/MATCH 才可 CHANGE；禁止仅凭 PSNR 接受/拒绝噪声候选
   - 每一步都要写 `--thinking`（信号来源）；`--reflection` 指向本挑战 `reflection.json`
   - 3 轮信号重释硬终止；无改善 → `NO_CHANGE`
6. 保存（全部写在本挑战目录）：
   - `predicted_params_R1.json`（R2 时 `predicted_params_R2.json`）：`{"challenge_id": "...", "round": "R1", "decision": "CHANGE|NO_CHANGE", "r0_pipeline": [[fn,sev],...], "pipeline": [{"function","severity","verdict"}...], "alternatives":[...], "analysis":{...}}`
   - `r1_decision.json`（R2 时 `r2_decision.json`）：`{"challenge_id","round","decision","r0_pipeline","r1_pipeline","support_psnr_r0","candidates_tested","test_candidate_calls","best_validation","signal_sources","reason"}`
   - `agent_timing_R1.json`（R2 时 `agent_timing_R2.json`）：`{"dc_start_utc","dc_end_utc","wall_seconds","candidate_tests","api_calls":null,"tokens":null,"notes"}`
   - 在 `reflection.json` 追加 `reflection_R1`（或 `reflection_R2`）键：信号重释列表、候选排名、support 验证结果、最终决策
   - **绝不修改 R0 的 `predicted_params.json`**；NO_CHANGE 时 R1 文件 pipeline 与上一轮逐项相同，但 decision 写 NO_CHANGE
7. 禁止读取（即使可读）：
   - `/data/zyli/projects/autoresearch/exp41/` 下任何文件
   - `.gt_mappings/`、`.ground_truth/`、任何 `degradation_gt` 目录
   - `rebuttal_runs/E6B_full24/` 下除**本挑战目录**与**本指令文件**以外的任何文件（尤其 `support_plan_gt.json`、其他 `challenges/`、`logs/`、`experiments/`、`selection_*`）
   - `E6A_full24/`、`E6_LLM/`、`E6/`、`E6_A/`、`E6A_repeat/` 等任何其他实验目录
   - 任何 `test_metrics*`、`results/`、训练日志
8. 完成前确认两个 JSON 可解析。最终回复 ≤120 字：轮次、decision、pipeline、test_candidate 次数、5 对 support 验证摘要。
