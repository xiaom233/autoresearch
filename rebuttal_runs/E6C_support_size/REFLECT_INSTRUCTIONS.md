# E6-C support-size R1/R2 反思子 Agent 指令

你的挑战目录、轮次（R1/R2）、上一轮 pipeline 与触发原因由 prompt 给出。目录含 **k 对** support（k=1 或 3；由目录名 `size1`/`size3` 指定）。退化仍未知，只能依据信号与 support 验证自行判断。**反思对象是信号重释，不是 R0 模型残差诊断。**

## 强制流程

1. 开始前完整 Read `/data/zyli/projects/autoresearch/.claude/skills/image-degradation-simulator/SKILL.md`；噪声/模糊判定前按 SKILL 重读对应行段。
2. 只允许使用：`run_full_analysis.py`、`peeled_noise_check.py`、`test_candidate.py`（`--mode signal` 用于含噪声候选，`--mode psnr` 用于纯确定性候选）。禁止其他分析脚本；禁止写脚本/子进程；禁止 `run_in_background`。
3. 允许读取：本挑战目录的 `degraded_*.png` / `clean_*.png`（共 k 对）、`full_analysis.json`、`peeled_noise.json`、`predicted_params.json`、`reflection.json`、`r0_result.json`；上一轮 `predicted_params_{R1}.json`（R2 时）；本指令文件。
4. 信号重释：
   - 重跑 pair0：`run_full_analysis.py --target degraded.png --clean clean.png --output full_analysis_R1.json`
   - 逐项检查被忽略/误判/阈值边界的信号（variance_ratio、contrast_spread_ratio、mean_shift、clipped_fraction、residual_intensity_corr、block_boundary、unique_G、gm_ratio、spectral_slope、anisotropy、glass_score、noise sigma/§B 六项）
   - 每条修正假设必须注明信号来源与数值；禁止用 R0 模型 residual 反推退化类型
5. 候选生成与验证（≤5 个新候选，≤15 次 `test_candidate.py`）：
   - 纯确定性候选：在**本目录全部 k 对**上做 PSNR 验证；`>40 dB` 或相对上一轮 `+2 dB` 才可 CHANGE
   - 含 noise 候选：用 `--mode signal` 在 k 对上比较 verify_signals；多数对由 WEAK→PARTIAL/MATCH 才可 CHANGE；禁止仅凭 PSNR 接受/拒绝噪声候选
   - **k=1 时只有 pair0**：单对验证噪声大，必须记录该限制；阈值不变（不因 k=1 放宽 +2 dB），证据不足就 NO_CHANGE
   - 3 轮信号重释硬终止；无改善 → `NO_CHANGE`
6. 保存（全部写在本挑战目录）：
   - `predicted_params_R1.json`（R2 时 `_R2`）：`{"challenge_id","round","decision":"CHANGE|NO_CHANGE","r0_pipeline":[[fn,sev],...],"pipeline":[...],"alternatives":[...],"analysis":{...}}`
   - `r1_decision.json`（R2 时 `r2_decision.json`）：`{"challenge_id","round","decision","r0_pipeline","r1_pipeline","support_psnr_r0","candidates_tested","test_candidate_calls","best_validation","signal_sources","reason"}`
   - `agent_timing_R1.json`（R2 时 `agent_timing_R2.json`）
   - 在 `reflection.json` 追加 `reflection_R1`（或 `reflection_R2`）
   - **绝不修改 R0 的 `predicted_params.json`**；NO_CHANGE 时 R1 文件 pipeline 与上一轮逐项相同，decision 写 NO_CHANGE
7. 禁止读取（即使可读）：
   - `/data/zyli/projects/autoresearch/exp41/` 下任何文件
   - `.gt_mappings/`、`.ground_truth/`、任何 `degradation_gt`
   - `E6C_support_size/` 下除**本挑战目录**与**本指令文件**外的任何文件（尤其 `support_plan_gt.json`、其他 `challenges/`、`logs/`、`experiments/`、`selection_*`）
   - `E6B_full24/`、`E6A_full24/`、`E6_LLM/`、`E6/`、`E6_A/`、`E6A_repeat/` 等任何其他实验目录（禁止用其他 size/setA 其余图像补足 support）
   - 任何 `test_metrics*`、`results/`、训练日志
8. 完成前确认 JSON 可解析。最终回复 ≤120 字：轮次、decision、pipeline、test_candidate 次数、k 对 support 验证摘要。
