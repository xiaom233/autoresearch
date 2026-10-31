# E6-C support-size 盲识别子 Agent 指令

你的挑战目录在 prompt 中给出。目录含 **k 对** `clean_i.png` / `degraded_i.png`（k=1 或 3；由目录名 `size1`/`size3` 与 prompt 指定；`clean.png`/`degraded.png` 是 pair0 别名）。退化步数、类型、严重度未知，必须完全依据信号分析自行判断。

## 强制流程

1. 开始前 Read 全文 `/data/zyli/projects/autoresearch/.claude/skills/image-degradation-simulator/SKILL.md`，严格按协议执行：逐层剥离、信号驱动探索、强制检查清单、预算 ≤25 次 `test_candidate.py`、3 轮硬终止；噪声/模糊判定前按 SKILL 要求重读对应行段。
2. 只允许使用以下脚本（解释器 `/data/zyli/projects/autoresearch/.venv/bin/python3`）：
   - `.claude/skills/image-degradation-simulator/scripts/run_full_analysis.py`
   - `.../peeled_noise_check.py`
   - `.../test_candidate.py`
   禁止其他分析脚本；禁止写 Python 脚本/子进程脚本；禁止 `run_in_background`。
3. 以 pair0 为主分析；最终候选必须在**本目录全部 k 对** support 上复核（k=1 时只有 pair0，需在 reflection 中标注“单 support，无法做跨对验证”）。
4. Qwen/API 不可用：若 SKILL Step 0.5 门控触发，跳过并在 reflection.json 中记录 `qwen_gate={"available": false}`。
5. 在挑战目录内保存：
   - `full_analysis.json`（pair0；其余可命名 `full_analysis_p{i}.json`）
   - `predicted_params.json`：`{"challenge_id": "<prompt 中给出的 ID>", "pipeline": [{"function": "...", "severity": N, "verdict": "..."}], "alternatives": [...], "analysis": {"verdict": "LIKELY|UNCERTAIN|POOR", "tier": N, "ci_pass_rate": "...", "mode": "...", "rationale": "..."}}`
   - `reflection.json`：含 `initial_analysis`、`noise_6step_check`（含噪声时 6 项数值）、`iterations`、`psnr_ranking`、`final_decision`
   - `agent_timing.json`：`{"dc_start_utc","dc_end_utc","wall_seconds","candidate_tests","api_calls":null,"tokens":null,"notes"}`
6. 禁止读取（即使可读）：
   - `/data/zyli/projects/autoresearch/exp41/` 下任何文件（含 `degradation_gt/`、`challenges/`）
   - `.gt_mappings/`、`.ground_truth/`、任何 `degradation_gt` 目录
   - `rebuttal_runs/E6C_support_size/` 下除**本挑战目录**与**本指令文件**以外的任何文件（尤其 `support_plan_gt.json`、其他 `challenges/`、`logs/`、`experiments/`、`selection_*`）
   - `E6B_full24/`、`E6A_full24/`、`E6_LLM/`、`E6/`、`E6_A/`、`E6A_repeat/` 等任何其他实验目录（**禁止用其他 size 或 setA 的其余图像补足 support**）
   - 任何 `test_metrics*`、`results/`、训练日志
7. 完成前确认两个 JSON 可解析。最终回复 ≤120 字：pipeline、verdict、test_candidate 次数、k 对验证结果。
