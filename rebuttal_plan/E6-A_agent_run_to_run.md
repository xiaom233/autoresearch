# E6-A：同一 support / 训练 seed 下的 Agent 重复运行

> **状态（2026-10-29）**：已按用户要求**停止所有实验**。E6-A 只完成 7 个 R0 + 部分第二批训练，
> **没有完成 18 个 run，没有 E6-A 结果**；本文保留设计、预测表与 partial 观察，不作为结果引用。

> 目的：补 E6-A（agent run-to-run）。六个预注册场景，每个场景用**同一组 5 对 support**（逐文件 sha256 相同），
> 由 agent 独立做 3 次盲识别，再用**完全相同的训练配置与 seed** 各训练一次 R0 specialist，
> 在**同一份 GT-LQ test**（DIV2K 100 / LSDIR 250）上评估，从而分离盲识别本身的 run-to-run 方差。
>
> 范围限制：E6-A 是 **R0-only**（不做训练后 R1/R2 反思）；因此它度量的是“盲识别 + 单轮 R0”的方差，
> 不是完整 AutoIR R0–R2 流程的方差。E6-B 是规则控制器的 support-set 稳定性（不同 support、含 R0–R2），
> 两者不能互相替代。

## 1. 设计

| 项 | 设置 |
|---|---|
| 场景 | `blind_0006/0008/0010/0013/0017/0024` |
| support | 每场景 5 对；3 次重复使用同一组（已验证 degraded_*.png / clean_*.png sha256 完全一致） |
| 盲识别 | 每次独立 agent run；输出 `predicted_params.json` / `reflection.json` / `full_analysis.json` |
| 训练 | Direct，Swin+SimpleGate，0.42M 参数，2 epoch / 15,094 steps，batch 16，LR 5e-4，seed 42 |
| checkpoint 选择 | 每 3,773 步存点，用 5 对 support 的 PSNR 选 best（与 E6-B 相同协议） |
| test | GT pipeline 生成的固定 LQ：DIV2K 100 + LSDIR 250；`eval_checkpoint.py test --seed-salt v1` |
| 指标 | RGB PSNR（主）；SSIM 一并记录；E6-B 主表只用 PSNR |
| GT 用途 | 仅评估与事后匹配统计，不进入盲识别/训练 |

LQ 协议与 E6-B 对齐已验证：`blind_0006` / `DIV2K/0801` 的 `lq_sha256` 在 E6-A 复算与 E6-B 记录中完全相同
（`37065df8...`）。

## 2. Agent 预测（18 次独立 blind ID）

GT 仅用于事后对比：

| 场景 | GT pipeline | rep1 | rep2 | rep3 |
|---|---|---|---|---|
| blind_0006 | jpeg:1 | jpeg:1 ✅ | jpeg:1 ✅ | jpeg:1 ✅ |
| blind_0008 | saturate_weaken_HSV:4 | saturate_weaken_HSV:4 ✅ | saturate_weaken_HSV:4 ✅ | saturate_weaken_HSV:4 ✅ |
| blind_0010 | blur_gaussian:1 + brightness_darken_gamma_HSV:5 | 同 GT ✅ | gaussian:1 + darken_gamma_RGB:5 + sat_weaken_YCrCb:2 ❌ | 同 GT ✅ |
| blind_0013 | speckle:3 + jpeg:2 | jpeg:4 + speckle:1 ❌ | impulse:3 + jpeg:2 ❌ | speckle:3 + jpeg:2 ✅ |
| blind_0017 | impulse:5 + blur_gaussian:3 + jpeg:4 | gaussian:3 + jpeg:4（漏 impulse）❌ | 同 rep1 ❌ | glass:3 + jpeg:4（漏 impulse）❌ |
| blind_0024 | gaussian_RGB:4 + blur_gaussian:3 + sat_strengthen_HSV:2 | gaussian_YCrCb:3 + blur:3 + sat:2 ❌ | 同 rep1 ❌ | speckle:1 + blur:2 + sat:3 ❌ |

- 完全匹配（函数+严重度+顺序）：**9/18**。
- 稳定场景：0006、0008（3/3 完全一致）。
- 不稳场景：0013（三种不同答案，仅 rep3 完全正确）、0017（3/3 漏掉 impulse）、0024（亚型/严重度在 rep 间漂移）。
- agent 输出的 `verdict` 与 `ci_pass_rate` 见各 `predicted_params.json`；UNCERTAIN 不等于错误，但 0013/0017/0024 的多个
  UNCERTAIN 预测确实互不一致。

## 3. 结果（test，R0-only）

> 待 18 个训练与测试完成后填写。本节只写实际数值，不预填结论。

## 4. 解释与限制

- 固定 support + 固定训练 seed **不能保证完全确定**：CUDA/cuDNN 与 5 对 support 的 val 会让 support PSNR 曲线略有差异，selected step 可能不同。`blind_0008` 三次预测完全相同（`saturate_weaken_HSV:4`），但 selected steps 为 15092/7546/15092，DIV2K test 为 **43.251/41.334/42.709（range 1.92 dB）**——这是训练/选点抖动，不是 agent 方差。因此 E6-A 的 spread 是 **agent 预测差异 + 训练/选点抖动** 的混合；要单独度量 agent 方差，需要固定 selected step（last-step sensitivity）。
- E6-A 不是完整系统：没有训练后 R1/R2 反思；不能把它当作 AutoIR 全流程的 run-to-run 方差。
- 训练/选点抖动与 agent 预测差异混在一起：0008（三次相同 pipeline）给出约 1.92 dB 的 DIV2K 抖动参考；补充的 last-step（step 15092，所有 run 固定同一 checkpoint 步）评估用于分离该抖动。
- API model ID/版本/温度等未记录（`UNKNOWN`），因此 E6-A 的方差是“该 agent 配置下”的方差，不能外推到其他 LLM。
- E6-B 的 support-set 变化（0.13–2.66 dB range）与 E6-A 的盲识别方差不属于同一来源，应分开报告。
- 不能据此声称“LLM 方差大于/小于规则控制器”；E6-B 没有做 agent 重复。

## 5. 产物

```text
rebuttal_runs/E6_A/challenges/blind_XXXX_repR/        # 5 support pairs + clean/degraded + agent 输出
rebuttal_runs/E6_A/experiments/blind_XXXX_repR/       # checkpoints
rebuttal_runs/E6_A/logs/blind_XXXX_repR.log           # 训练日志（含 EXP_META / best step）
rebuttal_runs/E6_A/tasks.jsonl                        # 18 个训练任务
rebuttal_runs/E6_A/selection.jsonl                    # best checkpoint + support PSNR + 预测
rebuttal_runs/E6_A/test_metrics_gpu*.jsonl            # 逐图 test
rebuttal_runs/E6_A/e6a_summary.json / .md             # 聚合
rebuttal_runs/scripts/e6a_select.py
rebuttal_runs/scripts/aggregate_e6a.py
```
