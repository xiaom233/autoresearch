# E4 rain：真实配对去雨（Rain100H）上的库外实验

日期：2026-10-28。目的：用成熟真实配对去雨测试集检验 AutoIR/规则控制器在**库外退化**上的行为，
不训练任何真实雨数据（除规则 specialist 用预测 DC 在 DIV2K 合成上训练）。

## 1. 数据

- 来源：`dronefreak/Rain13K` → `test/Rain100H`（HuggingFace 下载，原始文件在 `datasets/E4/rain/raw/test/Rain100H/{input,target}`）。
- 共 100 对；support 取 stem `1–5`（5 对），test 为剩余 **95 对**，按 sha256 校验 support/test 无重叠。
- 图像尺寸：61 张 480×320、34 张 320×480。
- 清单：`datasets/E4/rain/support_manifest.json`、`test_manifest.json`（逐文件 sha256）。
- 说明：support 与 test 来自同一 Rain100H test 池但互不重叠；没有使用 Rain13K 训练集，也没有用真实雨对做监督训练。

## 2. 方法

| 方法 | 说明 | 训练 |
|---|---|---|
| AutoIR（Skill 盲识别） | `image-degradation-simulator` 子 Agent 在 5 对 support 上做库内候选搜索，判定 `out_of_library` / `POOR`，输出**空 pipeline** | 0 |
| 规则控制器 + R0 specialist | 现有规则控制器（无新算子/阈值）在真实 support 上聚合出 `brightness_brighten_shfit_RGB:4 + contrast_strengthen_scale:4`，`UNCERTAIN`；Direct Swin+SG，2 epoch / 15,094 steps / seed42 / 1.5 GPU-h，训练退化=预测 DC（DIV2K train 合成），support 仅作 val | 1 次 R0 |
| M_blind | `resource/blind_pretrain/checkpoints/blind_pretrain_step15092.pt`，盲训练 0.42M RestoreNet | 0（复用） |
| DFPIR-official 5D | `benchmark/DFPIR/DFPIR-5D-...pth.tar`，官方 5D all-in-one；CLIP prompt 使用官方 derain prompt：`"Rain degradation with rain lines"`；tile 1024（单图） | 0 |
| DFPIR-blinded | `resource/.../dfpir_blind/checkpoints/dfpir_blind_step301920.pt`，项目盲训练 DFPIR | 0（复用） |
| 参考：PromptIR / AirNet | 官方权重 `benchmark/PromptIR/ckpt/model.ckpt`、`benchmark/AirNet/ckpt/All.pth`；两者训练任务均包含去雨，作 in-domain 参考 | 0 |

推理：RestoreNet 全图（≤480px）/ tile512+overlap32；DFPIR-blinded tile512+overlap64；官方 DFPIR pad16 / tile1024。
指标：RGB PSNR / SSIM(11×11 Gaussian σ=1.5) / VGG-LPIPS；逐图行在 `full_eval/*.jsonl`。

规则控制器在 support 上的决策记录：`rule_new/e4_rule_decision.json`（brightness 5/5、contrast 3/5 投票，
pipeline 拟合 PSNR 12.60；Skill 的 support 恒等 PSNR 均值为 ~12.5，best legal candidate 17.50 被拒绝）。

## 3. 结果（95 对）

| 方法 | PSNR | SSIM | LPIPS | ΔPSNR vs 恒等 | 95% CI | 胜/95 |
|---|---:|---:|---:|---:|---:|---:|
| 恒等（不恢复；AutoIR 空 pipeline） | **12.104** | — | — | — | — | — |
| M_blind | 11.840 | 0.3421 | 0.5916 | −0.264 | [−0.294, −0.235] | 5 |
| DFPIR-blinded（项目） | 12.148 | 0.3586 | 0.5463 | +0.044 | [+0.037, +0.050] | 91 |
| DFPIR-official 5D（含 derain 训练） | 14.647 | 0.4811 | 0.4595 | +2.543 | [+2.177, +2.936] | 95 |
| 规则控制器 + R0 specialist | _待训练完成后填入_ | | | | | |
| 参考：PromptIR（含 derain 训练） | 15.470 | 0.5059 | 0.4446 | +3.630 | [+3.210, +4.074] | 95 |
| 参考：AirNet（含 derain 训练） | 14.516 | 0.5011 | 0.4534 | +2.677 | [+2.351, +3.034] | 95 |

## 4. 初步结论

1. **AutoIR 在库外雨上正确拒绝**：Skill 判定 `out_of_library`/`POOR`，输出空 pipeline（= 恒等 12.104），
   没有把 X-Distortion 库内函数强行套到真实雨图上。
2. **库内盲训练模型不迁移**：M_blind 比不恢复还低 0.26 dB；项目 blinded DFPIR 仅 +0.04 dB。
   两者的训练库都不含雨，输出接近恒等/轻微破坏。
3. **含 derain 训练的模型能提升，但依然有限**：官方 5D DFPIR +2.54 dB，PromptIR +3.63 dB、AirNet +2.68 dB，
   且 95/95 全胜；但这仍远低于去雨专用模型的常见水平，说明该 Rain100H 版本雨密度很高/分布差异大，
   不能据此声称“通用模型已解决真实去雨”。
4. **规则控制器的风险**：它在 support 上强行给出 brightness+contrast（非雨），
   拟合 PSNR 12.60 仅略高于恒等；这正是 AutoIR 空 pipeline 想避免的“库内过度拟合”。

## 5. Caveats

- 只用了一个去雨测试集（Rain100H / Rain13K 版本），没有 Rain100L/Rain800 等其他库外雨数据。
- 官方 DFPIR 的官方 derain 评测集是 Rain100L（轻雨）；这里是 Rain100H（重雨），且用 RGB PSNR。
- 官方 DFPIR/PromptIR/AirNet 的训练数据与 Rain100H test 是否存在图像级重叠**未审计**（官方权重，训练集不在本地）。
- 没有训练真实雨数据；规则 specialist 的训练退化是预测 DC（brightness+contrast），不是雨。
- 规则模型为 R0-only，无训练后 R1/R2 反思（与 E4 SIDD/UHDM 相同的执行范围限制，见 `E4/EXECUTION_NOTE.md`）。
- support 5 对取自同一 Rain100H test 池（不重叠），不是独立训练分布。

## 6. 产物

```text
datasets/E4/rain/{support,test}/                     # 真实配对
datasets/E4/rain/skill_challenge/                    # Skill 盲识别工作目录（空 pipeline / POOR）
rebuttal_runs/E4/rain/rule_new/                      # 规则决策 + predicted params
rebuttal_runs/E4/rain/logs/rule_R0.log               # 规则 specialist 训练日志
rebuttal_runs/E4/rain/rule_R0/                       # checkpoints
rebuttal_runs/E4/rain/full_eval/*.jsonl              # 逐图评估
rebuttal_runs/E4/rain/full_eval/summary.json         # 聚合（含 bootstrap CI）
rebuttal_runs/scripts/aggregate_e4_rain_full.py
rebuttal_runs/scripts/run_e4_rain_rule_eval.sh
```
