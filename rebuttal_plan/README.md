# AutoIR AAAI 审稿回应执行包
> **2026-10-27 执行更新**：A0/A1、E1、E2、E3、E4、E5a、E6-B 已完成并写入 `02_结果与逐条回应_已填写.md`；E6-B 为规则控制器 6 场景×3 组 support（33 次训练、64.93 GPU-h，support-set range 0.13–2.66 dB，场景相关）。新增 **E1-oneshot 无反思对照**（23/24 有效场景 DIV2K）：rule 22.206→24.632（+2.43，18/23 胜），AutoIR 23.989→25.268（+1.28，15/23 胜）；LLM full vs rule full +0.636（CI 跨 0）。E1-oneshot 在 blind_0020 的 R0 训练 NaN，按 **1/24 terminal failure** 直接报告；E1-full 通过 R1 fallback 恢复，不静默删除。**LQ 配对审计**：A0/E1 原始指标在 7 个非确定性场景的 LQ hash 不一致（2,450/8,400 行 = 7×350），GT 管线未变；已用 `E1E2_fixed_lq` 的同一份固定 LQ 重评，LLM vs rule 表和 one-shot 已更新；17/24 场景 hash 一致。E6-A（agent run-to-run）正在训练（18 个 R0-only，结果未出）；**E4 rain 已完成**（Rain100H 95 对：AutoIR 空 pipeline/POOR、恒等 12.104；DFPIR-official 14.647、PromptIR 15.470、AirNet 14.516；规则 16.392 但为全局光度补偿，非去雨）；E3-G、E2-RS、E5b 仍未执行；E5c blind_0005 targeted 4-cell 与 DFPIR-scratch full test 已完成。下方“本次没有启动 GPU 训练”是 2026-10-24 初始审计状态，已不再适用；当前产物见 `rebuttal_runs/`。

日期：2026-10-24。范围：依据本机可读代码、实验报告及用户提供的六份评审制定；本次没有启动 GPU 训练，也没有将历史汇总视为已验证的最终结果。

**blind_0003 详细案例**：`case_blind_0003/report.md`（逐轮规则/LLM 归因、原始数据/日志索引、checkpoint 口径审计、E1 规则控制器对比）。

**Controller 层对比**：`controller_comaprison.md`（rule-based controller 的 7 层构造、冻结阈值、R0→R1→R2 状态机、LLM vs rule 对比、blind_0003 与 24 场景结果）。

**blind_0005 / CLR 失败复盘**：`case_blind_0005/report.md`（impulse-noise 漏识别、brightness 假阳性、severity 3 vs 5、CI 饱和、R3/R4 未训练、规则控制器同 restorer 44.92 dB 对照；回答审稿人问题 2）。

**blind_0020 / LLM vs 规则逐步决策**：`case_blind_0020/report.md`（规则 R0 NaN、R1 fallback 15.991；LLM R2 发现 near-threshold glass，18.084；同架构纯 DC 对比 +0.98 dB；回答 Reviewer 3 的 agent 独立贡献问题）。

**无反思 one-shot 对照**：`E1_oneshot_no_reflection.md`（E1-oneshot / AutoIR-oneshot vs 各自 full：规则反思 +2.43 dB，LLM 反思 +1.28 dB；LLM 反思平均不优于规则反思，详见文档）。

**LLM vs rule 逐场景选择对比**：`LLM_vs_rule_choices.md`（24 场景的 GT / LLM 选中 / rule 选中 / 差异类型 / GT match / support 与 test 指标；含初始 R0 表）。

**E5c blind_0005 4-cell**：`E5c_blind_0005_4cell.md`（DC-only +22.99 dB、RS-only +0.06 dB、RS at correct DC −3.29 dB；targeted，不是完整 E2-RS）。

**DFPIR-scratch**：best DIV2K fixed-LQ 23.833 vs AutoIR 24.968（−1.135）、DFPIR-ft-best 25.650（−1.817）、original 23.995（−0.161）；last/LSDIR 仍在补。

**剩余实验状态**：`05_剩余实验状态.md`（已完成/进行中/未执行、DFPIR-scratch 结果、E5c 进度、E4 rain 结果、E6-A 进行中、E3-G/E2-RS/E5b 未执行原因与优先级）。

最新修订先读 `04_E1-E4细化任务卡.md`：按exp41固定24场景、三次训练预算DFPIR、组合数量计数与真实/库外选择重新定义E1–E4。`count_degradation_space.py`及结果JSON提供可复算的组合计数；旧audit_local.json保留为此前审计快照。

按以下顺序使用：

1. 阅读 `01_实验执行计划.md`：证据、待查设置、实验协议、预算与启动条件。
2. 在 `02_结果与逐条回应模板.md` 填表：所有结果位置保留待填，禁止用预期结果代替实测。
3. 复制 `experiment_manifest.example.json` 为实际清单，填写路径及协议；每项确认后才将对应门禁设为 true。
4. 运行 `python audit_and_validate.py audit --paper-root <AutoIR材料目录> --out <新报告路径>`，复核已有材料。
5. 运行 `python audit_and_validate.py validate --manifest <实际清单>`，检查训练前门禁、数据划分、文件和预算。此工具不启动训练，不证明数据内容没有泄露。

本机可直接运行的例子（PowerShell，在本目录）：

```powershell
python .\audit_and_validate.py audit --paper-root 'C:\Users\Zheyuan\Desktop\AutoIR' --out .\audit_local.json
python .\audit_and_validate.py validate --manifest .\experiment_manifest.example.json
```

示例清单应当校验失败，因为路径、历史审计和预算还没有确认。历史代码实际位置是 `C:\Users\Zheyuan\Documents\GitHub\autoresearch`；`C:\Users\Zheyuan\Documents\autoresearch` 本次只看到 `.git`，不是训练工作目录。本包现已移入 autoresearch/rebuttal_plan；未修改原训练代码与实验。

训练服务器原脚本使用 Linux/CUDA 路径；不能将本包当成已部署的一键训练系统。执行计划明确区分“已有入口”和“需补代码”。


2026-10-26 新增：

- 05_审稿人逐条回应_英文草稿.md：对应5位审稿人的22条Weakness回答及12条独立Specific Points回应。
- 06_证据核对与提交前缺项.md：事实核对、旧填写稿冲突与未完成证据。
- verify_response_evidence.py / response_evidence_check_2026-10-26.json：本次已运行的数值一致性核对。

- 05已全文精简重写：完整保留审稿原文，Weakness与Specific Points分开逐条回答。
- 07_审稿意见对齐审校记录.md：记录引文核对、修正的问题归属及尚缺证据。
- 10_Specific_Points逐问回答.md：单独汇总11条具体提问及osqx指向Weakness的总指示，并逐条作答，与05一致。
