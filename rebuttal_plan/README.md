# AutoIR AAAI 审稿回应执行包
> **2026-10-27 执行更新**：A0/A1、E1、E2、E3、E4、E5a、E6-B 已完成并写入 `02_结果与逐条回应_已填写.md`；E6-B 为规则控制器 6 场景×3 组 support（33 次训练、64.93 GPU-h，support-set range 0.13–2.66 dB，场景相关）。E6-A（agent run-to-run）、E3-G、E4 rain、E2-RS、E5b、E5c 仍未执行。下方“本次没有启动 GPU 训练”是 2026-10-24 初始审计状态，已不再适用；当前产物见 `rebuttal_runs/`。

日期：2026-10-24。范围：依据本机可读代码、实验报告及用户提供的六份评审制定；本次没有启动 GPU 训练，也没有将历史汇总视为已验证的最终结果。

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
