# AutoIR AAAI 审稿回应执行包

日期：2026-10-24。范围：依据本机可读代码、实验报告及用户提供的六份评审制定；本次没有启动 GPU 训练，也没有将历史汇总视为已验证的最终结果。

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
