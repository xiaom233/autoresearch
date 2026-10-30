# E6-A full24 泄露与口径审计（2026-10-30）

## 1. 问题

E6-A full24 的 DIV2K macro 为 **26.817 dB（R0-only best）/ 27.654 dB（final）**，高于 E1 规则 24.272、AutoIR 24.968，也高于 exp41 R4 24.44。需要审查：
1. support/test 是否重叠、是否抬高 test；
2. Agent 是否读取了禁止路径/GT；
3. 与 LLM/rule-based 实验的差异是否合理。

## 2. support 来源与 test 重叠

- E6-A full24 的 support 直接复制自 exp41 challenge support；对 clean 图做像素级 hash 比对：
  - **240 个 support 文件（74 个唯一像素 hash）全部匹配 DIV2K_valid_HR（test）**；
  - 0 个匹配 DIV2K_train_HR；0 个匹配 LSDIR val。
- 因此每个场景的 5 张 support 都是该场景 100 张 DIV2K test 中的 5 张；24 场景共覆盖 **74/100** 张 test 图。
- E1/AutoIR 训练同样使用 exp41 challenge support（`AR_CHALLENGE_DIR=exp41/challenges/phase4/...`），所以该重叠是**项目共用协议**，不是 E6-A 独有；E6-A 与 E1/AutoIR 的相对比较仍公平。

### 2.1 排除 support 后的敏感性（不改训练，只从逐图 metrics 重算）

| 口径 | 含 support（原） | 排除本场景 5 张 support | 变化 |
|---|---:|---:|---:|
| R0-only best DIV2K macro | 26.817 | **26.819** | +0.002 |
| Final DIV2K macro | 27.654 | **27.659** | +0.005 |
| LSDIR | 不变 | 不变 | 0 |

结论：**support/test 重叠没有抬高 E6-A 的 test PSNR**（5/100 的选点效应 <0.01 dB）。这与 E1 的 `support_excluded`（24.9685→24.9686）一致。

### 2.2 对比：E6-B 的 support 不来自 test

- E6-B setA/setB 随机采样自 DIV2K_train_HR：240 个 support 文件 = **211 个唯一像素 hash，全部匹配 DIV2K_train；0 个匹配 DIV2K_valid/LSDIR**。
- 所以 E6-B 的 support-set 实验本身没有 test overlap；E6-A 与 E6-B 的 support 来源不同（exp41=valid vs setA/B=train），三组统计时必须披露这一 confound。

## 3. Agent 禁读路径审计

扫描本会话所有子 Agent 的 transcript（tool_use 输入，不含 prompt 文本）：

| 实验 | Agent 数 | 访问禁止路径的 Agent |
|---|---:|---:|
| E6-B full24（盲识别+策略） | 98 | **0** |
| E6-A full24（48 盲识别 + 46 R1 + 6 R2） | 100 | **1** |

唯一违规：`blind_0001_rep2`（E6-A）用 `grep` 读了 `rebuttal_plan/*.md`，输出中包含**其他场景**的 GT/失败提示（0006 JPEG:1、0011 poisson:3+gaussian:4、0013 speckle:3+JPEG:2；以及 0013/0017/0024 的失败模式）。该输出**不包含 blind_0001 自身的 GT**；它最终预测 `noise_poisson:5`（与 GT 一致）。

影响量化：把 0001 的 rep2 从统计中剔除、仅用 rep1（无违规）时，0001 场景均值从 28.83 降至 27.32，24 场景 macro 约 **−0.06 dB**（26.817→约 26.75；final 同量级）。没有任何其他 Agent 读取 exp41、`blind_agreement.md`、`degradation_gt/`、`.gt_mappings/`、`test_metrics` 或其他 challenge 的预测。

## 4. 为什么 26.8 / 27.65 比 rule/AutoIR 高

E6-A 使用 **Agent 盲识别 DC + 固定 Direct Swin+SG**（R0-only），不是规则控制器。逐场景对比（DIV2K）：

- E6-A vs AutoIR：mean **+1.85 dB**，**median −0.08 dB**；9/24 > +1 dB，8/24 < −1 dB。
- E6-A vs E1 rule：mean **+2.54 dB**，median **+0.65 dB**；11/24 > +1 dB，仅 2/24 < −1 dB。
- 增益集中在规则/AutoIR 的灾难性 DC 错误场景：0008 +19.37、0010 +9.39、0015 +7.81、0024 +7.47、0009 +5.73、0021 +5.34、0012 +3.87、0014 +3.34、0019 +2.10（这 9 个场景的 E1 规则最终 pipeline **全部不匹配 GT**，其中 7 个 E6-A rep2 匹配 GT）。
- 唯一明显回退：0004（E6-A −4.80 vs rule；E6-A rep1 gaussian 拖低场景均值，rep2 正确 motion）。
- GT 整链命中：**E6-A rep2 15/24、rep1 9/24；E1 规则最终 5/24**。DC 准确率差异直接解释 PSNR 差异。
- E6-B LLM 在 6 个场景上 30.194 dB；E6-A 在**同样 6 个场景**上为 28.31 dB，方向一致（E6-B 多了策略/R1/R2，且 6 场景本身均值高于全 24 场景）。E6-A 的 26.8 是**24 场景 macro**，不是把 6 场景的 30.2 直接对比。

## 5. 结论

1. **未发现系统性 GT 泄露**。E6-B 98 个 Agent 0 次违规；E6-A 仅 1 个 Agent 读到**其他场景**的 GT 提示，对自身场景无 GT，剔除后 macro 变化约 −0.06 dB。
2. **support/test 重叠确实存在**（E6-A/E1/AutoIR 共用 exp41 support，全部来自 DIV2K_valid），但排除本场景 5 张 support 后 test 变化 ≤0.005 dB；不能解释 2.5 dB 的 macro 差异。
3. **高 PSNR 来自 DC 准确率 + 重尾**：median 与 AutoIR 基本持平（−0.08 dB），少数场景规则/AutoIR 的灾难性错误拉低其 macro；E6-A rep2 GT 整链命中 15/24 vs 规则 5/24。
4. **三组统计的 confound**：exp41 support 来自 DIV2K_valid，setA/setB 来自 DIV2K_train；最终 24×3 表需同时报告 full test 与“排除 exp41 的 5 张 support”的 common-test sensitivity，不能把 support 来源差当成 support-set 方差。
5. 建议：后续 E6-B R1/R2 Agent 继续 worktree 隔离并逐批审计 transcript；如需严格干净表，可重跑 `blind_0001_rep2`（影响约 0.06 dB）。
