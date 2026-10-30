# 新实验统计口径对齐（2026-10-30）

> 按用户要求：不再纳入 exp41 旧实验；只对齐最近的新实验（E1 rule / E1 new LLM=E6-A / E6-B setA/setB）。

## 1. 对齐清单

| 项 | 设置 | E1 rule | E1 new LLM（E6-A final） | E6-A full24 | E6-B setA/setB |
|---|---|---|---|---|---|
| PSNR | RGB joint-MSE，`psnr_rgb` | ✅ | ✅ | ✅ | ✅ |
| test 图 | DIV2K_valid_HR 100 + LSDIR val1/HR 250 | ✅ 24×350 | ✅ 24×2×350 | ✅ 48×350 | ✅ 24×2×350 |
| image hash | 同一批图 | ✅ 0 mismatch vs E6A | ✅ | ✅ | ✅ 0 mismatch vs E6A |
| LQ | GT pipeline + seed-salt v1；7 场景 fixed-LQ | ✅ fixed-LQ | ✅ fixed-LQ（7 场景） | ✅ fixed-LQ（7 场景） | final test 将用 hybrid fixed-LQ |
| support 来源 | — | exp41（**DIV2K_valid**） | 同左 | 同左 | **DIV2K_train** |
| 重复单位 | scenario | 1 run | 2 reps（rep1/rep2 分列） | 2 reps | 2 sets（setA/setB 分列） |
| checkpoint 选择 | support PSNR | R0–R2 | R0/R1 | R0/R1 | R0–R2 |
| seed | 42 | ✅ | ✅ | ✅ | ✅ |
| 聚合 | per-image mean → scenario mean → 24 场景等权 macro | ✅ | ✅ | ✅ | ✅ |
| bootstrap | 场景级 10k | ✅ | ✅ | —（rep SD） | —（set SD） |

## 2. 可合并 / 不可合并

**可合并（同 support 来源、同 test/LQ）**

- **E1 rule vs E1 new LLM（E6-A final）**：24 场景、同 exp41 support、同 test、同 RGB/fixed-LQ；配对 bootstrap 有效。
  - new LLM rep1/rep2/mean：DIV2K 28.030 / 27.277 / 27.654；LSDIR 25.554 / 24.928 / 25.241。
  - vs rule：DIV2K +3.758 [+1.624,+6.257] / +3.006 [+1.274,+5.134] / +3.382 [+1.520,+5.619]（18/24 胜）；LSDIR +3.422 / +2.796 / +3.109（19/24）。
- **E6-B setA vs setB**：同 DIV2K_train support pool、同 test/LQ、同流程；support-set 方差有效。

**不可直接合并**

- **E6-A / E1 vs E6-B**：support 来源不同（DIV2K_valid vs DIV2K_train）。不能把三者直接放进一个 24×3 mean/SD 当作纯 support-set 方差；必须标注来源，或只在 common-test 上做敏感性。
- 用户要求的“新三组”若指 **E6-A + setA + setB**：可以计算 24×3 行，但第 1 组（E6-A）的 support 是 valid，第 2/3 组是 train；结论只能表述为“新实验的三组汇总”，不能表述为“同一 support 分布的三组”。
- exp41 旧链按用户要求不纳入。

## 3. E1 更新产物

- `E1/e1_newllm_summary.json`：macro、paired bootstrap、逐场景表。
- `E1/e1_newllm_summary.md`：表格。
- `E1/e1_newllm_scenarios.csv`：24 行（rule / old AutoIR / new rep1 / new rep2 / new mean）。
- `E1/PSNR_PROTOCOL.md`：RGB 口径证据。
- `scripts/update_e1_newllm.py`：可复现脚本。
- 原始 `E1/e1_final_summary.json` 未覆盖；旧 A0/exp41 AutoIR 保留为历史基线。

## 4. 结论

- PSNR、test 图、LQ、聚合方式在 **E1 new LLM / E6-A / E6-B** 之间已对齐（RGB joint-MSE；同一批 DIV2K+LSDIR；fixed-LQ）。
- 新 LLM（E6-A final）显著优于 rule：DIV2K +3.382 dB（18/24），LSDIR +3.109 dB（19/24），CI 均 >0。
- E6-B setA/setB 内部对齐；与 E6-A/E1 的 support 来源不同，不能无标注地混合。
- 旧 exp41 实验不纳入新统计。
