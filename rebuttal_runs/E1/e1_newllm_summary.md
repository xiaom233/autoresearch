# E1 新 LLM 对照（E6-A full24 final checkpoints）

E1 rule 为规则控制器；旧 AutoIR = A0/exp41 六月链（历史）；new LLM = E6-A full24 final（新 Skill agents，R0/R1 selected，fixed Direct Swin+SG）。

| 口径 | DIV2K macro | LSDIR macro |
|---|---:|---:|
| E1 rule | 24.272 | 22.132 |
| 旧 AutoIR（A0/exp41） | 24.968 | 22.956 |
| new LLM rep1 | 28.030 | 25.554 |
| new LLM rep2 | 27.277 | 24.928 |
| new LLM rep1/rep2 mean | 27.654 | 25.241 |

| 配对差 | DIV2K mean [95% CI] | wins | LSDIR mean [95% CI] | wins |
|---|---:|---:|---:|---:|
| new LLM rep1 − rule | +3.758 [+1.624,+6.257] | 18/24 | +3.422 [+1.419,+5.780] | 19/24 |
| new LLM rep2 − rule | +3.006 [+1.274,+5.134] | 18/24 | +2.796 [+1.160,+4.881] | 19/24 |
| new LLM mean − rule | +3.382 [+1.520,+5.619] | 18/24 | +3.109 [+1.348,+5.299] | 19/24 |

旧 AutoIR 仍是历史基线，不与 new LLM 混为同一版本。