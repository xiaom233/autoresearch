# E6-LLM support-set variation (full LLM flow)

6 scenarios x 3 support sets; Skill blind ID + strategy agent + Direct R0 + Skill R1/R2 reflection + support selection; fixed GT-LQ test.

| Scenario | set1 | set2 | set3 | mean | SD | range | rule set1 | rule set2 | rule set3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_0006 | 31.590 | 31.620 | 31.605 | 31.605 | 0.015 | 0.030 | 31.705 | 31.667 | 31.508 |
| blind_0008 | 39.986 | 42.942 | 42.096 | 41.675 | 1.522 | 2.956 | 42.268 | 41.356 | 43.576 |
| blind_0010 | 31.593 | 31.625 | 31.151 | 31.456 | 0.265 | 0.474 | 15.472 | 15.820 | 18.010 |
| blind_0013 | 29.206 | 29.197 | 29.086 | 29.163 | 0.067 | 0.120 | 26.029 | 26.221 | 26.005 |
| blind_0017 | 23.496 | 24.194 | 23.500 | 23.730 | 0.402 | 0.699 | 23.244 | 23.217 | 23.346 |
| blind_0024 | 24.373 | 21.443 | 24.780 | 23.532 | 1.820 | 3.336 | 11.283 | 13.545 | 13.940 |

DIV2K macro: 30.194 dB; mean within-scenario SD 0.682; mean range 1.2691175563103407.
LSDIR macro: 27.662 dB; mean within-scenario SD 0.638; mean range 1.1526352760881273.

vs rule E6-B:
- DIV2K (primary, 6-scenario paired bootstrap): +4.737 dB, CI [+0.341, +9.789], LLM wins 4/6 scenarios.
  - cell-level (18 cells, not independent): +4.737 dB, CI [+2.062, +7.705], wins 14/18.
- LSDIR (primary, 6-scenario paired bootstrap): +4.026 dB, CI [+0.183, +8.725], LLM wins 4/6 scenarios.
  - cell-level (18 cells, not independent): +4.026 dB, CI [+1.633, +6.719], wins 14/18.