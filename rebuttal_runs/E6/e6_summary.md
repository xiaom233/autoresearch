# E6-B support-set stability (T9)

Each scenario uses 3 disjoint DIV2K_train support sets of 5 pairs. Test is the fixed GT-LQ DIV2K/LSDIR set.

| Scenario | Set1 PSNR | Set2 PSNR | Set3 PSNR | mean | SD | range | selected rounds |
|---|---:|---:|---:|---:|---:|---:|---|
| blind_0006 | 31.705 | 31.667 | 31.508 | 31.627 | 0.104 | 0.197 | R1/R1/R1 |
| blind_0008 | 42.268 | 41.356 | 43.576 | 42.400 | 1.116 | 2.221 | R1/R1/R1 |
| blind_0010 | 15.472 | 15.820 | 18.010 | 16.434 | 1.376 | 2.538 | R1/R1/R0 |
| blind_0013 | 26.029 | 26.221 | 26.005 | 26.085 | 0.118 | 0.216 | R0/R1/R0 |
| blind_0017 | 23.244 | 23.217 | 23.346 | 23.269 | 0.068 | 0.128 | R0/R0/R0 |
| blind_0024 | 11.283 | 13.545 | 13.940 | 12.923 | 1.434 | 2.658 | R0/R0/R1 |

Macro over six scenarios (mean of scenario means):

- set1_DIV2K: 25.000 ± 11.191 dB (min 11.283, max 42.268)
- set2_DIV2K: 25.304 ± 10.311 dB (min 13.545, max 41.356)
- set3_DIV2K: 26.064 ± 10.541 dB (min 13.940, max 43.576)
- set1_LSDIR: 23.197 ± 10.613 dB (min 10.960, max 40.318)
- set2_LSDIR: 23.540 ± 9.903 dB (min 12.865, max 39.792)
- set3_LSDIR: 24.171 ± 9.963 dB (min 13.545, max 41.300)

Support-set spread per dataset:

- DIV2K: mean range 1.326 dB, max range 2.658 dB
- LSDIR: mean range 1.136 dB, max range 2.585 dB

Selected pipeline exact GT match:

- blind_0006: set1=False, set2=False, set3=False
- blind_0008: set1=True, set2=True, set3=True
- blind_0010: set1=False, set2=False, set3=False
- blind_0013: set1=False, set2=False, set3=False
- blind_0017: set1=False, set2=False, set3=False
- blind_0024: set1=False, set2=False, set3=False