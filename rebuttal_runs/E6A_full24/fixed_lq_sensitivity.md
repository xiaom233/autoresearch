# E6-A full24 fixed-LQ sensitivity (7 nondeterministic scenarios)

Original pass generated LQ per process (rep1/rep2 and passes differ); fixed pass uses one shared materialized realization.

- best: mean Δ(fixed−orig) DIV2K -0.001 dB (max |Δ| 0.010); LSDIR +0.002 dB (max |Δ| 0.018)
- last: mean Δ(fixed−orig) DIV2K -0.000 dB (max |Δ| 0.004); LSDIR +0.001 dB (max |Δ| 0.004)
- final: mean Δ(fixed−orig) DIV2K -0.001 dB (max |Δ| 0.006); LSDIR +0.001 dB (max |Δ| 0.006)

## Corrected macro (fixed LQ for 7 scenarios; original for other 17)

| kind | DIV2K macro | within-SD | max range | LSDIR macro | within-SD | max range |
|---|---:|---:|---:|---:|---:|---:|
| best | 26.817 | 1.491 | 10.398 | 24.549 | 1.256 | 9.062 |
| last | 25.526 | 1.535 | 10.398 | 23.382 | 1.321 | 9.062 |
| final | 27.654 | 0.855 | 13.043 | 25.241 | 0.724 | 10.854 |

| Scenario | rep | kind | orig DIV2K | fixed DIV2K | Δ | orig range | fixed range |
|---|---:|---|---:|---:|---:|---:|---:|
| blind_0003 | 1 | best | 28.129 | 28.128 | -0.001 | 0.043 | 0.045 |
| blind_0003 | 2 | best | 28.171 | 28.173 | +0.002 | 0.043 | 0.045 |
| blind_0005 | 1 | best | 44.536 | 44.526 | -0.010 | 0.454 | 0.451 |
| blind_0005 | 2 | best | 44.081 | 44.075 | -0.007 | 0.454 | 0.451 |
| blind_0017 | 1 | best | 23.045 | 23.047 | +0.001 | 0.416 | 0.416 |
| blind_0017 | 2 | best | 23.461 | 23.463 | +0.002 | 0.416 | 0.416 |
| blind_0018 | 1 | best | 17.500 | 17.501 | +0.001 | 0.072 | 0.071 |
| blind_0018 | 2 | best | 17.571 | 17.572 | +0.000 | 0.072 | 0.071 |
| blind_0020 | 1 | best | 16.694 | 16.697 | +0.003 | 0.702 | 0.703 |
| blind_0020 | 2 | best | 15.993 | 15.994 | +0.001 | 0.702 | 0.703 |
| blind_0022 | 1 | best | 25.506 | 25.509 | +0.002 | 2.164 | 2.167 |
| blind_0022 | 2 | best | 23.342 | 23.341 | -0.001 | 2.164 | 2.167 |
| blind_0023 | 1 | best | 21.834 | 21.833 | -0.001 | 0.274 | 0.275 |
| blind_0023 | 2 | best | 21.560 | 21.558 | -0.002 | 0.274 | 0.275 |
| blind_0003 | 1 | final | 28.127 | 28.128 | +0.001 | 0.046 | 0.045 |
| blind_0003 | 2 | final | 28.173 | 28.173 | -0.000 | 0.046 | 0.045 |
| blind_0005 | 1 | final | 44.529 | 44.526 | -0.003 | 0.448 | 0.451 |
| blind_0005 | 2 | final | 44.081 | 44.075 | -0.006 | 0.448 | 0.451 |
| blind_0017 | 1 | final | 23.045 | 23.047 | +0.001 | 0.418 | 0.416 |
| blind_0017 | 2 | final | 23.464 | 23.463 | -0.001 | 0.418 | 0.416 |
| blind_0018 | 1 | final | 17.498 | 17.501 | +0.002 | 0.074 | 0.071 |
| blind_0018 | 2 | final | 17.572 | 17.572 | -0.000 | 0.074 | 0.071 |
| blind_0020 | 1 | final | 18.287 | 18.281 | -0.006 | 2.463 | 2.455 |
| blind_0020 | 2 | final | 15.824 | 15.826 | +0.002 | 2.463 | 2.455 |
| blind_0022 | 1 | final | 25.507 | 25.509 | +0.001 | 0.029 | 0.029 |
| blind_0022 | 2 | final | 25.479 | 25.479 | +0.001 | 0.029 | 0.029 |
| blind_0023 | 1 | final | 21.835 | 21.833 | -0.002 | 0.277 | 0.275 |
| blind_0023 | 2 | final | 21.557 | 21.558 | +0.001 | 0.277 | 0.275 |