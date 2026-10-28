# E5a DC metric recomputation

## exp40 (public blind_id_detail.csv)

| Method/Round | Challenges | Pos Type micro | Pos Spec micro | Exact Type | Exact Spec | Step Acc | Multi Type F1 | Multi Spec F1 | Over/Under | Sev MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 24 | 18.75% | 0.00% | 12.50% | 0.00% | 100.00% | 23.61% | 0.00% | 0/0 | 2.15 |
| Qwen | 24 | 16.67% | 10.42% | 4.17% | 0.00% | 54.17% | 28.33% | 13.19% | 8/5 | 0.77 |
| R0 | 24 | 25.00% | 18.75% | 25.00% | 20.83% | 54.17% | 62.22% | 47.50% | 4/7 | 0.38 |
| R1 | 24 | 27.08% | 25.00% | 20.83% | 16.67% | 50.00% | 56.94% | 45.97% | 5/9 | 0.26 |

Ours (reported best-of-R0/R1): positional Type 31.25%, Spec 27.08%; selection basis UNKNOWN.

## exp41 (actual training params, evaluation-side GT)

| Method/Round | Challenges | Pos Type micro | Pos Spec micro | Exact Type | Exact Spec | Step Acc | Multi Type F1 | Multi Spec F1 | Over/Under | Sev MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 24 | 25.00% | 18.75% | 25.00% | 20.83% | 54.17% | 62.22% | 47.50% | 4/7 | 0.38 |
| R1 | 24 | 31.25% | 27.08% | 25.00% | 20.83% | 54.17% | 62.78% | 50.14% | 5/6 | 0.33 |
| R2 | 24 | 43.75% | 35.42% | 37.50% | 29.17% | 62.50% | 70.56% | 56.81% | 4/5 | 0.36 |
