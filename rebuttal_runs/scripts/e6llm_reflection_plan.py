#!/usr/bin/env python3
"""Decide which E6-LLM R0 chains need an R1 Skill-reflection agent.

Trigger (support-only, no test):
- R0 training NaN -> reflect (architecture fallback first)
- verdict UNCERTAIN / POOR -> reflect
- LIKELY/GOOD and support PSNR >= 35 -> skip
- LIKELY/GOOD and support PSNR < 35 -> reflect
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "rebuttal_runs/E6_LLM"
rows = json.load(open(E / "r0_collect.json"))
out = []
for r in rows:
    name = r["name"]
    params = json.load(open(E / "challenges" / name / "predicted_params.json"))
    verdict = str(params.get("analysis", {}).get("verdict", "UNKNOWN")).upper()
    psnr = r.get("support_psnr")
    if r.get("nan"):
        need, why = True, "R0 training NaN"
    elif verdict in ("UNCERTAIN", "POOR"):
        need, why = True, f"verdict={verdict}"
    elif psnr is not None and psnr < 35:
        need, why = True, f"LIKELY/GOOD but support PSNR {psnr:.2f}<35"
    else:
        need, why = False, f"LIKELY/GOOD with support PSNR {psnr}"
    out.append({"name": name, "verdict": verdict, "support_psnr": psnr,
                "best_step": r.get("best_step"), "nan": r.get("nan"),
                "need_r1": need, "reason": why})
json.dump(out, open(E / "reflection_plan.json", "w"), indent=2, ensure_ascii=False)
print(f"{'name':24s} verdict     support  need_r1 reason")
for x in out:
    print(f"{x['name']:24s} {x['verdict']:11s} {str(x['support_psnr']):>8s} "
          f"{str(x['need_r1']):7s} {x['reason']}")
print("need R1:", sum(x["need_r1"] for x in out), "/", len(out))
