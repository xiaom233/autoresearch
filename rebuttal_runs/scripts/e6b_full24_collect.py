#!/usr/bin/env python3
"""Collect E6-B full24 round results and build reflection plans (support-only)."""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6B_full24"
NAMES = [f"blind_{i:04d}_{s}" for i in range(1, 25) for s in ("setA", "setB")]
ROUND_FILE = {"R0": "predicted_params.json", "R1": "predicted_params_R1.json",
              "R2": "predicted_params_R2.json"}


def parse_round(name, rnd):
    if rnd == "R0":
        log = OUT / "logs" / f"{name}.log"
        ckpat = OUT / "experiments" / name / "checkpoints" / f"{name}_step*.pt"
    else:
        log = OUT / "logs" / f"{name}_{rnd}.log"
        ckpat = OUT / "experiments" / f"{name}_{rnd}" / "checkpoints" / f"{name}_{rnd}_step*.pt"
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    done = "TRAIN END" in txt
    nan = "loss: nan" in txt or "PSNR_RGB=nan" in txt
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    step = int(m.group(1)) if m else None
    psnr = (float(m.group(2)) if m.group(2) != "nan" else None) if m else None
    ck = None
    if step is not None:
        p = Path(str(ckpat).replace("*", str(step)))
        if p.exists():
            ck = str(p.relative_to(ROOT))
    return {"round": rnd, "done": done, "nan": nan, "best_step": step,
            "support_psnr": psnr, "checkpoint": ck, "log": str(log.relative_to(ROOT))}


def pipeline(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return [(s["function"], int(s["severity"])) for s in d.get("pipeline", [])]


def main():
    plan, r2plan = [], []
    for name in NAMES:
        cdir = OUT / "challenges" / name
        r0 = parse_round(name, "R0")
        if r0 and r0["done"]:
            (cdir / "r0_result.json").write_text(
                json.dumps(r0, indent=2, ensure_ascii=False), encoding="utf-8")
        r1 = parse_round(name, "R1")
        if r1 and r1["done"]:
            (cdir / "r1_result.json").write_text(
                json.dumps(r1, indent=2, ensure_ascii=False), encoding="utf-8")
        r2 = parse_round(name, "R2")
        if r2 and r2["done"]:
            (cdir / "r2_result.json").write_text(
                json.dumps(r2, indent=2, ensure_ascii=False), encoding="utf-8")

        p0 = pipeline(cdir / ROUND_FILE["R0"])
        pred = json.loads((cdir / ROUND_FILE["R0"]).read_text()) if p0 else {}
        verdict = str(pred.get("analysis", {}).get("verdict", "UNKNOWN")).upper()
        psnr = r0["support_psnr"] if r0 else None
        if not r0 or not r0["done"]:
            plan.append({"name": name, "status": "pending", "need_r1": None,
                         "verdict": verdict, "support_psnr": psnr})
            continue
        if r0["nan"]:
            need, why = True, "R0 training NaN"
        elif verdict in ("UNCERTAIN", "POOR"):
            need, why = True, f"verdict={verdict}"
        elif psnr is not None and psnr < 35:
            need, why = True, f"LIKELY but support PSNR {psnr:.2f}<35"
        else:
            need, why = False, f"LIKELY with support PSNR {psnr}"
        plan.append({"name": name, "status": "done", "need_r1": need,
                     "verdict": verdict, "support_psnr": psnr,
                     "best_step": r0["best_step"], "nan": r0["nan"], "reason": why})

        p1 = pipeline(cdir / ROUND_FILE["R1"])
        if r1 and r1["done"] and p1 is not None:
            change = p1 != p0
            delta = (r1["support_psnr"] - psnr) if (r1["support_psnr"] is not None
                                                     and psnr is not None) else None
            need_r2 = bool(change and delta is not None and delta > 2.0)
            if need_r2:
                why2 = f"CHANGE and R1 support +{delta:.2f} dB > 2"
            elif not change:
                why2 = "R1 NO_CHANGE"
            else:
                why2 = f"CHANGE but R1 support delta {delta:+.2f} dB <= 2"
            r2plan.append({"name": name, "r1_change": change,
                           "r0_support_psnr": psnr,
                           "r1_support_psnr": r1["support_psnr"],
                           "delta": delta, "need_r2": need_r2, "reason": why2})

    json.dump(plan, open(OUT / "reflection_plan.json", "w"), indent=2, ensure_ascii=False)
    json.dump(r2plan, open(OUT / "r2_plan.json", "w"), indent=2, ensure_ascii=False)
    done = [x for x in plan if x.get("status") == "done"]
    print(f"R0 done {len(done)}/48; R1 needed {sum(bool(x.get('need_r1')) for x in done)}")
    if r2plan:
        print(f"R2 plan: {sum(x['need_r2'] for x in r2plan)} needed of {len(r2plan)} R1-trained")
        for x in r2plan:
            print(f"  {x['name']:20s} change={x['r1_change']} delta={x['delta']} need_r2={x['need_r2']}")


if __name__ == "__main__":
    main()
