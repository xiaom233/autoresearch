#!/usr/bin/env python3
"""Select E6-A full24 checkpoints across R0/R1/R2 by best support PSNR."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuttal_runs/E6A_full24"
SCENARIOS = [f"blind_{i:04d}" for i in range(1, 25)]
REPS = [1, 2]
ROUNDS = ["R0", "R1", "R2"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse(name, rnd):
    if rnd == "R0":
        log = OUT / "logs" / f"{name}.log"
        ckdir = OUT / "experiments" / name / "checkpoints"
        ckname = f"{name}_step"
    else:
        log = OUT / "logs" / f"{name}_{rnd}.log"
        ckdir = OUT / "experiments" / f"{name}_{rnd}" / "checkpoints"
        ckname = f"{name}_{rnd}_step"
    if not log.exists():
        return None
    txt = log.read_text(errors="ignore")
    if "TRAIN END" not in txt:
        return None
    mm = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    exp_meta = json.loads(mm.group(1)) if mm else None
    if exp_meta is None:
        return None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    if not m:
        return None
    step, psnr = int(m.group(1)), m.group(2)
    if psnr == "nan" or "loss: nan" in txt:
        return None
    ck = ckdir / f"{ckname}{step}.pt"
    if not ck.exists():
        return None
    pfile = OUT / "challenges" / name / (
        "predicted_params.json" if rnd == "R0" else f"predicted_params_{rnd}.json")
    if not pfile.exists():
        return None
    pipe = [[s["function"], int(s["severity"])]
            for s in json.load(open(pfile))["pipeline"]]
    return {"round": rnd, "checkpoint": str(ck.relative_to(ROOT)),
            "checkpoint_sha256": sha_file(ck), "checkpoint_step": step,
            "support_psnr": float(psnr), "predicted_pipeline": pipe,
            "exp_meta": exp_meta}


def main():
    rows, all_rounds = [], {}
    for sid in SCENARIOS:
        for rep in REPS:
            name = f"{sid}_rep{rep}"
            cands = [x for x in (parse(name, r) for r in ROUNDS) if x]
            if not cands:
                print(f"[warn] no usable checkpoint {name}")
                continue
            # max support PSNR; ties -> earlier round
            best = max(cands, key=lambda x: (x["support_psnr"], -ROUNDS.index(x["round"])))
            rows.append({
                "run_id": f"E6A24_{name}", "scenario_id": sid, "rep": rep,
                "name": name, "controller": "E6A-full24-agent",
                "selected_round": best["round"],
                "checkpoint": best["checkpoint"],
                "checkpoint_sha256": best["checkpoint_sha256"],
                "checkpoint_step": best["checkpoint_step"],
                "support_psnr": best["support_psnr"],
                "predicted_pipeline": best["predicted_pipeline"],
                "exp_meta": best["exp_meta"],
                "candidates": [{k: v for k, v in c.items() if k != "exp_meta"}
                               for c in cands],
            })
            all_rounds[name] = cands
    with open(OUT / "selection_final.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(all_rounds, open(OUT / "round_selection.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"wrote {len(rows)} rows -> {OUT/'selection_final.jsonl'}")
    from collections import Counter
    print("selected rounds:", Counter(r["selected_round"] for r in rows))
    for r in rows:
        print(f"{r['name']:18s} {r['selected_round']} step{r['checkpoint_step']} "
              f"support={r['support_psnr']:.2f}")


if __name__ == "__main__":
    main()
