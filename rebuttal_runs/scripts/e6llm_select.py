#!/usr/bin/env python3
"""Select E6-LLM checkpoints across R0/R1/R2 by support PSNR."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "rebuttal_runs/E6_LLM"
ROUNDS = ["R0", "R1", "R2"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_log(name, rnd):
    p = E / "logs" / f"{name}_{rnd}.log"
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    meta = json.loads(m.group(1)) if m else None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+|nan)\)", txt)
    if not m:
        return None
    step, psnr = int(m.group(1)), m.group(2)
    nan = "loss: nan" in txt or "PSNR_RGB=nan" in txt
    ck = E / "runs" / name / rnd / "checkpoints" / f"{rnd}_step{step}.pt"
    if not ck.exists() or nan or psnr == "nan":
        return None
    params = E / "challenges" / name / (f"predicted_params_{rnd}.json" if rnd != "R0" else "predicted_params.json")
    if not params.exists():
        params = E / "challenges" / name / "predicted_params.json"
    pipe = [[s["function"], int(s["severity"])] for s in json.load(open(params))["pipeline"]]
    return {"round": rnd, "checkpoint": str(ck.relative_to(ROOT)),
            "checkpoint_sha256": sha_file(ck), "checkpoint_step": step,
            "support_psnr": float(psnr), "predicted_pipeline": pipe,
            "exp_meta": meta}


def main():
    rows = []
    for d in sorted((E / "challenges").iterdir()):
        if not (d / "predicted_params.json").exists():
            continue
        name = d.name
        cands = [x for x in (parse_log(name, r) for r in ROUNDS) if x]
        if not cands:
            print(f"[warn] no usable checkpoint {name}")
            continue
        best = max(cands, key=lambda x: (x["support_psnr"], ROUNDS.index(x["round"])))
        sid, setk = name.rsplit("_set", 1)
        rows.append({"run_id": f"E6LLM_{name}", "scenario_id": sid,
                     "support_set": f"set{setk}", "name": name,
                     **best, "candidates": [{k: v for k, v in c.items() if k != "exp_meta"}
                                            for c in cands]})
    out = E / "selection.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    for r in rows:
        print(f"{r['name']:24s} {r['round']} step{r['checkpoint_step']} support={r['support_psnr']:.2f}")


if __name__ == "__main__":
    main()
