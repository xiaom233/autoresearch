#!/usr/bin/env python3
"""Select E6-B checkpoints by support PSNR across R0/R1/R2."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E6 = ROOT / "rebuttal_runs/E6"
SCENARIOS = ["blind_0006", "blind_0008", "blind_0010",
             "blind_0013", "blind_0017", "blind_0024"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_log(p):
    txt = p.read_text(errors="ignore")
    m = re.search(r"=== EXP_META: (\{.*?\}) ===", txt, re.S)
    meta = json.loads(m.group(1)) if m else None
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", txt)
    if not m:
        return None, None, meta
    return int(m.group(1)), float(m.group(2)), meta


def main():
    rows = []
    for k in [1, 2, 3]:
        base = E6 / "run" / f"set{k}"
        for sid in SCENARIOS:
            cands = []
            for rnd in ["R0", "R1", "R2"]:
                log = base / "logs" / f"{sid}_{rnd}.log"
                if not log.exists():
                    continue
                step, psnr, meta = parse_log(log)
                if step is None or meta is None:
                    continue
                ck = base / "experiments" / f"{sid}_{rnd}" / "checkpoints" / f"{sid}_{rnd}_step{step}.pt"
                if not ck.exists():
                    print(f"[warn] missing {ck}")
                    continue
                cands.append({"round": rnd, "step": step, "support_psnr": psnr,
                              "checkpoint": str(ck.relative_to(ROOT)),
                              "sha256": sha_file(ck), "exp_meta": meta})
            if not cands:
                print(f"[warn] no candidates {sid} set{k}")
                continue
            best = max(cands, key=lambda x: (x["support_psnr"], x["round"]))
            rows.append({
                "run_id": f"E6_set{k}_{sid}",
                "scenario_id": sid,
                "support_set": f"set{k}",
                "controller": "rule",
                "round": best["round"],
                "checkpoint": best["checkpoint"],
                "checkpoint_sha256": best["sha256"],
                "checkpoint_step": best["step"],
                "support_psnr": best["support_psnr"],
                "exp_meta": best["exp_meta"],
                "candidates": [{kk: vv for kk, vv in c.items() if kk != "exp_meta"} for c in cands],
            })
    out = E6 / "selection.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
