#!/usr/bin/env python3
"""Label raw E2 per-image metrics by checkpoint role (best/last/T1/T2/T3)."""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E2 = ROOT / "rebuttal_runs" / "E2"


def main():
    summaries = {}
    for p in glob.glob(str(E2 / "*" / "*_summary.json")):
        d = json.load(open(p))
        summaries[d["scenario"]] = d
    out = open(E2 / "test_metrics_labeled.jsonl", "w")
    n = 0
    for p in glob.glob(str(E2 / "test_metrics" / "dfpir_gpu*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            sid = r["scenario_id"]
            ck = r["checkpoint"]
            s = summaries.get(sid, {})
            labels = []
            if r.get("controller") == "DFPIR-original" or "dfpir_blind_step" in ck:
                labels = ["DFPIR-original"]
            else:
                if ck == s.get("best_checkpoint"):
                    labels.append("DFPIR-ft-best")
                if ck == s.get("last_checkpoint"):
                    labels.append("DFPIR-ft-last")
                stem = Path(ck).stem
                for tag in ("T1of3", "T2of3", "T3of3"):
                    if tag in stem:
                        labels.append(f"DFPIR-ft-{tag.replace('of3','')}")
            for ctrl in labels:
                rr = dict(r)
                rr["controller"] = ctrl
                out.write(json.dumps(rr, ensure_ascii=False) + "\n")
                n += 1
    out.close()
    print(f"wrote {n} labeled rows")


if __name__ == "__main__":
    main()
