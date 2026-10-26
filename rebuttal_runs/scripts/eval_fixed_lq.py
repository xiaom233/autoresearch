#!/usr/bin/env python3
"""Re-evaluate the 7 nondeterministic scenarios on one materialized LQ."""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(os_root))
sys.path.insert(0, str(os_root / "rebuttal_runs/scripts"))
from eval_paired import load_restorenet, load_dfpir, psnr, ssim, lpips_fn  # noqa: E402
from train_dfpir_budget import tiled_infer  # noqa: E402

ROOT = os_root
FIX = ROOT / "rebuttal_runs/E1E2_fixed_lq"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pad_to_window(x, ws):
    _, _, h, w = x.shape
    ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


def collect_configs(sid):
    cfg = {}
    a0 = {r["scenario_id"]: r for r in map(json.loads, open(
        ROOT / "rebuttal_runs/A0/support_selection.jsonl"))}
    e1 = {r["scenario_id"]: r for r in map(json.loads, open(
        ROOT / "rebuttal_runs/E1/selection.jsonl"))}
    r = a0[sid]
    cfg["AutoIR"] = {"kind": "restorenet", "checkpoint": r["selected_checkpoint"],
                     "meta": r["exp_meta"], "round": r["selected_round"]}
    r = e1[sid]
    cfg["rule"] = {"kind": "restorenet", "checkpoint": r["checkpoint"],
                   "meta": r["exp_meta"], "round": r["selected_round"]}

    labeled = [json.loads(l) for l in open(
        ROOT / "rebuttal_runs/E2/test_metrics_labeled.jsonl")]
    snaps = []
    for p in glob.glob(str(ROOT / "rebuttal_runs/E2/test_metrics_snapshots/*.jsonl")):
        snaps += [json.loads(l) for l in open(p)]
    for ctrl in ["DFPIR-original", "DFPIR-ft-best", "DFPIR-ft-last"]:
        rows = {x["scenario_id"]: x for x in labeled if x["controller"] == ctrl}
        x = rows[sid]
        cfg[ctrl] = {"kind": "dfpir", "checkpoint": x["checkpoint"],
                     "meta": None, "round": x.get("round", "")}
    for ctrl, rnd in [("DFPIR-ft-T1", "T1of3"), ("DFPIR-ft-T2", "T2of3"),
                      ("DFPIR-ft-T3", "T3of3")]:
        rows = {x["scenario_id"]: x for x in snaps
                if x["controller"] == ctrl and x["scenario_id"] == sid}
        x = rows[sid]
        cfg[ctrl] = {"kind": "dfpir", "checkpoint": x["checkpoint"],
                     "meta": None, "round": rnd}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--gpu-id", type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda")

    sid = args.scenario
    index = json.load(open(FIX / sid / "index.json"))
    pairs = index["pairs"]
    clean = {(p["dataset"], p["index"]): np.array(
        Image.open(ROOT / p["clean_path"]).convert("RGB"), dtype=np.uint8) for p in pairs}
    lq = {(p["dataset"], p["index"]): np.array(
        Image.open(ROOT / p["lq_path"]).convert("RGB"), dtype=np.uint8) for p in pairs}
    cfg = collect_configs(sid)
    out_path = FIX / sid / "metrics.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                done.add((r["controller"], r["dataset"], r["image_id"]))
    out = open(out_path, "a")

    # cache by checkpoint path so best==last/T3 are evaluated once
    cache = {}
    for i, (ctrl, c) in enumerate(cfg.items()):
        ck = ROOT / c["checkpoint"]
        if not ck.exists():
            print(f"[skip] {sid} {ctrl}: missing {ck}", flush=True)
            continue
        key = str(ck.resolve())
        if key not in cache:
            print(f"[load] {sid} {ctrl} {c['kind']} {ck.name}", flush=True)
            if c["kind"] == "restorenet":
                model = load_restorenet(str(ck), c["meta"])
            else:
                model = load_dfpir(str(ck))
            model.eval()
            ck_sha = sha_file(ck)
            rows = []
            for p in pairs:
                i2 = p["index"]
                ds, image_id = p["dataset"], p["image_id"]
                if (ctrl, ds, image_id) in done:
                    continue
                key2 = (ds, i2)
                lq_t = torch.from_numpy(lq[key2]).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(device)
                clean_t = torch.from_numpy(clean[key2]).permute(2, 0, 1).float().div(255.).to(device)
                with torch.no_grad():
                    if c["kind"] == "restorenet":
                        x, h, w = pad_to_window(lq_t, model.window_size)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred = model(x)
                        pred = pred[:, :, :h, :w].float().clamp(0, 1)[0]
                    else:
                        pred = tiled_infer(model, lq_t, tile=512, overlap=64)[0]
                rows.append({
                    "controller": ctrl, "round": c["round"],
                    "scenario_id": sid, "dataset": ds, "image_id": image_id,
                    "checkpoint": c["checkpoint"], "checkpoint_sha256": ck_sha,
                    "lq_sha256": p["lq_sha256"], "image_sha256": p["image_sha256"],
                    "metric_version": "psnr_rgb_v1|ssim_gauss11x11_s1.5|lpips_vgg|fixed_lq_v1",
                    "psnr_rgb": psnr(pred, clean_t), "ssim_rgb": ssim(pred, clean_t),
                    "lpips_vgg": float(lpips_fn()(pred.unsqueeze(0) * 2 - 1,
                                                  clean_t.unsqueeze(0) * 2 - 1).item()),
                })
            cache[key] = rows
            del model
            torch.cuda.empty_cache()
        for r in cache[key]:
            r2 = dict(r)
            r2["controller"] = ctrl
            r2["round"] = c["round"]
            out.write(json.dumps(r2, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[done] {sid} {ctrl}: {len(cache[key])} rows", flush=True)
    out.close()


if __name__ == "__main__":
    main()
