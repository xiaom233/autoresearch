#!/usr/bin/env python3
"""Measure AirNet output sensitivity to two LQ realizations of the same GT pipeline."""
import argparse
import glob
import hashlib
import io
import json
import random
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "benchmark/AirNet")
from net.model import AirNet  # noqa: E402
from x_distortion import add_distortion  # noqa: E402


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


def stable_seed(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def make_lq(clean, bid, image_id, pipeline, salt):
    seed = stable_seed(bid, image_id, "test-lq-v1", salt)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    img = clean.copy()
    for step in pipeline:
        img = add_distortion(img, severity=int(step["severity"]),
                             distortion_name=step["function"])
    return img


def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 100.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def tar_deg_map(sid):
    out = {}
    for f in sorted(glob.glob("benchmark_results/airnet/DIV2K_gpu*.tar")):
        with tarfile.open(f) as t:
            for m in t.getmembers():
                if m.name.endswith(".degraded.jpg") and f"/{sid}/" in "/" + m.name:
                    img = m.name.split("/")[-1].split(".")[0]
                    out[img] = np.array(Image.open(io.BytesIO(
                        t.extractfile(m).read())).convert("RGB"))
    return out


def infer(model, x):
    with torch.no_grad():
        y = model(x, x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y.float().clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rebuttal_runs/AUDIT_airnet_lq_sensitivity.json")
    args = ap.parse_args()
    opt = argparse.Namespace()
    opt.batch_size = 5
    model = AirNet(opt).cuda().eval()
    model.load_state_dict(torch.load("benchmark/AirNet/ckpt/All.pth",
                                     map_location="cuda", weights_only=True))
    out = {}
    for sid in ["blind_0001", "blind_0005"]:
        pipe = json.load(open(f"exp41/degradation_gt/{sid}_params.json"))["pipeline"]
        pa, pb, deltas = [], [], []
        degmap = tar_deg_map(sid)
        for p in sorted(glob.glob("datasets/DIV2K/DIV2K_valid_HR/*.png"))[:50]:
            name = p.split("/")[-1].split(".")[0]
            clean = load_rgb(p)
            h, w = clean.shape[:2]
            y, x = (h - 512) // 2, (w - 512) // 2
            c = torch.from_numpy(clean[y:y + 512, x:x + 512]).permute(
                2, 0, 1).float().div(255).unsqueeze(0).cuda()
            A = degmap.get(name)
            if A is None:
                continue
            B = make_lq(clean, sid, f"DIV2K/{name}", pipe, "v1")
            A = torch.from_numpy(A[y:y + 512, x:x + 512]).permute(
                2, 0, 1).float().div(255).unsqueeze(0).cuda()
            B = torch.from_numpy(B[y:y + 512, x:x + 512]).permute(
                2, 0, 1).float().div(255).unsqueeze(0).cuda()
            va, vb = psnr(infer(model, A), c), psnr(infer(model, B), c)
            pa.append(va)
            pb.append(vb)
            deltas.append(vb - va)
        out[sid] = {"n": len(pa), "psnr_A": float(np.mean(pa)),
                    "psnr_B": float(np.mean(pb)),
                    "mean_delta": float(np.mean(deltas)),
                    "max_abs_delta": float(np.max(np.abs(deltas)))}
        print(sid, out[sid], flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
