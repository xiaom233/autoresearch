#!/usr/bin/env python3
"""Evaluate DFPIR checkpoints on the same fixed-LQ test protocol as AutoIR."""
import argparse
import glob
import json
import os
import sys
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rebuttal_runs" / "scripts"))

from eval_checkpoint import (  # noqa: E402
    infer, load_rgb, to_tensor, make_lq, psnr, ssim, lpips_fn, sha256_file, sha256_bytes,
)
from train_dfpir_budget import DFPIRBlind, tiled_infer  # noqa: E402
from net.model import ChannelShuffle_skip_textguaid  # noqa: E402

DEVICE = torch.device("cuda")


def load_dfpir(ckpt_path):
    base = ChannelShuffle_skip_textguaid(
        dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
    model = DFPIRBlind(base).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="DIV2K,LSDIR")
    ap.add_argument("--seed-salt", default="v1")
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    args = ap.parse_args()
    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

    rows = [json.loads(l) for l in open(ROOT / args.selection)]
    if args.gpu_id is not None:
        rows = [r for i, r in enumerate(rows) if i % args.num_gpus == args.gpu_id]
    datasets = {
        "DIV2K": sorted(glob.glob(str(ROOT / "datasets/DIV2K/DIV2K_valid_HR/*.png"))),
        "LSDIR": sorted(glob.glob(str(ROOT / "datasets/LSDIR/val1/HR/val/*.png"))),
    }
    out = open(ROOT / args.out, "a")
    for row in rows:
        bid = row["scenario_id"]
        pipeline = json.load(open(ROOT / "exp41" / "degradation_gt" / f"{bid}_params.json"))["pipeline"]
        model = load_dfpir(ROOT / row["checkpoint"])
        for ds_name in args.datasets.split(","):
            for img_path in datasets[ds_name]:
                image_id = __import__("pathlib").Path(img_path).stem
                clean_np = load_rgb(img_path)
                lq_np = make_lq(clean_np, bid, f"{ds_name}/{image_id}", pipeline, args.seed_salt)
                d_t, c_t = to_tensor(lq_np).unsqueeze(0).cuda(), to_tensor(clean_np).cuda()
                pred = tiled_infer(model, d_t)[0]
                rec = {
                    "run_id": row["run_id"], "scenario_id": bid,
                    "controller": row.get("controller", "DFPIR-ft"),
                    "round": row.get("round"), "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row.get("checkpoint_sha256"),
                    "checkpoint_step": row.get("checkpoint_step"),
                    "dataset": ds_name, "image_id": image_id,
                    "image_hash": sha256_file(img_path),
                    "lq_sha256": sha256_bytes(lq_np.tobytes()),
                    "metric_version": "psnr_rgb_v1|ssim_gauss11x11_s1.5|lpips_vgg|dfpir_tile512_ov64",
                    "psnr_rgb": psnr(pred, c_t),
                    "ssim_rgb": ssim(pred, c_t),
                    "lpips_vgg": float(lpips_fn()(pred.unsqueeze(0) * 2 - 1,
                                                   c_t.unsqueeze(0) * 2 - 1).item()),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            print(f"[dfpir-eval] {bid} {ds_name} opt{row.get('checkpoint_step')} done", flush=True)
        del model
        torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
