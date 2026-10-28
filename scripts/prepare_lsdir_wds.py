#!/usr/bin/env python3
"""Stream-crop LSDIR tar.gz shards into DIV2K-style WebDataset shards.

Crop 256x256 with stride 128 (same as prepare.py for DIV2K), PNG compression 3,
1000 samples per shard. Sample keys are prefixed with ``lsdir_`` and shard names
with ``lsdir-`` so they never collide with DIV2K when datasets are mixed.
"""
import argparse
import glob
import os
import tarfile
import time
import multiprocessing as mp

import cv2
import numpy as np
import webdataset as wds
from tqdm import tqdm

CROP = 256
STEP = 128
PNG_LEVEL = 3
KEY_PREFIX = "lsdir"
IMG_EXTS = {".png", ".jpg", ".jpeg"}
BATCH_SAMPLES = 128


def _crop_origins(size):
    origins = np.arange(0, size - CROP + 1, STEP)
    if len(origins) and size - (origins[-1] + CROP) > 0:
        origins = np.append(origins, size - CROP)
    return origins


def process_shard(shard_path, queue, limit_per_shard=0):
    n_img = n_patch = n_skip = 0
    batch = []
    try:
        with tarfile.open(shard_path, "r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                ext = os.path.splitext(member.name)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                n_img += 1
                if limit_per_shard and n_img > limit_per_shard:
                    break
                fobj = tf.extractfile(member)
                if fobj is None:
                    n_skip += 1
                    continue
                img = cv2.imdecode(np.frombuffer(fobj.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    n_skip += 1
                    continue
                h, w = img.shape[:2]
                if h < CROP or w < CROP:
                    n_skip += 1
                    continue
                stem = os.path.splitext(os.path.basename(member.name))[0]
                index = 0
                for y in _crop_origins(h):
                    for x in _crop_origins(w):
                        index += 1
                        crop = np.ascontiguousarray(img[y:y + CROP, x:x + CROP])
                        ok, buf = cv2.imencode(
                            ".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
                        if not ok:
                            continue
                        batch.append((f"{KEY_PREFIX}_{stem}_s{index:03d}", buf.tobytes()))
                        n_patch += 1
                        if len(batch) >= BATCH_SAMPLES:
                            queue.put(("batch", batch))
                            batch = []
        if batch:
            queue.put(("batch", batch))
        queue.put(("done", os.path.basename(shard_path), n_img, n_patch, n_skip, ""))
    except Exception as exc:  # noqa: BLE001 - report worker failure to parent
        queue.put(("done", os.path.basename(shard_path), n_img, n_patch, n_skip, repr(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="/data/zyli/datasets/LSDIR/raw/train_HR/shard-*.tar.gz")
    parser.add_argument("--out-dir", default="/data/zyli/datasets/LSDIR/LSDIR_train_HR_wds")
    parser.add_argument("--workers", type=int, default=0,
                        help="number of archives processed concurrently (0 = all)")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit-per-shard", type=int, default=0,
                        help="only process the first N images of each archive (debug)")
    args = parser.parse_args()

    shards = sorted(glob.glob(args.input_glob))
    if not shards:
        raise SystemExit(f"No archives match {args.input_glob}")
    workers = args.workers or len(shards)
    workers = min(workers, len(shards))

    os.makedirs(args.out_dir, exist_ok=True)
    if glob.glob(os.path.join(args.out_dir, "*.tar")):
        raise SystemExit(f"Output dir already contains tar shards: {args.out_dir}")

    queue = mp.Queue(maxsize=64)
    procs = [
        mp.Process(target=process_shard, args=(path, queue, args.limit_per_shard))
        for path in shards[:workers]
    ]
    for proc in procs:
        proc.start()

    pattern = os.path.join(args.out_dir, "lsdir-train-%06d.tar")
    total_patch = total_img = total_skip = 0
    seen = set()
    finished = 0
    failures = []
    start = time.time()

    pbar = tqdm(unit="patch", desc="Packing", mininterval=10)
    try:
        with wds.ShardWriter(pattern, maxcount=args.shard_size, verbose=0) as sink:
            while finished < workers:
                item = queue.get()
                if item[0] == "done":
                    _, name, n_img, n_patch, n_skip, err = item
                    finished += 1
                    total_img += n_img
                    total_patch += n_patch
                    total_skip += n_skip
                    if err:
                        failures.append((name, err))
                        print(f"ERROR {name}: {err}", flush=True)
                    else:
                        print(f"done {name}: images={n_img} patches={n_patch} skipped={n_skip}",
                              flush=True)
                    continue
                for key, blob in item[1]:
                    if key in seen:
                        raise RuntimeError(f"Duplicate sample key: {key}")
                    seen.add(key)
                    sink.write({"__key__": key, "png": blob})
                    pbar.update(1)
    finally:
        pbar.close()
        for proc in procs:
            proc.join()

    elapsed = time.time() - start
    print(f"shards={len(glob.glob(os.path.join(args.out_dir, 'lsdir-train-*.tar')))} "
          f"images={total_img} patches={total_patch} skipped={total_skip} "
          f"elapsed={elapsed / 60:.1f} min", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} archive(s) failed")


if __name__ == "__main__":
    main()
