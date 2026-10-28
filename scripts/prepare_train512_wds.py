#!/usr/bin/env python3
"""Build 512x512 WebDataset shards from DIV2K_train_HR and LSDIR train_HR.

Crop 512x512 with stride 256 (the same 50% overlap as the existing 256/128
datasets), PNG compression 3, 1000 samples per shard. DIV2K is used in full;
LSDIR is subsampled uniformly per patch so the mixed set lands near
``--target-total`` samples. Outputs the two source datasets first, then a
mixed directory of symlinks named ``train-%06d.tar`` plus ``index_plan.json``
so it is a drop-in replacement for DIV2K_LSDIR_train_wds.
"""
import argparse
import glob
import hashlib
import json
import os
import tarfile
import time
import multiprocessing as mp

import cv2
import numpy as np
import webdataset as wds
from tqdm import tqdm

CROP = 512
STEP = 256
PNG_LEVEL = 3
SHARD_SIZE = 1000
BATCH_SAMPLES = 32
# Patch-level estimate of the full 512/256 LSDIR crop count (sampled from raw).
EST_LSDIR_FULL_PATCHES = 1_120_000

DIV2K_SRC = "/data/zyli/datasets/DIV2K/DIV2K_train_HR"
DIV2K_OUT = "/data/zyli/datasets/DIV2K/DIV2K_train_HR_512_wds"
LSDIR_GLOB = "/data/zyli/datasets/LSDIR/raw/train_HR/shard-*.tar.gz"
LSDIR_OUT = "/data/zyli/datasets/LSDIR/LSDIR_train_HR_512_wds"
MIXED_OUT = "/data/zyli/datasets/DIV2K_LSDIR_train512_wds"
IMG_EXTS = (".png", ".jpg", ".jpeg")


def crop_origins(size):
    origins = np.arange(0, size - CROP + 1, STEP)
    if len(origins) and size - (origins[-1] + CROP) > 0:
        origins = np.append(origins, size - CROP)
    return origins


def keep_key(key, sample_rate):
    if sample_rate >= 1.0:
        return True
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2 ** 64 < sample_rate


def encode_patches(img, prefix, sample_rate):
    h, w = img.shape[:2]
    if h < CROP or w < CROP:
        return []
    items = []
    index = 0
    for y in crop_origins(h):
        for x in crop_origins(w):
            index += 1
            key = f"{prefix}_s{index:03d}"
            if not keep_key(key, sample_rate):
                continue
            crop = np.ascontiguousarray(img[y:y + CROP, x:x + CROP])
            ok, buf = cv2.imencode(
                ".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
            if ok:
                items.append((key, buf.tobytes()))
    return items


def div2k_worker(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []
    stem = os.path.splitext(os.path.basename(path))[0]
    return encode_patches(img, stem, 1.0)


def lsdirc_worker(shard_path, sample_rate, queue):
    n_img = n_kept = n_total = n_skip = 0
    batch = []
    try:
        with tarfile.open(shard_path, "r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                ext = os.path.splitext(member.name)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    n_skip += 1
                    continue
                img = cv2.imdecode(
                    np.frombuffer(fobj.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    n_skip += 1
                    continue
                n_img += 1
                h, w = img.shape[:2]
                if h < CROP or w < CROP:
                    n_skip += 1
                    continue
                stem = os.path.splitext(os.path.basename(member.name))[0]
                index = 0
                for y in crop_origins(h):
                    for x in crop_origins(w):
                        index += 1
                        n_total += 1
                        key = f"lsdir_{stem}_s{index:03d}"
                        if not keep_key(key, sample_rate):
                            continue
                        crop = np.ascontiguousarray(img[y:y + CROP, x:x + CROP])
                        ok, buf = cv2.imencode(
                            ".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
                        if not ok:
                            continue
                        batch.append((key, buf.tobytes()))
                        n_kept += 1
                        if len(batch) >= BATCH_SAMPLES:
                            queue.put(("batch", batch))
                            batch = []
        if batch:
            queue.put(("batch", batch))
        queue.put(("done", os.path.basename(shard_path), n_img, n_kept,
                   n_total, n_skip, ""))
    except Exception as exc:  # noqa: BLE001 - report worker failure to parent
        queue.put(("done", os.path.basename(shard_path), n_img, n_kept,
                   n_total, n_skip, repr(exc)))


def shard_counts(n_samples):
    full, rem = divmod(n_samples, SHARD_SIZE)
    counts = [SHARD_SIZE] * full
    if rem:
        counts.append(rem)
    return counts


def build_div2k(out_dir, workers):
    paths = sorted(glob.glob(os.path.join(DIV2K_SRC, "*.png")))
    if not paths:
        raise SystemExit(f"No images under {DIV2K_SRC}")
    pattern = os.path.join(out_dir, "train-%06d.tar")
    total = 0
    seen = set()
    pbar = tqdm(total=len(paths), unit="image", desc="DIV2K", mininterval=5)
    with wds.ShardWriter(pattern, maxcount=SHARD_SIZE, verbose=0) as sink:
        with mp.Pool(min(workers, len(paths))) as pool:
            for items in pool.imap_unordered(div2k_worker, paths, chunksize=1):
                for key, blob in items:
                    if key in seen:
                        raise RuntimeError(f"Duplicate sample key: {key}")
                    seen.add(key)
                    sink.write({"__key__": key, "png": blob})
                    total += 1
                pbar.update(1)
    pbar.close()
    print(f"DIV2K: images={len(paths)} patches={total}", flush=True)
    return total


def build_lsdir(out_dir, sample_rate, n_workers):
    shards = sorted(glob.glob(LSDIR_GLOB))
    if not shards:
        raise SystemExit(f"No archives match {LSDIR_GLOB}")
    n_workers = min(n_workers, len(shards))

    queue = mp.Queue(maxsize=32)
    procs = [
        mp.Process(target=lsdirc_worker, args=(p, sample_rate, queue))
        for p in shards
    ]
    for proc in procs:
        proc.start()

    pattern = os.path.join(out_dir, "lsdir-train-%06d.tar")
    total = 0
    seen = set()
    finished = 0
    failures = []
    start = time.time()

    pbar = tqdm(unit="patch", desc="LSDIR", mininterval=10)
    try:
        with wds.ShardWriter(pattern, maxcount=SHARD_SIZE, verbose=0) as sink:
            while finished < n_workers:
                item = queue.get()
                if item[0] == "done":
                    _, name, n_img, n_kept, n_total, n_skip, err = item
                    finished += 1
                    if err:
                        failures.append((name, err))
                        print(f"ERROR {name}: {err}", flush=True)
                    else:
                        print(f"done {name}: images={n_img} kept={n_kept} "
                              f"crops={n_total} skipped={n_skip}", flush=True)
                    continue
                for key, blob in item[1]:
                    if key in seen:
                        raise RuntimeError(f"Duplicate sample key: {key}")
                    seen.add(key)
                    sink.write({"__key__": key, "png": blob})
                    total += 1
                    pbar.update(1)
    finally:
        pbar.close()
        for proc in procs:
            proc.join()

    elapsed = time.time() - start
    print(f"LSDIR: patches={total} elapsed={elapsed / 60:.1f} min", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} archive(s) failed")
    return total


def build_mixed(div2k_dir, lsdir_dir, div2k_counts, lsdir_counts):
    os.makedirs(MIXED_OUT)
    names = []
    counts = []
    index = 0
    for src_dir, prefix, src_counts in (
            (div2k_dir, "train", div2k_counts),
            (lsdir_dir, "lsdir-train", lsdir_counts)):
        for i, count in enumerate(src_counts):
            src = os.path.join(src_dir, f"{prefix}-{i:06d}.tar")
            name = f"train-{index:06d}.tar"
            os.symlink(src, os.path.join(MIXED_OUT, name))
            names.append(name)
            counts.append(count)
            index += 1
    with open(os.path.join(MIXED_OUT, "index_plan.json"), "w") as f:
        json.dump({"shards": names, "counts": counts}, f)
    print(f"mixed: shards={len(names)} samples={sum(counts)} -> {MIXED_OUT}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=700_000,
                        help="target total sample count for the mixed set")
    parser.add_argument("--lsdir-sample-rate", type=float, default=0.0,
                        help="override the automatically chosen LSDIR rate")
    parser.add_argument("--div2k-workers", type=int, default=16)
    parser.add_argument("--lsdir-workers", type=int, default=0,
                        help="concurrent archives (0 = all)")
    args = parser.parse_args()

    for out in (DIV2K_OUT, LSDIR_OUT, MIXED_OUT):
        if os.path.exists(out):
            raise SystemExit(f"Output path already exists: {out}")

    div2k_tmp = DIV2K_OUT + "_tmp"
    lsdir_tmp = LSDIR_OUT + "_tmp"
    for tmp in (div2k_tmp, lsdir_tmp):
        os.makedirs(tmp)  # raises if a previous run left a partial directory

    div2k_total = build_div2k(div2k_tmp, args.div2k_workers)
    div2k_counts = shard_counts(div2k_total)

    rate = args.lsdir_sample_rate
    if rate <= 0:
        rate = (args.target_total - div2k_total) / EST_LSDIR_FULL_PATCHES
        rate = max(0.0, min(1.0, rate))
    print(f"LSDIR sample rate: {rate:.4f}", flush=True)

    lsdirc_total = build_lsdir(lsdir_tmp, rate, args.lsdir_workers or 17)
    lsdirc_counts = shard_counts(lsdirc_total)

    os.rename(div2k_tmp, DIV2K_OUT)
    os.rename(lsdir_tmp, LSDIR_OUT)
    build_mixed(DIV2K_OUT, LSDIR_OUT, div2k_counts, lsdirc_counts)


if __name__ == "__main__":
    main()
