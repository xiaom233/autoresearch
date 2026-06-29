import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import random
import torch
import json
import io
from PIL import Image

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False

# Default degradation categories for random pipeline generation
DEGRADATION_CATEGORIES = {
    "blur": [
        "blur_gaussian", "blur_motion", "blur_glass",
        "blur_lens", "blur_zoom", "blur_jitter",
    ],
    "noise": [
        "noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_speckle",
        "noise_spatially_correlated", "noise_poisson", "noise_impulse",
    ],
    "compression": [
        "compression_jpeg", "compression_jpeg_2000",
    ],
}
_ALL_DEGRADATIONS = [f for funcs in DEGRADATION_CATEGORIES.values() for f in funcs]


def scandir(input_folder, full_path=True):
    """
    扫描文件夹下的所有图像文件，返回文件路径或文件名的列表。

    参数:
        input_folder (str): 要扫描的文件夹路径。
        full_path (bool): 若为 True，返回完整路径；否则只返回文件名。

    返回:
        list: 图像文件的路径/文件名列表。
    """
    # 支持的图像扩展名（可根据需要增删）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    images = []

    try:
        with os.scandir(input_folder) as entries:
            for entry in entries:
                # 仅处理文件，忽略子目录
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in image_extensions:
                        images.append(entry.path if full_path else entry.name)
    except FileNotFoundError:
        print(f"错误：文件夹未找到 -> {input_folder}")
    except PermissionError:
        print(f"错误：没有访问权限 -> {input_folder}")

    return images

def extract_dataset():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    Args:
        opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and
            longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.

            * DIV2K_train_HR
            * DIV2K_train_LR_bicubic/X2
            * DIV2K_train_LR_bicubic/X3
            * DIV2K_train_LR_bicubic/X4

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR images — base crop_size=128 (reduced from 256 for faster training).
    # For aligned LR patches: x2→64, x3→42, x4→32.
    opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_HR'
    opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    opt['crop_size'] = 128
    opt['step'] = 64
    opt['thresh_size'] = 0
    if extract_subimages(opt):
        make_webdataset(opt['save_folder'], 'datasets/DIV2K/DIV2K_train_HR_wds/train-%06d.tar')

    # LRx2 images — crop=256/2=128, step=128/2=64
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    # opt['crop_size'] = 128
    # opt['step'] = 64
    # opt['thresh_size'] = 0
    # if extract_subimages(opt):
    #     make_webdataset(opt['save_folder'], 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub_wds/train-%06d.tar')
    #
    # LRx3 images — crop=256/3≈85, step=128/3≈43 (256 not divisible by 3, use 240 for x3)
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    # opt['crop_size'] = 85
    # opt['step'] = 43
    # opt['thresh_size'] = 0
    # if extract_subimages(opt):
    #     make_webdataset(opt['save_folder'], 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub_wds/train-%06d.tar')

    # LRx4 images — crop=256/4=64, step=128/4=32
    # opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
    # opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    # opt['crop_size'] = 64
    # opt['step'] = 32
    # opt['thresh_size'] = 0
    # if extract_subimages(opt):
    #     make_webdataset(opt['save_folder'], 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub_wds/train-%06d.tar')


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Skip extraction.')
        return False

    img_list = scandir(input_folder, full_path=True)

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')
    return True


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info

def make_webdataset(input_folder, output_pattern, shard_size=1000):
    """Pack cropped sub-images into WebDataset tar shards for efficient streaming I/O.

    Args:
        input_folder (str): Path to folder containing cropped sub-images.
        output_pattern (str): Pattern for shard files, e.g. 'datasets/DIV2K/train-%06d.tar'.
        shard_size (int): Number of samples per shard (tar file).
    """
    if not HAS_WDS:
        raise ImportError("webdataset is required. Install with: pip install webdataset")

    img_list = scandir(input_folder, full_path=True)
    if not img_list:
        print(f"No images found in {input_folder}")
        return

    output_dir = osp.dirname(output_pattern) or '.'
    os.makedirs(output_dir, exist_ok=True)

    with wds.ShardWriter(output_pattern, maxcount=shard_size) as sink:
        for img_path in tqdm(img_list, desc='Packaging WebDataset'):
            basename = osp.splitext(osp.basename(img_path))[0]
            with open(img_path, 'rb') as f:
                sink.write({'__key__': basename, 'png': f.read()})

    n_shards = (len(img_list) + shard_size - 1) // shard_size
    print(f'Created {n_shards} shard(s) from {len(img_list)} images.')


def generate_random_pipeline():
    """Generate a random degradation pipeline using only blur, noise, compression.

    Each sample gets:
      - 1, 2, or 3 degradations with equal probability (0.33 each)
      - Random severity 1–5 per degradation
      - Random function from the three default categories

    Returns:
        list of (function_name, severity) tuples.
    """
    n_degradations = random.choices([1, 2, 3], weights=[0.33, 0.33, 0.34])[0]
    pipeline = []
    for _ in range(n_degradations):
        func = random.choice(_ALL_DEGRADATIONS)
        sev = random.randint(1, 5)
        pipeline.append((func, sev))
    return pipeline


def load_degradation_params(params_path=None):
    """Load degradation pipeline from params.json, or generate random if None.

    The params.json contains a 'pipeline' array where each entry has:
        {"step": 1, "function": "blur_gaussian", "severity": 3, ...}

    Args:
        params_path (str or None): Path to params.json. If None, generates
            a random pipeline from blur/noise/compression categories.

    Returns:
        list of (function_name, severity) tuples, in application order.
    """
    if params_path is None:
        pipeline = generate_random_pipeline()
        print("Generated random degradation pipeline (blur/noise/compression only):")
        for i, (func, sev) in enumerate(pipeline):
            print(f"  Step {i + 1}: {func} (severity={sev})")
        return pipeline

    with open(params_path, 'r') as f:
        params = json.load(f)

    pipeline = []
    for step in params.get('pipeline', []):
        # Handle multiple formats: dict {"function":"fn","severity":N}, dict {"name":"fn","severity":N}, str "fn:N"
        if isinstance(step, str):
            fn, sev = step.rsplit(':', 1)
            pipeline.append((fn.strip(), int(sev)))
        elif isinstance(step, dict):
            fn = step.get('function', step.get('name', ''))
            sev = step.get('severity', 1)
            pipeline.append((fn, int(sev)))

    print(f"Loaded degradation pipeline from {params_path}:")
    for i, (func, sev) in enumerate(pipeline):
        print(f"  Step {i + 1}: {func} (severity={sev})")

    return pipeline


def make_dataloader_gpt(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

def make_dataloader_restoration(params_path=None, shards_url=None, batch_size=16,
                                 num_workers=4, shuffle_buffer=1000,
                                 endless=True):
    """Create a streaming DataLoader for image restoration training.

    Reads clean images from WebDataset shards, applies the degradation pipeline
    from params.json on-the-fly, and yields (inputs, targets, epoch) tuples.

    Args:
        params_path (str or None): Path to params.json. If None, generates a random
            pipeline from blur/noise/compression with random severity (1–5) and
            equal probability for 1/2/3 degradations.
        shards_url (str): Glob pattern for WebDataset shards, e.g. 'datasets/DIV2K/train-*.tar'.
        batch_size (int): Batch size.
        num_workers (int): Number of dataloader worker processes.
        shuffle_buffer (int): WebDataset shuffle buffer size (0 to disable).
        endless (bool): If True, loop forever incrementing epoch. If False, single pass.

    Yields:
        inputs: Degraded images [B, 3, H, W] float32 in [0, 1].
        targets: Clean images [B, 3, H, W] float32 in [0, 1].
        epoch: Current epoch number (starts at 1).
    """
    if not HAS_WDS:
        raise ImportError("webdataset is required. Install with: pip install webdataset")

    from x_distortion import add_distortion

    # When params_path is None, generate a fresh random pipeline per sample.
    # When params_path is provided, use the same fixed pipeline for all samples.
    if params_path is None:
        def get_pipeline():
            return generate_random_pipeline()
    else:
        pipeline = load_degradation_params(params_path)
        def get_pipeline():
            return pipeline

    def decode_and_degrade(sample):
        """Decode raw image bytes, apply degradation pipeline, return (degraded, clean)."""
        for ext in ('png', 'jpg', 'jpeg', 'ppm', 'webp'):
            if ext in sample:
                img = np.array(Image.open(io.BytesIO(sample[ext])).convert('RGB'), dtype=np.uint8)
                clean = img.copy()
                for func_name, severity in get_pipeline():
                    img = add_distortion(img, severity=severity, distortion_name=func_name)
                degraded_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
                return degraded_t, clean_t
        raise KeyError(f"No image key found in sample: {list(sample.keys())}")

    # Expand glob pattern to list (webdataset 1.x compat)
    import glob as _glob
    shards_list = sorted(_glob.glob(shards_url))
    if not shards_list:
        raise FileNotFoundError(f"No shards found matching: {shards_url}")

    epoch = 1
    while True:
        dataset = wds.WebDataset(shards_list, shardshuffle=shuffle_buffer if shuffle_buffer > 0 else False)
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(decode_and_degrade)
        loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for batch in loader:
            yield batch[0], batch[1], epoch

        if not endless:
            break
        epoch += 1


def demo_dataloader():
    """Demo: create a restoration dataloader and fetch one batch.

    Finds the first available params.json from the image-degradation-simulator
    workspace and uses it to apply degradation on-the-fly.
    """
    # Find a params.json from the skill workspace
    workspace = '.claude/skills/image-degradation-simulator-workspace'
    params_files = []
    for root, dirs, files in os.walk(workspace):
        for f in files:
            if f == 'params.json' and 'with_skill' in root:
                params_files.append(osp.join(root, f))
    if not params_files:
        print(f"No params.json found in {workspace}/**/with_skill/")
        return
    params_path = params_files[0]
    print(f"Using params: {params_path}")

    shards_url = 'datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar'
    if not osp.exists(osp.dirname(shards_url)):
        print(f"WebDataset shards not found at {shards_url}. Run extract_dataset() first.")
        return

    loader = make_dataloader_restoration(
        params_path=params_path,
        shards_url=shards_url,
        batch_size=4,
        endless=False,
    )
    for inputs, targets, epoch in loader:
        print(f"Epoch {epoch}: inputs {inputs.shape}, targets {targets.shape}")
        print(f"  input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  target range: [{targets.min():.3f}, {targets.max():.3f}]")
        break
    print("DataLoader demo complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DIV2K dataset preparation and restoration dataloader')
    parser.add_argument('--demo', action='store_true', help='After extraction, demo the restoration dataloader')
    args = parser.parse_args()

    extract_dataset()

    if args.demo:
        demo_dataloader()
