"""
Autoresearch image restoration training script. Single-GPU, single-file.
Usage: uv run train.py

import shutil
Reads clean images via WebDataset, applies a degradation pipeline on-the-fly
(from params.json), and trains a restoration model to recover clean images.
"""

import os, sys, contextlib
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import random as _random

from prepare import load_degradation_params, make_dataloader_restoration, scandir
from x_distortion import add_distortion
from model import RestoreNet

# Monkey-patch: category-balanced random pipeline (blur/noise/compression equal weight)
import prepare as _prepare
_DEG_CATS = {
    "blur": ["blur_gaussian", "blur_motion", "blur_glass", "blur_lens", "blur_zoom", "blur_jitter"],
    "noise": ["noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_speckle",
              "noise_spatially_correlated", "noise_poisson", "noise_impulse"],
    "compression": ["compression_jpeg", "compression_jpeg_2000"],
}
def _balanced_random_pipeline():
    n = _random.choices([1, 2, 3], weights=[0.33, 0.33, 0.34])[0]
    cats = list(_DEG_CATS.keys())
    pipeline = []
    used_cats = set()    # 禁止同一类别出现两次
    used_funcs = set()   # 禁止同一函数出现两次
    for _ in range(n):
        available_cats = [c for c in cats if c not in used_cats]
        if not available_cats:
            break
        cat = _random.choice(available_cats)
        funcs = [f for f in _DEG_CATS[cat] if f not in used_funcs]
        func = _random.choice(funcs)
        sev = _random.randint(1, 5)
        pipeline.append((func, sev))
        used_cats.add(cat)
        used_funcs.add(func)
    return pipeline
_prepare.generate_random_pipeline = _balanced_random_pipeline

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — the agent modifies this section)
# ---------------------------------------------------------------------------

# Data
# Training degradation:
#   None = random pipeline per sample (blind restoration baseline)
#   "path/to/params.json" = fixed degradation from skill output (targeted)
PARAMS_PATH = "params.json"
TRAIN_SHARDS = "datasets/DIV2K/DIV2K_train_HR_wds/train-*.tar"

# Validation: standard SR benchmarks + DIV2K valid. Uses HR/original clean images
# degraded with VAL_PARAMS_PATH (ground-truth degradation, NOT skill prediction).
# 训练时中间验证: Set14 quick val (始终) + AR_CHALLENGE_DIR 挑战图 (如设置)
# GT 重评估 Phase 5.5 单独执行，使用 DIV2K + LSDIR + GT 退化参数
VAL_DIRS = []
VAL_COUNT = 0
CHALLENGE_DIR = os.environ.get("AR_CHALLENGE_DIR", "")  # 盲识别挑战 clean 图目录

# Validation degradation: specific degradation to test against.
# For blind restoration baseline (PARAMS_PATH=None), this tests a specific target.
# Set to None to use same random degradation as training.
VAL_PARAMS_PATH = "params.json"  # test M_spec on target degradation

# Model architecture (SwinIR)
ATTENTION_TYPE = "swin"        # "swin" / "mdta" (channel) / "ocab" (overlap spatial)
EMBED_DIM = 64
DEPTHS = (2, 2, 2, 2)       # Swin blocks per RSTB stage
NUM_HEADS = (4, 4, 4, 4)     # attention heads per stage
WINDOW_SIZE = 8
MLP_RATIO = 2

# Training
EPOCH_BUDGET = 1              # epochs to train (0 = use TIME_BUDGET fallback)
TIME_BUDGET = 600             # fallback: training wall-clock budget in seconds
CHECKPOINT_INTERVAL = 4       # save/validate N times per epoch
BATCH_SIZE = 16
NUM_WORKERS = 4
SHUFFLE_BUFFER = 1000
MAX_STEPS = 0                 # 0=auto (EPOCH_BUDGET/TIME_BUDGET), >0=直接指定步数
TRAIN_CROP = 128              # 训练随机裁剪尺寸, 默认128 (原图256→随机裁剪128)

# Optimization
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.999)
WARMUP_STEPS = 100
LR_SCHEDULE = "constant"       # "cosine" or "constant"

# Loss function: "l1", "mse", "huber", "l1+mse", "l1+fft", "l1+edge"
LOSS_FN = "l1"
HUBER_DELTA = 0.1
GRAD_CLIP = 0.0              # 0 = no gradient clipping
AMP_DTYPE = "bfloat16"       # autocast dtype: "bfloat16" or "float32"
LOSS_WEIGHTS = (1.0, 0.1)    # (primary, auxiliary) weight for combined losses

# 激活函数: "gelu", "relu", "silu", "swiglu"
ACTIVATION = "gelu"

# 退化增广: None（不增广）, "jitter"（severity ±1）, "shuffle"（打乱顺序）,
#           "jitter+shuffle"（两者结合）, "random_steps"（随机步数）
DEG_AUGMENT = None

# 课程学习：多阶段退化切换 "step:params,step:params,..." (params="random" → None)
CURRICULUM_CONFIG = None

# 混合难度预热: "step:pct1:pct2,step:..." (百分比, 替代 CURRICULUM_CONFIG)
# 双退化: "0:70:30,2500:40:60,5000:10:90,7547:0:100"
# 三退化: "0:50:30:20,2500:25:35:40,5000:5:20:75,7547:0:0:100"
MIXED_WARMUP = None

# 自适应 Phase 切换: 基于 EMA loss 斜率检测 (0=禁用, >0=平滑窗口)
# 当 EMA loss 在窗口内改善 < 阈值时自动切下一phase
ADAPTIVE_SWITCH = 0          # 检查间隔步数 (推荐500)
ADAPTIVE_THRESHOLD = 0.005   # 相对改善率阈值 (推荐0.005=0.5%)
ADAPTIVE_PATIENCE = 3        # 连续平坦次数 (推荐3, 防噪声)
ADAPTIVE_WARMUP = 500        # 新phase后等待步数 (推荐500)

# Fine-tune 阶段 2 策略参数（配合 CURRICULUM_CONFIG 使用）
REPLAY_RATIO = 0.0        # 阶段 2+ 混入随机退化的比例 (0.0-1.0)
FREEZE_STAGES = 0          # 阶段 2+ 冻结前 N 个 RSTB 层 + conv_first
PHASE2_LR_MULT = 1.0       # 阶段 2+ 的 LR 倍率 (< 1.0 = 更小 LR)

# 高级架构参数（Phase 5 第二批实验）
WINDOW_SHIFT_RATIO = 0.5     # 窗口位移比例（0.5=标准半窗位移, 0.25=四分之一）
HEAD_DIM = 16                # 注意力头维度（0=由 EMBED_DIM/NUM_HEADS 推导）
NORM_TYPE = "layernorm"      # 归一化类型: "layernorm", "batchnorm", "instancenorm"
SKIP_RSTB = "standard"       # RSTB 跳跃连接: "standard", "none", "learnable", "dense"
CONV_KERNEL = 3              # 卷积核大小
CHANNEL_MIX = "none"         # 通道混合: "none", "se", "eca", "layerscale", "spatial_gate"
STAGE_CONFIG = "uniform"     # 层级配置: "uniform", "ascending", "descending", "pyramid"
NUM_STAGES = 4               # RSTB 层级数量

# 结构优化组件开关（exp10 验证）
USE_GCM = False               # FiLM-GCM: 全局通道调制 (contrast+结构化 +3.52 dB, +26K)
USE_CSN = False               # CSN: 通道 Shuffle 归一化 (通用稳定, +1K)
USE_COLOR_PRE = False         # ColorPre: 低分辨率颜色预处理 (通用稳, +0.8K)
USE_COLOR_MLP = False         # ColorMLP: 逐像素颜色 MLP (brightness_HSV, +0.1K)
USE_PCP = False               # PCP shared: 层级间通道扰动 (待充分验证, +3K)
USE_FREQMOD = False           # FreqMod: FFT 频域调制 (❌ 已废弃, SF8 -9.67, +276K)
USE_CHANNEL_CURVE = False     # ChannelCurve: 通道曲线校正 (stretch +0.38, +0.1K)
USE_DUAL_BRANCH = False       # DualBranch: 全局分支调制 (鲁棒性最强, +3K)
USE_COLOR_MLP_OUTPUT = False  # ColorMLP Output: 输出层颜色校正 (+0.1K)
USE_SIMPLE_GATE = False       # SimpleGate: 通道对半分→相乘替代GELU (0参数, NAFNet ECCV2022)
USE_SCA = False               # SCA: GAP→L2Norm→通道缩放 (~0参数, NAFNet ECCV2022)
USE_GDFN = False              # GDFN: 门控DWConv FFN (X-Restormer 2024, +17K)
USE_FPRO = False              # FProLite: FFT频域门控调制 (FPro ECCV2024, +2K)

# Logging
LOG_INTERVAL = 25            # steps between PSNR logging
CKPT_PREFIX = "ckpt"         # 多实验并行时 checkpoint 文件名前缀
QUIET_PIPELINE = 0            # 1=抑制 prepare.py 打印退化管线（防盲识别泄露）
LOAD_CKPT = None              # 盲预训练 checkpoint 路径（用于 Ft 微调）
SKIP_FINAL_VAL = 0            # 1=跳过训练结束后的全量验证（验证的是预测退化非GT，意义不大）

# Environment variable overrides for parallel experiments
for _v in ("PARAMS_PATH", "VAL_PARAMS_PATH", "EMBED_DIM", "BATCH_SIZE", "LEARNING_RATE",
           "WEIGHT_DECAY", "WARMUP_STEPS", "TIME_BUDGET", "EPOCH_BUDGET",
           "MAX_STEPS",
           "CHECKPOINT_INTERVAL", "WINDOW_SIZE", "MLP_RATIO",
           "LOG_INTERVAL", "NUM_WORKERS", "SHUFFLE_BUFFER", "TRAIN_CROP",
           "LR_SCHEDULE", "VAL_COUNT", "DEPTHS", "NUM_HEADS", "ADAM_BETAS",
           "LOSS_FN", "HUBER_DELTA", "GRAD_CLIP", "LOSS_WEIGHTS", "AMP_DTYPE",
           "ACTIVATION", "DEG_AUGMENT", "CKPT_PREFIX", "ATTENTION_TYPE",
           "WINDOW_SHIFT_RATIO", "HEAD_DIM", "NORM_TYPE", "SKIP_RSTB",
           "CONV_KERNEL", "CHANNEL_MIX", "STAGE_CONFIG", "NUM_STAGES", "QUIET_PIPELINE",
           "LOAD_CKPT", "SKIP_FINAL_VAL",
           "CURRICULUM_CONFIG", "REPLAY_RATIO", "FREEZE_STAGES", "PHASE2_LR_MULT",
           "ADAPTIVE_SWITCH", "ADAPTIVE_THRESHOLD",
           "USE_GCM", "USE_CSN", "USE_COLOR_PRE", "USE_COLOR_MLP",
           "USE_PCP", "USE_FREQMOD", "USE_CHANNEL_CURVE", "USE_DUAL_BRANCH",
           "USE_COLOR_MLP_OUTPUT", "USE_SIMPLE_GATE", "USE_SCA",
           "USE_GDFN", "USE_FPRO"):
    _env = os.environ.get(f"AR_{_v}")
    if _env is not None:
        if _v in ("PARAMS_PATH", "VAL_PARAMS_PATH"):
            globals()[_v] = None if _env == "None" else _env
        elif _v in ("LR_SCHEDULE", "LOSS_FN", "AMP_DTYPE",
                  "ACTIVATION", "DEG_AUGMENT", "CKPT_PREFIX", "ATTENTION_TYPE",
                  "NORM_TYPE", "SKIP_RSTB", "CHANNEL_MIX", "STAGE_CONFIG",
                  "CURRICULUM_CONFIG", "MIXED_WARMUP", "LOAD_CKPT"):
            globals()[_v] = _env
        else:
            globals()[_v] = eval(_env)

# ---------------------------------------------------------------------------
# Stdout suppression for pipeline loading (防盲识别泄露)
# ---------------------------------------------------------------------------
import contextlib
@contextlib.contextmanager
def quiet_pipeline():
    """Context manager to suppress pipeline prints from prepare.py.
    Used when AR_QUIET_PIPELINE=1 to prevent ground truth leakage into training logs."""
    import io
    if QUIET_PIPELINE:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
    else:
        yield

# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class ValDataset:
    """Load full validation images from one or more directories, apply degradation.

    Returns (degraded, clean) float32 tensors in [0, 1], shape [3, H, W].
    Images are kept at their original resolution for per-image PSNR evaluation.
    """

    def __init__(self, val_dirs, count=0, params_path=None):
        if isinstance(val_dirs, str):
            val_dirs = [val_dirs]
        self.img_paths = []
        for d in val_dirs:
            paths = sorted(scandir(d, full_path=True))
            if count and len(paths) > count:
                paths = paths[:count]
            self.img_paths.extend(paths)
        self.pipeline = load_degradation_params(params_path) if params_path else []
        print(f"ValDataset: {len(self.img_paths)} images from {len(val_dirs)} dir(s)")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"), dtype=np.uint8)
        clean = img.copy()
        for func_name, severity in self.pipeline:
            img = add_distortion(img, severity=severity, distortion_name=func_name)

        degraded_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return degraded_t, clean_t


RGB_TO_YCRCB = torch.tensor([[0.299, 0.587, 0.114],
                                  [-0.169, -0.331, 0.500],
                                  [0.500, -0.419, -0.081]])

def rgb_to_y(pred, target):
    """PSNR on Y channel (BT.601). Input [B,3,H,W] in [0,1]."""
    w = RGB_TO_YCRCB[0].to(pred.device).view(1, 3, 1, 1)
    return (pred * w).sum(dim=1, keepdim=True), (target * w).sum(dim=1, keepdim=True)

def compute_psnr(pred, target):
    """PSNR in dB. Assumes pixel values in [0, 1]."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())

def _gaussian_kernel(size=11, sigma=1.5, channels=1):
    """Create 2D Gaussian kernel [channels, 1, size, size]."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)

_SSIM_KERNEL = None

def compute_ssim(pred, target):
    """GPU SSIM for a single image [C,H,W] in [0,1]. Returns scalar."""
    global _SSIM_KERNEL
    C = pred.shape[0]
    if _SSIM_KERNEL is None or _SSIM_KERNEL.shape[0] != C:
        _SSIM_KERNEL = _gaussian_kernel(11, 1.5, C).to(pred.device)
    kernel = _SSIM_KERNEL
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # Add batch dim: [1, C, H, W]
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    mu_x = F.conv2d(pred, kernel, groups=C, padding=5)
    mu_y = F.conv2d(target, kernel, groups=C, padding=5)
    mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
    sigma_x2 = F.conv2d(pred ** 2, kernel, groups=C, padding=5) - mu_x2
    sigma_y2 = F.conv2d(target ** 2, kernel, groups=C, padding=5) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, groups=C, padding=5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()

@torch.no_grad()
def evaluate(model, val_dataset, device, autocast_ctx):
    """Compute PSNR/SSIM in RGB and Y over full validation set.
    直接用 DataLoader 并行加载（跳过串行预加载，大幅加速大图验证）。
    """
    eval_model = getattr(model, '_orig_mod', model)  # uncompiled for variable-size eval
    eval_model.eval()
    n = len(val_dataset)
    if n == 0:
        return {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}

    # 直接用 DataLoader 并行加载（num_workers=4 并行解码+退化）
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                          num_workers=4, pin_memory=True,
                                          persistent_workers=False)
    metrics = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
    for degraded, clean in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        with autocast_ctx:
            pred = eval_model(degraded)
        pred_f = pred.float()
        clean_f = clean.float()
        metrics["psnr_rgb"] += compute_psnr(pred_f, clean_f)
        pred_y, clean_y = rgb_to_y(pred_f, clean_f)
        metrics["psnr_y"] += compute_psnr(pred_y, clean_y)
        metrics["ssim_rgb"] += compute_ssim(pred_f[0], clean_f[0])
        metrics["ssim_y"] += compute_ssim(pred_y[0], clean_y[0])
    model.train()
    return {k: v / n for k, v in metrics.items()}

def fmt_metrics(m):
    return (f"PSNR_RGB={m['psnr_rgb']:.2f} PSNR_Y={m['psnr_y']:.2f} "
            f"SSIM_RGB={m['ssim_rgb']:.4f} SSIM_Y={m['ssim_y']:.4f}")

def evaluate_all(model, val_sets, device, autocast_ctx):
    """Evaluate per-dataset and overall. Returns (per_set_dict, overall_dict)."""
    per_set = {}
    overall = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0}
    total_n = 0
    for name, ds in val_sets:
        m = evaluate(model, ds, device, autocast_ctx)
        n = len(ds)
        per_set[name] = m
        for k in overall:
            overall[k] += m[k] * n
        total_n += n
    if total_n > 0:
        for k in overall:
            overall[k] /= total_n
    return per_set, overall



def main():
    global MAX_STEPS, USE_TIME_BUDGET, VAL_INTERVAL
# ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    # === 实验元数据日志 (防bug, 可追溯) ===
    import datetime
    _exp_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ckpt_prefix": CKPT_PREFIX,
        "params_path": PARAMS_PATH,
        "val_params_path": VAL_PARAMS_PATH,
        "curriculum_config": CURRICULUM_CONFIG,
        "mixed_warmup": MIXED_WARMUP,
        "model": {"embed_dim": EMBED_DIM, "depths": list(DEPTHS),
                  "attention_type": ATTENTION_TYPE, "activation": ACTIVATION,
                  "window_size": WINDOW_SIZE,
                  "gcm": USE_GCM, "csn": USE_CSN,
                  "color_pre": USE_COLOR_PRE, "color_mlp": USE_COLOR_MLP,
                  "pcp": USE_PCP, "dual_branch": USE_DUAL_BRANCH,
                  "simple_gate": USE_SIMPLE_GATE, "sca": USE_SCA,
                  "gdfn": USE_GDFN, "fpro": USE_FPRO,
                  "channel_curve": USE_CHANNEL_CURVE},
        "training": {"epoch_budget": EPOCH_BUDGET, "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE, "loss_fn": LOSS_FN},
    }
    _exp_meta["degradation_pipeline"] = load_degradation_params(PARAMS_PATH) if PARAMS_PATH else "random"
    print(f"=== EXP_META: {json.dumps(_exp_meta, ensure_ascii=False, default=str)} ===")

    print(f"AMP dtype: {AMP_DTYPE}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Params: {PARAMS_PATH}")
    print(f"Train shards: {TRAIN_SHARDS}")

    # Validation: quick (Set14 only, for intermediate checkpoints) + per-dataset full sets
    VAL_QUICK_DIRS = ["datasets/Set14/GTmod4"]
    with quiet_pipeline():
        val_quick = ValDataset(VAL_QUICK_DIRS, count=0, params_path=VAL_PARAMS_PATH)
        val_full_sets = [(d.split("/")[-2], ValDataset([d], count=VAL_COUNT, params_path=VAL_PARAMS_PATH))
                          for d in VAL_DIRS]
        # Add challenge images as validation set (if AR_CHALLENGE_DIR is set)
        if CHALLENGE_DIR:
            import glob as _glob
            challenge_tmp = f"/tmp/{os.path.basename(CHALLENGE_DIR)}_val"
            os.makedirs(challenge_tmp, exist_ok=True)
            for src in sorted(_glob.glob(f"{CHALLENGE_DIR}/clean_*.png")):
                dst = os.path.join(challenge_tmp, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            challenge_clean = [f for f in _glob.glob(f"{CHALLENGE_DIR}/clean_*.png")
                              if not f.endswith("clean.png")]  # skip backward-compat copy
            if challenge_clean:
                val_full_sets.append(("Challenge", ValDataset([challenge_tmp], count=0, params_path=VAL_PARAMS_PATH)))
    total_val = sum(len(v[1]) for v in val_full_sets)
    print(f"Val quick: {len(val_quick)} images  |  Val full: {total_val} images across {len(val_full_sets)} sets")

    # Model
    model = RestoreNet(in_ch=3, embed_dim=EMBED_DIM, depths=DEPTHS,
                        num_heads=NUM_HEADS, window_size=WINDOW_SIZE, mlp_ratio=MLP_RATIO,
                        activation=ACTIVATION, head_dim=HEAD_DIM,
                        stage_config=STAGE_CONFIG, num_stages=NUM_STAGES,
                        window_shift_ratio=WINDOW_SHIFT_RATIO, skip_type=SKIP_RSTB,
                        conv_kernel=CONV_KERNEL, attention_type=ATTENTION_TYPE,
                        use_gcm=USE_GCM, use_csn=USE_CSN,
                        use_color_pre=USE_COLOR_PRE, use_color_mlp=USE_COLOR_MLP,
                        use_pcp=USE_PCP, use_freqmod=USE_FREQMOD,
                        use_channel_curve=USE_CHANNEL_CURVE, use_dual_branch=USE_DUAL_BRANCH,
                        use_color_mlp_output=USE_COLOR_MLP_OUTPUT,
                        use_simple_gate=USE_SIMPLE_GATE, use_sca=USE_SCA,
                        use_gdfn=USE_GDFN, use_fpro=USE_FPRO)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    # 盲预训练 checkpoint 加载（用于 Ft 微调）
    if LOAD_CKPT:
        ckpt = torch.load(LOAD_CKPT, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {LOAD_CKPT} (step={ckpt.get('step', '?')})", flush=True)

    model = torch.compile(model, dynamic=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)

    # Training dataloader
    # 退化增广：对固定管线做 severity jitter / 顺序打乱 / 随机步数
    _train_params_path = PARAMS_PATH
    if DEG_AUGMENT is not None and PARAMS_PATH is not None:
        with quiet_pipeline():
            _base_pipeline = _prepare.load_degradation_params(PARAMS_PATH)
        print(f"退化增广模式: {DEG_AUGMENT}（基础管线: {PARAMS_PATH}）")
        def _augmented_pipeline():
            import copy
            pipe = copy.deepcopy(_base_pipeline)  # list of (func, sev) tuples
            if "jitter" in (DEG_AUGMENT or ""):
                pipe = [(func, max(1, min(5, sev + _random.choice([-1, 0, 1]))))
                        for func, sev in pipe]
            if "shuffle" in (DEG_AUGMENT or ""):
                _random.shuffle(pipe)
            if DEG_AUGMENT == "random_steps":
                n = _random.choice([1, 2, 3])
                pipe = pipe[:n]
            return pipe
        _prepare.generate_random_pipeline = _augmented_pipeline
        _train_params_path = None  # 使用随机模式（触发 per-sample augmentation）

    # --- 课程学习/混合预热：预创建多阶段 DataLoader ---
    _phase_loaders = None
    _phase_frozen = False  # 是否已执行冻结操作
    if MIXED_WARMUP is not None:
        # 混合难度预热：每阶段按概率混合不同复杂度
        full_pipeline = _prepare.load_degradation_params(PARAMS_PATH)
        _phase_loaders = []
        _orig_gen = _prepare.generate_random_pipeline  # 保存原始
        for p in MIXED_WARMUP.split(","):
            parts = p.split(":")
            step_start = int(parts[0])
            probs = [float(x) / 100.0 for x in parts[1:]]
            n_levels = len(probs)
            full_n = len(full_pipeline)
            # 为每阶段创建独立生成器（闭包捕获当前 probs）
            def _make_mixed_gen(probs=probs, full_n=full_n, full=full_pipeline,
                               n_levels=n_levels):
                import random as _rnd
                def _mixed():
                    r = _rnd.random()
                    cum = 0.0
                    for i, p in enumerate(probs):
                        cum += p
                        if r < cum:
                            steps = full_n - n_levels + 1 + i
                            return full[:steps]
                    return full
                return _mixed
            _prepare.generate_random_pipeline = _make_mixed_gen()
            with quiet_pipeline():
                ldr = make_dataloader_restoration(
                    params_path=None,  # 随机模式, 使用 mixed gen
                    shards_url=TRAIN_SHARDS,
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    shuffle_buffer=SHUFFLE_BUFFER, train_crop=TRAIN_CROP,
                )
            _phase_loaders.append((step_start, ldr))
        _prepare.generate_random_pipeline = _orig_gen  # 恢复
        _phase_loaders.sort(key=lambda t: t[0])
        train_loader = _phase_loaders[0][1]
        print(f"Mixed Warmup: {len(_phase_loaders)} phases at steps "
              f"{[s for s,_ in _phase_loaders]}")
    elif CURRICULUM_CONFIG is not None:
        _phase_loaders = []
        for p in CURRICULUM_CONFIG.split(","):
            step_start, ppath = p.split(":", 1)
            ppath = ppath.strip()
            _is_random_phase = ppath in ("random", "None")
            _params = None if _is_random_phase else ppath
            # Replay: 非随机阶段混入随机退化
            if REPLAY_RATIO > 0.0 and not _is_random_phase:
                _target_pipeline = _prepare.load_degradation_params(ppath)
                def _make_replay_pipeline(tgt=_target_pipeline):
                    import random as _rnd
                    def _replay():
                        if _rnd.random() < REPLAY_RATIO:
                            return _prepare.generate_random_pipeline()
                        return tgt
                    return _replay
                _prepare.generate_random_pipeline = _make_replay_pipeline()
                _params = None  # 使用随机模式触发 per-sample 调用
            with quiet_pipeline():
                ldr = make_dataloader_restoration(
                    params_path=_params, shards_url=TRAIN_SHARDS,
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    shuffle_buffer=SHUFFLE_BUFFER, train_crop=TRAIN_CROP,
                )
            _phase_loaders.append((int(step_start), ldr))
        _phase_loaders.sort(key=lambda t: t[0])
        train_loader = _phase_loaders[0][1]
        strat_desc = []
        if REPLAY_RATIO > 0: strat_desc.append(f"replay={REPLAY_RATIO}")
        if FREEZE_STAGES > 0: strat_desc.append(f"freeze={FREEZE_STAGES}")
        if PHASE2_LR_MULT < 1.0: strat_desc.append(f"lr_mult={PHASE2_LR_MULT}")
        extra = f" ({', '.join(strat_desc)})" if strat_desc else ""
        print(f"Curriculum: {len(_phase_loaders)} phases at steps "
              f"{[s for s,_ in _phase_loaders]}{extra}")
    else:
        train_loader = make_dataloader_restoration(
            params_path=_train_params_path,
            shards_url=TRAIN_SHARDS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle_buffer=SHUFFLE_BUFFER,
            train_crop=TRAIN_CROP,
        )

    x, y, epoch = next(train_loader)
    x, y = x.to(device), y.to(device)

    print(f"Train: x {tuple(x.shape)} [{x.min():.3f},{x.max():.3f}], "
          f"y {tuple(y.shape)} [{y.min():.3f},{y.max():.3f}]")
    # Training regime
    TOTAL_PATCHES = 120765
    STEPS_PER_EPOCH = TOTAL_PATCHES // BATCH_SIZE  # 7548
    if MAX_STEPS > 0:
        # 直接指定步数（用于专家训练等精确步数场景）
        VAL_INTERVAL = max(1, MAX_STEPS // CHECKPOINT_INTERVAL)
        USE_TIME_BUDGET = False
    elif EPOCH_BUDGET > 0:
        MAX_STEPS = int(EPOCH_BUDGET * STEPS_PER_EPOCH)
        VAL_INTERVAL = max(1, MAX_STEPS // CHECKPOINT_INTERVAL)
        USE_TIME_BUDGET = False
    else:
        # TIME_BUDGET mode: estimate steps from budget, apply checkpoint interval
        est_steps = int(TIME_BUDGET / 0.35)  # rough estimate at ~350ms/step
        VAL_INTERVAL = max(1, est_steps // CHECKPOINT_INTERVAL) if CHECKPOINT_INTERVAL > 0 else 10 ** 9
        MAX_STEPS = 10 ** 9
        USE_TIME_BUDGET = True

    print(f"Regime: {'time_budget='+str(TIME_BUDGET)+'s' if USE_TIME_BUDGET else 'epoch_budget='+str(EPOCH_BUDGET)+' ('+str(MAX_STEPS)+' steps)'}")
    print(f"Checkpoints: {CHECKPOINT_INTERVAL} per epoch, VAL_INTERVAL={VAL_INTERVAL}")
    print(f"Batch: {BATCH_SIZE}  LR: {LEARNING_RATE}  Loss: {LOSS_FN}  Clip: {GRAD_CLIP}")

    # ---------------------------------------------------------------------------
    # Loss helpers
    # ---------------------------------------------------------------------------

    def _sobel_kernels():
        """Return (kx, ky) Sobel kernels [1, 1, 3, 3]."""
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.
        return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)

    def edge_loss(pred, target):
        """Sobel edge-aware loss (L1 on gradient magnitude difference)."""
        # 转灰度以适配单通道 Sobel kernel
        w = torch.tensor([0.299, 0.587, 0.114], device=pred.device).view(1, 3, 1, 1)
        pred_gray = (pred * w).sum(dim=1, keepdim=True)
        target_gray = (target * w).sum(dim=1, keepdim=True)
        kx, ky = _sobel_kernels()
        kx, ky = kx.to(pred.device), ky.to(pred.device)
        gx_p = F.conv2d(pred_gray, kx, padding=1)
        gy_p = F.conv2d(pred_gray, ky, padding=1)
        gx_t = F.conv2d(target_gray, kx, padding=1)
        gy_t = F.conv2d(target_gray, ky, padding=1)
        mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-6)
        mag_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-6)
        return F.l1_loss(mag_p, mag_t)

    def fft_loss(pred, target):
        """Frequency-domain L1 loss on magnitude spectrum."""
        pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
        target_f = torch.fft.rfft2(target.float(), norm='ortho')
        return F.l1_loss(pred_f.abs(), target_f.abs())

    def compute_loss(pred, target):
        """Flexible loss dispatch based on LOSS_FN."""
        if LOSS_FN == "l1":
            return F.l1_loss(pred, target)
        if LOSS_FN == "mse":
            return F.mse_loss(pred, target)
        if LOSS_FN == "huber":
            return F.huber_loss(pred, target, delta=HUBER_DELTA)
        if LOSS_FN == "l1+mse":
            w1, w2 = LOSS_WEIGHTS
            return w1 * F.l1_loss(pred, target) + w2 * F.mse_loss(pred, target)
        if LOSS_FN == "l1+fft":
            w1, w2 = LOSS_WEIGHTS
            return w1 * F.l1_loss(pred, target) + w2 * fft_loss(pred, target)
        if LOSS_FN == "l1+edge":
            w1, w2 = LOSS_WEIGHTS
            return w1 * F.l1_loss(pred, target) + w2 * edge_loss(pred, target)
        return F.l1_loss(pred, target)

    # ---------------------------------------------------------------------------
    # LR schedule
    # ---------------------------------------------------------------------------

    def get_lr_multiplier(step, total_steps):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        if LR_SCHEDULE == "cosine":
            progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    t_start_training = time.time()
    smooth_loss = 0
    total_training_time = 0
    step = 0
    best_train_psnr = 0
    best_checkpoint = {"step": 0, "metrics": None}
    checkpoint_steps = set(range(VAL_INTERVAL, MAX_STEPS + 1, VAL_INTERVAL))

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        # Forward + backward
        with autocast_ctx:
            pred = model(x)
            loss = compute_loss(pred, y)

        loss.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR update
        lrm = get_lr_multiplier(step, MAX_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lrm

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Fetch next batch (with curriculum phase switching)
        if _phase_loaders is not None:
            next_step = step + 1
            for start_step, ldr in reversed(_phase_loaders):
                if next_step >= start_step:
                    if train_loader is not ldr:
                        print(f"\n  [Curriculum] Phase switch at step "
                              f"{next_step}")
                        train_loader = ldr
                        # 冻结浅层（仅在第一次非随机阶段切换时执行）
                        if FREEZE_STAGES > 0 and not _phase_frozen:
                            _m = getattr(model, '_orig_mod', model)
                            _m.conv_first.requires_grad_(False)
                            for _si in range(FREEZE_STAGES):
                                for _p in _m.layers[_si].parameters():
                                    _p.requires_grad_(False)
                            # 重建 optimizer（仅优化未冻结参数）
                            optimizer = torch.optim.AdamW(
                                [p for p in model.parameters() if p.requires_grad],
                                lr=LEARNING_RATE * PHASE2_LR_MULT,
                                betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
                            _phase_frozen = True
                            print(f"  [Freeze] conv_first + {FREEZE_STAGES} RSTB stages frozen, "
                                  f"LR={LEARNING_RATE * PHASE2_LR_MULT:.1e}")
                        # 降低阶段 2 LR（不与 freeze 重复，freeze 已重建 optimizer）
                        elif PHASE2_LR_MULT < 1.0 and not _phase_frozen:
                            for group in optimizer.param_groups:
                                group['lr'] = LEARNING_RATE * PHASE2_LR_MULT
                            # 短 warmup
                            for group in optimizer.param_groups:
                                group['initial_lr'] = group['lr']
                            _phase_frozen = True  # 复用标志避免重复
                            print(f"  [LR] Phase2 LR={LEARNING_RATE * PHASE2_LR_MULT:.1e}")
                    break
        x, y, epoch = next(train_loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Skip warmup steps for timing
        if step > 5:
            total_training_time += dt

        # Logging
        loss_f = loss.item()
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        # === 自适应 Phase 切换 (patience-based, 防误触发) ===
        # 机制: 新phase前WARMUP步不检查 → 每SWITCH步计算loss相对改善率 →
        #       连续PATIENCE次低改善才切换 (避免噪声误触发)
        _adp = getattr(main, '_adp_state', None)
        _adp_warmup = ADAPTIVE_WARMUP
        _adp_patience = ADAPTIVE_PATIENCE
        _adp_ema_beta = 0.98
        if ADAPTIVE_SWITCH > 0 and _phase_loaders is not None:
            if _adp is None:
                _adp = {'ema': smooth_loss, 'ema_prev': smooth_loss,
                        'check_at': ADAPTIVE_SWITCH + _adp_warmup,
                        'flat_count': 0, 'phase_switched': False}
                main._adp_state = _adp
            _adp['ema'] = _adp_ema_beta * _adp['ema'] + (1 - _adp_ema_beta) * loss_f
            if step >= _adp['check_at']:
                rel_improve = (_adp['ema_prev'] - _adp['ema']) / max(_adp['ema_prev'], 1e-6)
                _adp['ema_prev'] = _adp['ema']
                _adp['check_at'] = step + ADAPTIVE_SWITCH
                if rel_improve < ADAPTIVE_THRESHOLD:
                    _adp['flat_count'] += 1
                    if _adp['flat_count'] >= _adp_patience:
                        for start_step, ldr in _phase_loaders:
                            if start_step > step and train_loader is not ldr:
                                print(f"\n  [Adaptive] Phase switch at step {step}"
                                      f" (flat={_adp['flat_count']}/{_adp_patience}"
                                      f", rel_improve={rel_improve:.4%})")
                                train_loader = ldr
                                # 重置: warmup + EMA
                                _adp['ema'] = smooth_loss
                                _adp['ema_prev'] = smooth_loss
                                _adp['check_at'] = step + ADAPTIVE_SWITCH + _adp_warmup
                                _adp['flat_count'] = 0
                                break
                else:
                    _adp['flat_count'] = max(0, _adp['flat_count'] - 1)  # 改善恢复→减计数(非直接清零)

        if step % LOG_INTERVAL == 0:
            with torch.no_grad(), autocast_ctx:
                train_psnr = compute_psnr(pred.float(), y)
                best_train_psnr = max(best_train_psnr, train_psnr)

        # Checkpoint: save + quick validation on Set14 only
        if step > 0 and step in checkpoint_steps:
            ckpt_dir = f"{CKPT_PREFIX}/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = os.path.basename(CKPT_PREFIX)
            ckpt_file = f"{ckpt_dir}/{ckpt_name}_step{step}.pt"
            torch.save({"step": step, "model": getattr(model, '_orig_mod', model).state_dict()}, ckpt_file)
            m = evaluate(model, val_quick, device, autocast_ctx)
            print(f"\n  [Ckpt @ step {step}/{MAX_STEPS}] Set14: {fmt_metrics(m)}", flush=True)
            if best_checkpoint["metrics"] is None or m["psnr_rgb"] > best_checkpoint["metrics"]["psnr_rgb"]:
                best_checkpoint = {"step": step, "metrics": m}

        if USE_TIME_BUDGET:
            progress = min(total_training_time / TIME_BUDGET, 1.0)
        else:
            progress = min(step / MAX_STEPS, 1.0)
        pct = 100 * progress
        imgs_per_sec = int(BATCH_SIZE / dt) if dt > 0 else 0

        print(f"\rstep {step:05d}/{MAX_STEPS} ({pct:.1f}%) | loss: {debiased_loss:.6f} | "
              f"lr: {LEARNING_RATE * lrm:.2e} | dt: {dt*1000:.0f}ms | imgs/s: {imgs_per_sec} | "
              f"epoch: {epoch}    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if USE_TIME_BUDGET:
            if step > 10 and total_training_time >= TIME_BUDGET:
                break
        else:
            if step >= MAX_STEPS:
                break

    print()

    # ---------------------------------------------------------------------------
    # Final evaluation — load best checkpoint, run on ALL validation datasets
    # ---------------------------------------------------------------------------

    if best_checkpoint["metrics"] is not None:
        ckpt_dir = f"{CKPT_PREFIX}/checkpoints"
        ckpt_name = os.path.basename(CKPT_PREFIX)
        ckpt_file = f"{ckpt_dir}/{ckpt_name}_step{best_checkpoint['step']}.pt"
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
            getattr(model, '_orig_mod', model).load_state_dict(ckpt["model"])
            print(f"\nLoaded best checkpoint: step {best_checkpoint['step']} "
                  f"(Set14 PSNR_RGB={best_checkpoint['metrics']['psnr_rgb']:.2f})", flush=True)

    if SKIP_FINAL_VAL:
        print(f"Skipping final validation (SKIP_FINAL_VAL=1). GT reeval should be done separately.", flush=True)
        per_set = {}
        final_metrics = {"psnr_rgb": 0.0, "psnr_y": 0.0, "ssim_rgb": 0.0, "ssim_y": 0.0, "psnr_rgb_avg": 0.0}
    else:
        print(f"Running full validation on {total_val} images across {len(val_full_sets)} sets...", flush=True)
        per_set, final_metrics = evaluate_all(model, val_full_sets, device, autocast_ctx)
    total_imgs = step * BATCH_SIZE

    # Print per-dataset table (skip if SKIP_FINAL_VAL)
    if not SKIP_FINAL_VAL:
        print(f"\n{'Dataset':<14} {'PSNR_RGB':>10} {'PSNR_Y':>10} {'SSIM_RGB':>10} {'SSIM_Y':>10}")
        print("-" * 58)
        for name in [v[0] for v in val_full_sets]:
            m = per_set[name]
            print(f"{name:<14} {m['psnr_rgb']:>10.2f} {m['psnr_y']:>10.2f} {m['ssim_rgb']:>10.4f} {m['ssim_y']:>10.4f}")
        print("-" * 58)
        print(f"{'Overall':<14} {final_metrics['psnr_rgb']:>10.2f} {final_metrics['psnr_y']:>10.2f} {final_metrics['ssim_rgb']:>10.4f} {final_metrics['ssim_y']:>10.4f}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print("---")
    print(f"best_step:         {best_checkpoint['step']}")
    print(f"psnr_rgb:          {final_metrics['psnr_rgb']:.2f}")
    print(f"psnr_y:            {final_metrics['psnr_y']:.2f}")
    print(f"ssim_rgb:          {final_metrics['ssim_rgb']:.4f}")
    print(f"ssim_y:            {final_metrics['ssim_y']:.4f}")
    print(f"best_train_psnr:   {best_train_psnr:.2f}")
    print(f"final_loss:        {loss.item():.6f}")
    print(f"training_seconds:  {total_training_time:.1f}")
    print(f"total_seconds:     {t_end - t_start:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_steps:         {step}")
    print(f"num_params_M:      {num_params / 1e6:.1f}")
    print(f"total_imgs:        {total_imgs}")
    print(f"batch_size:        {BATCH_SIZE}")
    print(f"learning_rate:     {LEARNING_RATE}")
    print(f"val_psnr_db:       {final_metrics['psnr_rgb']:.2f}")

    # 自动保存结构化结果（JSON + 追加 CSV）
    results = {
        "ckpt_prefix": CKPT_PREFIX,
        "params_path": PARAMS_PATH,
        "val_params_path": VAL_PARAMS_PATH,
        "config": {
            "embed_dim": EMBED_DIM, "depths": list(DEPTHS), "num_heads": list(NUM_HEADS),
            "window_size": WINDOW_SIZE, "mlp_ratio": MLP_RATIO,
            "activation": ACTIVATION, "lr_schedule": LR_SCHEDULE,
            "loss_fn": LOSS_FN, "grad_clip": GRAD_CLIP,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        },
        "per_dataset": {name: per_set.get(name, {}) for name in [v[0] for v in val_full_sets]},
        "overall": final_metrics,
        "summary": {
            "best_step": best_checkpoint["step"],
            "best_train_psnr": best_train_psnr,
            "final_loss": loss.item(),
            "training_seconds": total_training_time,
            "total_seconds": t_end - t_start,
            "peak_vram_mb": peak_vram_mb,
            "num_steps": step,
            "num_params_M": num_params / 1e6,
            "total_imgs": total_imgs,
            "batch_size": BATCH_SIZE,
            "val_psnr_db": final_metrics['psnr_rgb'],
        },
    }
    # 保存 JSON
    results_dir = f"{CKPT_PREFIX}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/{os.path.basename(CKPT_PREFIX)}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"results_json:      {results_path}")


if __name__ == "__main__":
    main()
