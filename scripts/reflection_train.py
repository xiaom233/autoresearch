#!/usr/bin/env python3
"""
训练驱动反思脚本：Stage 1 训单步模型 → Stage 2 residual 分析 → 生成 v2 预测。

用法:
  .venv/bin/python3 scripts/reflection_train.py \
    --case exp22/challenges/phase4/blind_00002682 \
    --train-step 'blur_lens:3' \
    --ckpt-prefix exp22/experiments/reflection_blind_00002682_stage1

机制:
  Stage 1: 仅训练高置信度那一步 (EPOCH=2)
           model(target) → pred → residual = target - pred
  Stage 2: 在 residual 上跑 run_full_analysis
           分析 pred 中残留的信号 → 确定缺失的第二步
           生成 predicted_params_v2.json
"""

import argparse
import json
import os
import sys
import subprocess
import time
import numpy as np
from PIL import Image
from collections import OrderedDict

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import analysis functions
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".claude", "skills", "image-degradation-simulator", "scripts"))
from run_full_analysis import run_full_analysis, to_native, NumpyEncoder, load_image, compute_block_boundary, compute_gm_ratio

PYTHON = ".venv/bin/python3"


def create_params_file(pipeline_str, output_path):
    """Create a params.json file from a pipeline string like 'blur_lens:3' or 'brightness_darken_shfit_HSV:4'."""
    steps = []
    for part in pipeline_str.split(","):
        part = part.strip()
        if ":" in part:
            func, sev = part.rsplit(":", 1)
            steps.append({"function": func.strip(), "severity": int(sev.strip())})

    params = {"pipeline": steps}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Created params: {output_path} → {params}")
    return output_path


def train_model(params_path, ckpt_prefix, gpu_id=0, epoch_budget=2, time_budget=0, extra_env=None):
    """Train a model with given params. Returns True on success.

    If time_budget > 0, uses TIME_BUDGET mode (EPOCH_BUDGET=0).
    gpu_id: GPU to use (sets CUDA_VISIBLE_DEVICES for subprocess).
    """
    log_file = f"{ckpt_prefix}_train.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["AR_PARAMS_PATH"] = os.path.abspath(params_path)
    env["AR_VAL_PARAMS_PATH"] = os.path.abspath(params_path)
    env["AR_CKPT_PREFIX"] = ckpt_prefix
    env["AR_QUIET_PIPELINE"] = "1"  # 防泄露

    if time_budget > 0:
        env["AR_EPOCH_BUDGET"] = "0"
        env["AR_TIME_BUDGET"] = str(time_budget)
        env["AR_BATCH_SIZE"] = "8"  # smaller batch to avoid torch.compile OOM
        regime_desc = f"time_budget={time_budget}s batch_size=8"
    else:
        env["AR_EPOCH_BUDGET"] = str(epoch_budget)
        regime_desc = f"epoch_budget={epoch_budget}"

    if extra_env:
        for k, v in extra_env.items():
            env[k] = v

    cmd = [
        PYTHON, "train.py"
    ]

    print(f"\n{'='*60}")
    print(f"Stage 1: 训练单步模型")
    print(f"  Params: {params_path}")
    print(f"  Checkpoint: {ckpt_prefix}")
    print(f"  模式: {regime_desc}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}\n")

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT
        )
        proc.wait()

    success = proc.returncode == 0
    print(f"  训练结果: {'成功' if success else '失败'} (exit={proc.returncode})")

    # Parse val_psnr_db from log
    val_psnr = None
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if "val_psnr_db:" in line:
                    try:
                        val_psnr = float(line.strip().split("val_psnr_db:")[-1].split()[0])
                    except:
                        pass

    return success, val_psnr, log_file


def load_best_checkpoint(ckpt_prefix):
    """Find the best checkpoint (highest step number)."""
    ckpt_dir = f"{ckpt_prefix}/checkpoints"
    if not os.path.isdir(ckpt_dir):
        return None

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        return None

    # Sort by step number
    def get_step(fname):
        try:
            return int(fname.split("_step")[-1].replace(".pt", ""))
        except:
            return 0

    ckpts.sort(key=get_step, reverse=True)
    return os.path.join(ckpt_dir, ckpts[0])


def inference_on_image(checkpoint_path, target_path, output_path, device="cuda:0"):
    """Run inference: load model → process target → save pred.

    Build model with default architecture (matches train.py defaults when no
    env var overrides are set).  We import model.py directly to avoid train.py
    side-effects (dataset loading etc.).
    """
    import torch
    from model import RestoreNet

    print(f"\n  加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Reconstruct model with default architecture (same as train.py defaults)
    # embed_dim=64, depths=(2,2,2,2), num_heads=(4,4,4,4), window_size=8
    model = RestoreNet(
        in_ch=3, embed_dim=64, depths=(2, 2, 2, 2),
        num_heads=(4, 4, 4, 4), window_size=8, mlp_ratio=2,
        activation="gelu", head_dim=0, stage_config="uniform",
        num_stages=4, window_shift_ratio=0.5, skip_type="standard",
        conv_kernel=3, attention_type="swin",
        use_gcm=False, use_channel_curve=False, use_color_mlp=False,
        use_freqmod=False, use_color_pre=False, use_dual_branch=False,
        use_pcp=False, use_csn=False, use_color_mlp_output=False,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # Load target and pad to multiple of window_size
    img = np.array(Image.open(target_path).convert("RGB"), dtype=np.float32)
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    _, _, h, w = img_t.shape
    ws = 8  # window_size default
    pad_h = (ws - h % ws) % ws
    pad_w = (ws - w % ws) % ws

    if pad_h > 0 or pad_w > 0:
        img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(img_t)

    pred = pred[:, :, :h, :w]  # crop padding
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

    Image.fromarray(pred).save(output_path)
    print(f"  推理完成: {output_path} ({pred.shape})")
    return pred


def analyze_residual(pred_path, target_path, clean_path):
    """Run run_full_analysis on the prediction vs clean, and residual analysis."""
    print(f"\n  Stage 2: 残差分析")

    # Load images
    target = np.array(Image.open(target_path).convert("RGB"), dtype=np.float64)
    pred = np.array(Image.open(pred_path).convert("RGB"), dtype=np.float64)
    clean = np.array(Image.open(clean_path).convert("RGB"), dtype=np.float64)

    # Residual = target - pred (what model removed)
    residual = target - pred
    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)

    # Save residual visualization
    residual_path = pred_path.replace(".png", "_residual.png")
    Image.fromarray(residual_vis).save(residual_path)
    print(f"  Residual 已保存: {residual_path}")

    # What's still in pred? → Compare pred vs clean
    # Run full analysis on pred (treating pred as "degraded", clean as reference)
    # This tells us what step_B looks like (the step the model couldn't remove)

    # Save pred_temp and run analysis
    tmp_pred = pred_path.replace(".png", "_analysis.png")
    Image.fromarray(np.clip(pred, 0, 255).astype(np.uint8)).save(tmp_pred)

    print(f"\n  对 model 输出 (pred) 跑 run_full_analysis (检测残留信号)...")
    report = run_full_analysis(pred_path, clean_path)
    report = to_native(report)

    # Also analyze the residual
    print(f"\n  对 residual 跑 run_full_analysis (检测被模型移除的信号)...")
    # residual_centered: center around 128 for analysis
    # We'll analyze residual vs zero image (or just compute stats)
    residual_stats = {
        "mean": float(residual.mean()),
        "std": float(residual.std()),
        "min": float(residual.min()),
        "max": float(residual.max()),
    }

    # Block boundary on residual - strong 8x8 pattern = JPEG was removed
    residual_block_boundary = compute_block_boundary(residual_vis)

    # Additional analyses
    # 1. Check if pred has 8x8 blocking (JPEG artifact)
    pred_bb = compute_block_boundary(pred)

    # 2. Check gm_ratio of pred (is there residual blur?)
    pred_gm = compute_gm_ratio(pred)
    clean_gm = compute_gm_ratio(clean)
    pred_gm_ratio = pred_gm / max(clean_gm, 1e-6)

    # 3. Noise in residual
    from run_full_analysis import estimate_noise_sigma
    noise_sigma = estimate_noise_sigma(residual)
    avg_noise_sigma = np.mean(noise_sigma)

    # 4. Global signal in residual
    residual_mean_shift = residual.mean() / 255.0

    analysis = {
        "pred_block_boundary": round(pred_bb, 4),
        "pred_gm_ratio": round(pred_gm_ratio, 4),
        "residual_stats": residual_stats,
        "residual_noise_sigma": round(avg_noise_sigma, 2),
        "residual_block_boundary": round(residual_block_boundary, 4),
        "residual_mean_shift": round(residual_mean_shift * 255, 1),
        "run_full_analysis_on_pred": report,
    }

    return analysis


def diagnose_missing_step(analysis, r0_prediction):
    """Based on residual analysis, determine what the missing step(s) might be."""
    pred_report = analysis["run_full_analysis_on_pred"]
    step1 = pred_report.get("step1_compression", {})
    step2 = pred_report.get("step2_noise", {})
    step3 = pred_report.get("step3_blur", {})
    step4 = pred_report.get("step4_global", {})

    findings = []

    # Check for compression in pred
    bb = analysis["pred_block_boundary"]
    if bb > 1.05:
        findings.append({
            "signal": "compression_jpeg",
            "confidence": "high" if bb > 1.1 else "medium",
            "evidence": f"pred 仍显示 8×8 块效应 (bb={bb:.2f})",
            "action": "add compression_jpeg to pipeline"
        })

    # Check for blur in pred
    gm = analysis["pred_gm_ratio"]
    if gm < 0.85:
        findings.append({
            "signal": "blur",
            "confidence": "high" if gm < 0.62 else "medium",
            "evidence": f"pred 仍显示梯度衰减 (gm_ratio={gm:.4f})",
            "subtype": step3.get("verdict", "UNKNOWN"),
            "action": "add/refine blur step"
        })

    # Check for noise in residual
    sigma = analysis["residual_noise_sigma"]
    if sigma > 1.9:
        findings.append({
            "signal": "noise",
            "confidence": "high" if sigma > 5 else "medium",
            "evidence": f"residual noise sigma={sigma:.1f} > 1.9 阈值",
            "subtype": step2.get("verdict", "UNKNOWN"),
            "action": "add noise step"
        })

    # Check for global in residual
    mean_shift = analysis["residual_mean_shift"]
    if abs(mean_shift) > 3.5:  # ~1.4% of 255
        findings.append({
            "signal": "global",
            "confidence": "medium",
            "evidence": f"residual mean shift={mean_shift:.1f}",
            "action": "add global step"
        })

    # Check run_full_analysis step results
    s1_v = step1.get("verdict", "NO_COMPRESSION")
    s2_v = step2.get("verdict", "NO_NOISE")
    s3_v = step3.get("verdict", "NO_BLUR")
    s4_v = step4.get("verdict", "NO_GLOBAL")

    if s1_v != "NO_COMPRESSION" and not any(f["signal"] == "compression_jpeg" for f in findings):
        findings.append({
            "signal": "compression_jpeg",
            "confidence": "medium",
            "evidence": f"run_full_analysis step1: {s1_v}",
            "action": "add compression_jpeg to pipeline"
        })

    if s3_v != "NO_BLUR" and not any(f["signal"] == "blur" for f in findings):
        findings.append({
            "signal": "blur",
            "confidence": step3.get("confidence", "medium"),
            "evidence": f"run_full_analysis step3: {s3_v}",
            "subtype": s3_v,
            "action": "add/refine blur step"
        })

    if s4_v != "NO_GLOBAL" and not any(f["signal"] == "global" for f in findings):
        findings.append({
            "signal": "global",
            "confidence": step4.get("confidence", "low"),
            "evidence": f"run_full_analysis step4: {s4_v}",
            "action": "add global step"
        })

    return findings


def save_v2_prediction(case_dir, train_step_str, findings, analysis, r0_pred):
    """Save v2 prediction with the complete 2-step pipeline."""
    output_path = os.path.join(case_dir, "predicted_params_v2.json")

    # Determine the complete pipeline
    train_func, train_sev = train_step_str.strip().split(":")
    train_sev = int(train_sev)

    # Determine 2-step pipeline
    pipeline = [{"function": train_func, "severity": train_sev}]

    # Add detected missing step
    for finding in findings:
        if finding["signal"] == "compression_jpeg":
            pipeline.append({
                "function": "compression_jpeg",
                "severity": 3,  # will be refined by PSNR test
                "note": f"Detected from residual: pred_block_boundary={analysis['pred_block_boundary']:.2f}"
            })
        elif finding["signal"] == "blur":
            subtype_map = {
                "GAUSSIAN_BLUR": "blur_gaussian",
                "LENS_BLUR": "blur_lens",
                "MOTION_BLUR": "blur_motion",
                "ZOOM_BLUR": "blur_zoom",
                "GLASS_BLUR": "blur_glass",
                "JITTER_BLUR": "blur_jitter",
            }
            blur_func = subtype_map.get(finding.get("subtype", ""), "blur_gaussian")
            pipeline.append({
                "function": blur_func,
                "severity": 2,
                "note": f"Detected from residual: pred_gm_ratio={analysis['pred_gm_ratio']:.4f}"
            })
        elif finding["signal"] == "noise":
            subtype_map = {
                "GAUSSIAN_RGB": "noise_gaussian_RGB",
                "IMPULSE": "noise_impulse",
                "SPECKLE": "noise_speckle",
                "POISSON": "noise_poisson",
            }
            noise_func = subtype_map.get(finding.get("subtype", ""), "noise_gaussian_RGB")
            pipeline.append({
                "function": noise_func,
                "severity": 2,
                "note": f"Detected from residual: noise_sigma={analysis['residual_noise_sigma']:.1f}"
            })
        elif finding["signal"] == "global":
            pipeline.append({
                "function": "brightness_darken_shfit_HSV",
                "severity": 2,
                "note": f"Detected from residual: mean_shift={analysis['residual_mean_shift']:.1f}"
            })

    # Don't duplicate steps (by function name AND by category)
    seen_funcs = set()
    seen_categories = set()
    # Category mapping
    _blur_funcs = {"blur_lens", "blur_gaussian", "blur_motion", "blur_zoom", "blur_glass", "blur_jitter"}
    _noise_funcs = {"noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_impulse", "noise_speckle", "noise_poisson"}
    _comp_funcs = {"compression_jpeg", "compression_jpeg2000"}
    _global_funcs = {"brightness_darken_shfit_HSV", "brightness_brighten_shfit_HSV", "contrast_weaken_HSV",
                     "contrast_strengthen_HSV", "saturation_weaken_HSV", "saturation_strengthen_HSV",
                     "gamma_brighten_RGB", "gamma_darken_RGB", "quantize"}

    deduped = []
    for step in pipeline:
        func = step["function"]
        if func in seen_funcs:
            continue
        # Determine category
        if func in _blur_funcs: cat = "blur"
        elif func in _noise_funcs: cat = "noise"
        elif func in _comp_funcs: cat = "compression"
        elif func in _global_funcs: cat = "global"
        else: cat = "other"

        if cat in seen_categories and cat != "other":
            continue  # skip duplicate category
        deduped.append(step)
        seen_funcs.add(func)
        seen_categories.add(cat)

    verdict = "UNCERTAIN" if len(findings) > 0 else "BEYOND_CAPABILITY"

    v2 = {
        "pipeline": deduped,
        "analysis": {
            "verdict": verdict,
            "method": "training-driven-reflection",
            "stage1_trained_step": train_step_str,
            "detected_missing": [f["signal"] for f in findings],
            "diagnosis": findings,
            "residual_analysis_summary": {
                "pred_block_boundary": analysis["pred_block_boundary"],
                "pred_gm_ratio": analysis["pred_gm_ratio"],
                "residual_noise_sigma": analysis["residual_noise_sigma"],
                "residual_mean_shift": analysis["residual_mean_shift"],
                "residual_block_boundary": analysis["residual_block_boundary"],
            }
        }
    }

    with open(output_path, "w") as f:
        json.dump(v2, f, indent=2, ensure_ascii=False)
    print(f"\n  V2 预测已保存: {output_path}")
    return v2


def main():
    parser = argparse.ArgumentParser(description="训练驱动反思")
    parser.add_argument("--case", required=True, help="案例目录 (含 degraded.png, clean.png, predicted_params.json)")
    parser.add_argument("--train-step", required=True, help="高置信度单步，如 'blur_lens:3' 或 'brightness_darken_shfit_HSV:4'")
    parser.add_argument("--ckpt-prefix", required=True, help="Checkpoint 前缀")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数 (默认2)")
    parser.add_argument("--time-budget", type=int, default=0, help="时间预算(秒), >0 时覆盖 epoch_budget")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练 (已有 checkpoint)")
    args = parser.parse_args()

    case_dir = os.path.abspath(args.case)
    target_path = os.path.join(case_dir, "degraded.png")
    clean_path = os.path.join(case_dir, "clean.png")
    pred_path = os.path.join(case_dir, "predicted_params.json")

    # Validate inputs
    for p, name in [(target_path, "degraded.png"), (clean_path, "clean.png"), (pred_path, "predicted_params.json")]:
        if not os.path.exists(p):
            print(f"错误: 缺少 {name} at {p}")
            sys.exit(1)

    r0_pred = json.load(open(pred_path))
    print(f"R0 预测: {r0_pred['pipeline']}")
    print(f"R0 verdict: {r0_pred['analysis']['verdict']}")
    print(f"Stage 1 训练步骤: {args.train_step}")

    # Stage 1: Train single-step model
    params_file = os.path.join(case_dir, "params_stage1.json")
    create_params_file(args.train_step, params_file)

    # Extract GPU ID from --device (e.g., "cuda:0" -> 0)
    gpu_id = 0
    if ":" in args.device:
        try:
            gpu_id = int(args.device.split(":")[-1])
        except ValueError:
            pass

    if not args.skip_train:
        success, val_psnr, train_log = train_model(
            params_file, args.ckpt_prefix,
            gpu_id=gpu_id,
            epoch_budget=args.epochs, time_budget=args.time_budget
        )
        if not success:
            print("错误: 训练失败")
            sys.exit(1)
        print(f"  训练 val_psnr_db: {val_psnr}")
    else:
        print("跳过训练 (--skip-train)")

    # Load checkpoint
    ckpt = load_best_checkpoint(args.ckpt_prefix)
    if not ckpt:
        print(f"错误: 找不到 checkpoint at {args.ckpt_prefix}/checkpoints/")
        sys.exit(1)
    print(f"Best checkpoint: {ckpt}")

    # Stage 1 inference
    pred_img_path = os.path.join(case_dir, "pred_stage1.png")
    pred = inference_on_image(ckpt, target_path, pred_img_path, args.device)

    # Stage 2: Residual analysis
    analysis = analyze_residual(pred_img_path, target_path, clean_path)

    print(f"\n{'='*60}")
    print(f"Stage 2: 残差分析结果")
    print(f"{'='*60}")
    print(f"  pred block_boundary (残留JPEG信号): {analysis['pred_block_boundary']:.4f}")
    print(f"  pred gm_ratio (残留blur信号): {analysis['pred_gm_ratio']:.4f}")
    print(f"  residual noise_sigma: {analysis['residual_noise_sigma']:.2f}")
    print(f"  residual mean_shift: {analysis['residual_mean_shift']:.1f}")
    print(f"  residual block_boundary: {analysis['residual_block_boundary']:.4f}")

    # Diagnose
    findings = diagnose_missing_step(analysis, r0_pred)
    print(f"\n  检测到的缺失信号:")
    if findings:
        for f in findings:
            print(f"    - {f['signal']} (置信度: {f['confidence']}): {f['evidence']}")
    else:
        print(f"    (无明确信号 → BEYOND_CAPABILITY)")

    # Save v2
    v2 = save_v2_prediction(case_dir, args.train_step, findings, analysis, r0_pred)

    # Save full analysis
    analysis_path = os.path.join(case_dir, "reflection_stage1_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  完整分析已保存: {analysis_path}")

    print(f"\n{'='*60}")
    print(f"反思完成")
    print(f"  Verdict: {v2['analysis']['verdict']}")
    print(f"  Pipeline: {v2['pipeline']}")
    print(f"  缺失信号: {[f['signal'] for f in findings]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
