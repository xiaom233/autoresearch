#!/usr/bin/env python3
"""
test_candidate.py — 一步完成退化施加 + 验证 + reflection.json 更新

替代 apply_multi.py + compare_degradation.py/verify_signals.py 的组合调用。
Agent 只需一条命令完成候选测试，工具自动维护 reflection.json。

用法:
  # PSNR 模式 (sigma < 2)
  .venv/bin/python3 test_candidate.py \
    --target degraded.png --clean clean.png \
    --pipeline "blur_gaussian:3,compression_jpeg:1" \
    --thinking "Round 1: gaussian blur 覆盖 baseline" \
    --reflection reflection.json

  # SIGNAL 模式 (sigma >= 2)
  .venv/bin/python3 test_candidate.py \
    --target degraded.png --clean clean.png \
    --pipeline "noise_poisson:2" \
    --thinking "Round 1: 信号报告推荐 POISSON" \
    --reflection reflection.json --mode signal

  # 自动模式 (默认, 从 reflection.json 的 mode 字段判断)
  .venv/bin/python3 test_candidate.py \
    --target degraded.png --clean clean.png \
    --pipeline "blur_motion:3" \
    --thinking "R1: motion blur severity sweep" \
    --reflection reflection.json
"""

import argparse, json, os, sys, subprocess, re, time
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)


def apply_degradation(input_path, pipeline_str, output_path):
    """Apply degradation pipeline using x_distortion."""
    from x_distortion import add_distortion

    pairs = []
    for part in pipeline_str.split(","):
        part = part.strip()
        if not part: continue
        name, sev = part.rsplit(":", 1)
        pairs.append((name.strip(), int(sev)))

    img = np.array(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    for name, severity in pairs:
        img = add_distortion(img, severity, name)

    Image.fromarray(img).save(output_path)
    return len(pairs)


def compute_psnr(target_path, simulated_path):
    """Compute PSNR (numpy) + CI metrics (compare_degradation.py)."""
    target = np.array(Image.open(target_path).convert("RGB"), dtype=np.float64)
    simulated = np.array(Image.open(simulated_path).convert("RGB"), dtype=np.float64)

    mse = np.mean((target - simulated) ** 2)
    psnr = float('inf') if mse == 0 else round(20 * np.log10(255.0 / np.sqrt(mse)), 2)

    # Run CI comparison
    ci = "?/?"
    ci_verdict = "?"
    ci_details = []
    try:
        r = subprocess.run([
            sys.executable,
            os.path.join(SCRIPT_DIR, "compare_degradation.py"),
            "--target", target_path, "--simulated", simulated_path
        ], capture_output=True, text=True)
        ci_data = json.loads(r.stdout)
        ci = ci_data.get("ci_pass_rate", "?/?")
        ci_verdict = ci_data.get("verdict", "?")

        # Extract key CI metrics for Agent visibility
        for m in ci_data.get("content_independent_metrics", []):
            metric = m.get("metric", "?")
            diff = m.get("difference", 0)
            pct = m.get("pct_error", 0)
            # Determine if this metric passed (threshold-based)
            passed = pct < 20  # rough threshold, compare_degradation has its own logic
            status = "✓" if passed else "✗"
            ci_details.append({
                "metric": metric, "diff": round(diff, 4),
                "pct_err": round(pct, 1), "pass": passed
            })
    except:
        pass

    # Print all CI metrics for Agent (with descriptions)
    CI_DESCRIPTIONS = {
        "impulse_total_pct":    "离群像素比例(检测脉冲噪声)",
        "impulse_pct_zero":     "像素饱和到0的比例(暗部clipping)",
        "impulse_pct_255":      "像素饱和到255的比例(亮部clipping)",
        "block_boundary_ratio": "8x8块边界不连续性(检测JPEG压缩)",
        "unique_R":             "R通道唯一颜色数(检测颜色量化)",
        "unique_G":             "G通道唯一颜色数(检测颜色量化)",
        "unique_B":             "B通道唯一颜色数(检测颜色量化)",
        "overshoot_ratio":      "过冲伪影比例(检测过度锐化)",
        "zero_crossing_density":"零交叉密度(反映纹理/噪声水平)",
        "hf_lf_ratio":          "高频/低频能量比(反映图像锐度)",
    }
    if ci_details:
        print(f"\n  CI metrics ({ci}):")
        for d in ci_details:
            marker = "✓" if d["pass"] else "✗"
            desc = CI_DESCRIPTIONS.get(d['metric'], "")
            print(f"    {marker} {d['metric']}: diff={d['diff']:.3f} ({d['pct_err']:.0f}%)  {desc}")

    return {
        "psnr_rgb": psnr,
        "ci_pass": ci,
        "ci_verdict": ci_verdict,
    }


def verify_signals(target_path, simulated_path, clean_path, pipeline_str):
    """Run verify_signals.py --json: print text table, parse JSON for reflection."""
    r = subprocess.run([
        sys.executable,
        os.path.join(SCRIPT_DIR, "verify_signals.py"),
        "--target", target_path, "--clean", clean_path,
        "--simulated", simulated_path,
        "--pipeline", pipeline_str, "--json"
    ], capture_output=True, text=True)

    # Split output: text table (before last '{') and JSON (from last '{')
    json_start = r.stdout.rfind('\n{')
    if json_start >= 0:
        # Print only the text table (Agent-readable)
        print(r.stdout[:json_start])
        # Parse JSON
        try:
            data = json.loads(r.stdout[json_start:])
            return {
                "method": "verify_signals",
                "verdict": data.get("verdict", "UNKNOWN"),
                "matched_signals": data.get("matches", []),
                "mismatched_signals": data.get("mismatches", []),
            }
        except:
            pass
    else:
        # No JSON found, just print everything
        print(r.stdout, end="")
    return {"method": "verify_signals", "verdict": "ERROR"}


def load_reflection(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_reflection(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Test a degradation candidate and update reflection.json")
    parser.add_argument("--target", required=True, help="Degraded target image")
    parser.add_argument("--clean", required=True, help="Clean reference image")
    parser.add_argument("--pipeline", required=True, help="Degradation pipeline, e.g. 'fn1:sev1,fn2:sev2'")
    parser.add_argument("--thinking", required=True, help="Agent reasoning for this test (信号来源、假设动机)")
    parser.add_argument("--reflection", default=None, help="Path to reflection.json (created if missing)")
    parser.add_argument("--mode", choices=["auto", "psnr", "signal"], default="auto",
                        help="Verification mode: auto(from reflection), psnr, signal")
    parser.add_argument("--round", type=int, default=None, help="Iteration round number (auto-increment if omitted)")
    parser.add_argument("--output-sim", default=None, help="Save simulated image to path (optional)")
    args = parser.parse_args()

    ref = load_reflection(args.reflection) if args.reflection else {}

    # Determine mode
    mode = args.mode
    if mode == "auto":
        ia = ref.get("initial_analysis", {})
        stored_mode = ia.get("mode", "PSNR")
        mode = "signal" if stored_mode.startswith("SIGNAL") else "psnr"

    # Determine round
    round_num = args.round
    if round_num is None:
        iters = ref.get("iterations", [])
        round_num = len(iters) + 1

    # Parse pipeline steps for display
    pipeline_steps = []
    for part in args.pipeline.split(","):
        part = part.strip()
        if not part: continue
        fn, sev = part.rsplit(":", 1)
        pipeline_steps.append((fn.strip(), int(sev)))

    # ── Step A: 施加退化 ──
    print(f"{'='*60}")
    print(f"[Round {round_num}] Pipeline: {args.pipeline}")
    print(f"  Thinking: {args.thinking}")
    print(f"  Steps:")
    for i, (fn, sev) in enumerate(pipeline_steps, 1):
        print(f"    {i}. {fn} (severity={sev})")
    print(f"{'='*60}")

    sim_path = args.output_sim or f"/tmp/test_candidate_{os.getpid()}.png"
    apply_degradation(args.clean, args.pipeline, sim_path)
    print(f"  → 退化已施加, 模拟图: {sim_path}")

    # ── Step B: 验证 ──
    print(f"\n  Verification mode: {mode.upper()}")
    if mode == "signal":
        result = verify_signals(args.target, sim_path, args.clean, args.pipeline)
    else:
        result = compute_psnr(args.target, sim_path)
        result["method"] = "PSNR"
        # Print PSNR summary if not already printed by compare_degradation
        psnr = result.get("psnr_rgb", 0)
        ci = result.get("ci_pass", "?/?")
        print(f"\n  PSNR: {psnr:.2f} dB  |  CI: {ci}")

    result["pipeline"] = args.pipeline
    result["mode"] = mode

    # Build iteration record
    iteration = {
        "round": round_num,
        "hypothesis": {
            "pipeline": args.pipeline,
            "thinking": args.thinking,
        },
        "test_result": result,
    }

    # Determine decision
    if mode == "signal":
        v = result.get("verdict", "")
        if v == "MATCH": iteration["decision"] = "ACCEPT"
        elif v == "PARTIAL": iteration["decision"] = "ADJUST_SEVERITY"
        elif v == "WEAK": iteration["decision"] = "TRY_ALTERNATIVE"
        else: iteration["decision"] = "REJECT"
    else:
        psnr = result.get("psnr_rgb", 0)
        if psnr > 40: iteration["decision"] = "ACCEPT"
        elif psnr > 30: iteration["decision"] = "ADJUST_SEVERITY"
        elif psnr > 20: iteration["decision"] = "TRY_ALTERNATIVE"
        else: iteration["decision"] = "REJECT"

    # Update reflection.json
    if args.reflection:
        if "iterations" not in ref:
            ref["iterations"] = []
        ref["iterations"].append(iteration)

        # Auto-update psnr_ranking
        if "psnr_ranking" not in ref:
            ref["psnr_ranking"] = []
        psnr = result.get("psnr_rgb", None)
        if psnr is not None and psnr > 0:
            ref["psnr_ranking"].append({
                "pipeline": args.pipeline,
                "psnr_rgb": psnr,
                "ci_pass": result.get("ci_pass", "N/A"),
                "verdict": result.get("verdict", "?"),
                "mode": mode,
                "round": round_num,
            })
            ref["psnr_ranking"].sort(key=lambda x: x.get("psnr_rgb", 0), reverse=True)

        save_reflection(args.reflection, ref)

    # Print summary
    if mode == "signal":
        print(f"[R{round_num}] {args.pipeline} | {result.get('verdict','?')} | "
              f"matched: {result.get('matched_signals',[])} | "
              f"mismatched: {result.get('mismatched_signals',[])} | "
              f"→ {iteration['decision']}")
    else:
        print(f"[R{round_num}] {args.pipeline} | PSNR={result.get('psnr_rgb',0):.2f}dB "
              f"CI={result.get('ci_pass','?')} | → {iteration['decision']}")

    if args.reflection:
        print(f"  reflection.json updated ({len(ref['iterations'])} iterations, {len(ref.get('psnr_ranking',[]))} ranked)")


if __name__ == "__main__":
    main()
