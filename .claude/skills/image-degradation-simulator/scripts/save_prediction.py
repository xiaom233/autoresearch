#!/usr/bin/env python3
"""Save predicted degradation params with enforced evidence checks.

Gatekeeper: refuses to save GOOD verdicts unless mandatory checks are documented.
Forces agents to complete the SKILL.md checklist before claiming success.

Usage:
    # GOOD with evidence file (recommended):
    python save_prediction.py output.json blur:3,jpeg:2 --verdict GOOD --evidence /tmp/ev.json

    # GOOD with inline PSNR (simple deterministic case):
    python save_prediction.py output.json blur:3 --verdict GOOD --psnr 52.3

    # NEEDS_WORK always allowed:
    python save_prediction.py output.json blur:3,noise:2 --verdict NEEDS_WORK

    # Force bypass (only for NEEDS_WORK):
    python save_prediction.py output.json blur:3 --verdict NEEDS_WORK --force

Evidence JSON format:
{
    "psnr_db": 45.2,           // required for deterministic-only pipelines
    "noise_checks": {           // required if pipeline contains noise_*
        "impulse_pct": 0.12,
        "speckle_vm_slope": -0.003,
        "poisson_var_slope": 0.5,
        "spatial_corr": 0.08,
        "ycrcb_rgb_ratio": 1.05,
        "gaussian": true
    },
    "var_slope_on_residual": 0.3,  // required if unique_G < 25 (quantization vs Poisson)
    "radial_ratio": 1.05,          // recommended if blur present
    "dir_change_pct": 2.1          // recommended if blur present
}
"""
import json, sys, os


def is_noise(func_name):
    return func_name.startswith("noise_")


def is_blur(func_name):
    return func_name.startswith("blur_")


def is_compression(func_name):
    return func_name.startswith("compression_")


def is_deterministic(func_name):
    """Non-random degradations where PSNR verification works."""
    # All noise functions are non-deterministic (random seed)
    # blur_glass and blur_jitter use shuffle_pixels_njit (random)
    non_deterministic = {
        "blur_glass", "blur_jitter",
        "noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_speckle",
        "noise_spatially_correlated", "noise_poisson", "noise_impulse",
    }
    return func_name not in non_deterministic


def validate(pipeline, verdict, evidence):
    """Return (ok, errors, warnings)."""
    errors = []
    warnings = []

    if verdict == "NEEDS_WORK":
        return True, [], []  # always allow NEEDS_WORK

    # GOOD verdict — enforce checks
    funcs = [s["function"] for s in pipeline]
    has_noise = any(is_noise(f) for f in funcs)
    has_blur = any(is_blur(f) for f in funcs)
    has_compression = any(is_compression(f) for f in funcs)
    all_deterministic = all(is_deterministic(f) for f in funcs)

    # 1. PSNR check — required for deterministic pipelines
    if all_deterministic:
        psnr = evidence.get("psnr_db")
        if psnr is None:
            errors.append(
                "缺少 PSNR: 确定性退化管线必须提供 PSNR 验证（--psnr 或 evidence 中 psnr_db）。"
                "无法达到 >40dB 请用 --verdict NEEDS_WORK。"
            )
        elif isinstance(psnr, (int, float)) and psnr < 35:
            warnings.append(
                f"PSNR={psnr:.1f}dB < 35dB: 确定性退化应 >40dB。"
                "如已尽力请用 --verdict NEEDS_WORK。"
            )

    # 2. Noise 6-item checklist — required if noise present
    if has_noise:
        noise_checks = evidence.get("noise_checks")
        if not noise_checks:
            errors.append(
                "缺少噪声 6 项检查: 含 noise 的管线必须提供 noise_checks。"
                "需要: impulse_pct, speckle_vm_slope, poisson_var_slope, "
                "spatial_corr, ycrcb_rgb_ratio, gaussian (true/false)"
            )
        else:
            required_keys = ["impulse_pct", "speckle_vm_slope", "poisson_var_slope",
                           "spatial_corr", "ycrcb_rgb_ratio"]
            missing = [k for k in required_keys if k not in noise_checks]
            if missing:
                errors.append(f"噪声检查不完整: 缺少 {', '.join(missing)}")

    # 3. Quantization vs Poisson check
    # We can't know unique_G from here, but if agent provides var_slope_on_residual
    # and it's high, we should warn about Poisson
    var_slope = evidence.get("var_slope_on_residual")
    if var_slope is not None and isinstance(var_slope, (int, float)) and var_slope > 1.0:
        has_quant = any("quantization" in f for f in funcs)
        if has_quant:
            errors.append(
                f"残差 var_slope={var_slope:.2f} > 1.0: 疑似 Poisson 噪声而非 quantization！"
                "请确认已排除 Poisson 后再判 quantization。"
            )

    # 4. Blur checks — recommended
    if has_blur:
        if "radial_ratio" not in evidence and "dir_change_pct" not in evidence:
            warnings.append(
                "建议提供 radial_ratio 和 dir_change_pct 以支持 blur 子类型判断。"
            )

    # 5. Compression masking check — for dual degradation
    if len(funcs) >= 2 and not has_compression:
        psnr = evidence.get("psnr_db")
        if psnr is not None and isinstance(psnr, (int, float)) and psnr < 35:
            warnings.append(
                f"双退化 PSNR={psnr:.1f}dB < 35dB 且无 compression: "
                "compression 是最容易被掩盖的退化（exp17 发现）。"
                "建议执行掩盖推理检查（测试 JPEG/JPEG2000）。"
            )

    ok = len(errors) == 0
    return ok, errors, warnings


def main():
    if len(sys.argv) < 3:
        print("Usage: save_prediction.py <output.json> <func1:sev1> [func2:sev2] ... "
              "[--verdict GOOD|NEEDS_WORK] [--evidence <json_file>] [--psnr <dB>] [--force]")
        sys.exit(1)

    output = sys.argv[1]
    args = sys.argv[2:]

    pipeline = []
    verdict = "NEEDS_WORK"
    evidence_file = None
    psnr_val = None
    force = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--verdict" and i + 1 < len(args):
            verdict = args[i + 1]; i += 2
        elif arg == "--evidence" and i + 1 < len(args):
            evidence_file = args[i + 1]; i += 2
        elif arg == "--psnr" and i + 1 < len(args):
            try:
                psnr_val = float(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif arg == "--force":
            force = True; i += 1
        elif arg.startswith("--ci"):
            i += 2  # skip legacy flag
        elif ":" in arg and not arg.startswith("--"):
            func, sev = arg.split(":")
            pipeline.append({"function": func, "severity": int(sev)})
            i += 1
        else:
            i += 1

    # Load evidence
    evidence = {}
    if evidence_file and os.path.exists(evidence_file):
        try:
            with open(evidence_file) as f:
                evidence = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[save_prediction] WARNING: 无法读取 evidence 文件: {e}")
    if psnr_val is not None:
        evidence.setdefault("psnr_db", psnr_val)

    # Validate (skip if force)
    if not force:
        ok, errors, warnings = validate(pipeline, verdict, evidence)
    else:
        ok, errors, warnings = True, [], []
        print("[save_prediction] --force: 跳过所有检查")

    # Print warnings
    for w in warnings:
        print(f"[save_prediction] ⚠️  {w}")

    # Print errors and reject if any
    if not ok:
        print(f"\n[save_prediction] ❌ 无法保存 GOOD — 以下检查未完成:\n")
        for e in errors:
            print(f"  ✗ {e}")
        print(f"\n  完成上述检查后重试，或使用 --verdict NEEDS_WORK 保存。")
        print(f"  用法: --evidence /tmp/evidence.json 提供检查数据")
        print(f"  证据格式示例:")
        print(f'  {{"psnr_db": 45.2, "noise_checks": {{"impulse_pct": 0.1, ...}}, ...}}')
        sys.exit(1)

    # Save
    result = {
        "pipeline": pipeline,
        "analysis": {
            "verdict": verdict,
            "evidence": evidence,
        }
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    # Auto-version: never overwrite
    if os.path.exists(output):
        base = output.replace('.json', '')
        v = 2
        while os.path.exists(f"{base}_v{v}.json"):
            v += 1
        output = f"{base}_v{v}.json"
        print(f"[save_prediction] versioned: {output}")

    with open(output, 'w') as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"[save_prediction] ✅ 已保存: {output} (verdict={verdict})")


if __name__ == "__main__":
    main()
