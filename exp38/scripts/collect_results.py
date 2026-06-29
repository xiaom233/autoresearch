#!/usr/bin/env python3
"""Collect PSNR results from exp38 experiment logs."""

import os, re, glob

EXP = "exp38"
LOG_DIR = f"{EXP}/logs"

results = {}
for logf in sorted(glob.glob(f"{LOG_DIR}/*.log")):
    name = os.path.basename(logf).replace(".log", "")
    if name.startswith("gpu"):
        continue
    try:
        with open(logf) as f:
            content = f.read()
        match = re.search(r"^val_psnr_db:\s+([\d.]+)", content, re.MULTILINE)
        if match:
            psnr = float(match.group(1))
            results[name] = psnr
        else:
            # Check if training completed but no PSNR line
            if "val_psnr_db:" not in content and "step 15094" in content:
                results[name] = None  # completed but no PSNR
    except Exception as e:
        print(f"  ERROR reading {logf}: {e}")

# Group by phase
phases = {
    "A": "L2 策略-架构解耦",
    "B": "Motion Fwd/Rev 验证",
    "C": "RandomCurric 安全性",
    "D": "True Ft vs RandomCurric",
    "E": "contrast 组件复现",
}

print("=" * 80)
print("exp38 实验结果汇总")
print("=" * 80)

for phase_id, phase_desc in phases.items():
    print(f"\n### Phase {phase_id}: {phase_desc}")
    phase_results = [(k, v) for k, v in results.items() if k.startswith(f"{phase_id}_")
                     or k.startswith(f"{phase_id}1_") or k.startswith(f"{phase_id}2_")
                     or k.startswith(f"{phase_id}3_")]
    if not phase_results:
        # Try prefix-based matching
        for prefix in [f"{phase_id}_", f"{phase_id}1_", f"{phase_id}2_", f"{phase_id}3_"]:
            phase_results.extend([(k, v) for k, v in results.items() if k.startswith(prefix)])

    # Sort by name
    phase_results = sorted(set(phase_results))

    if not phase_results:
        print("  (未完成)")
        continue

    for name, psnr in phase_results:
        if psnr is not None:
            print(f"  {name:<40} {psnr:>8.2f} dB")
        else:
            print(f"  {name:<40} {'NO PSNR':>10}")

# Print summary comparisons
print("\n" + "=" * 80)
print("关键对比")
print("=" * 80)

# Phase A comparisons
if all(k in results for k in ["A1_L2_Swin_Direct", "A2_L2_MDTA_Direct", "A3_L2_OCAB_Direct"]):
    a1 = results["A1_L2_Swin_Direct"]
    a2 = results["A2_L2_MDTA_Direct"]
    a3 = results["A3_L2_OCAB_Direct"]
    print(f"\nL2 Direct: Swin={a1:.2f}  MDTA={a2:.2f} (Δ={a2-a1:+.2f})  OCAB={a3:.2f} (Δ={a3-a1:+.2f})")
    if a2 >= a1 - 0.5 and a3 >= a1 - 0.5:
        print("  → MDTA/OCAB 对 contrast 无结构性问题, crash 来自 RandomCurric 策略")
    elif a2 < a1 - 5:
        print("  → MDTA 确实不适合 contrast, 架构结论成立")

if all(k in results for k in ["A4_L2_Swin_RandomCurric", "A5_L2_MDTA_RandomCurric", "A6_L2_OCAB_RandomCurric"]):
    a4 = results["A4_L2_Swin_RandomCurric"]
    a5 = results["A5_L2_MDTA_RandomCurric"]
    a6 = results["A6_L2_OCAB_RandomCurric"]
    print(f"\nL2 RandomCurric: Swin={a4:.2f}  MDTA={a5:.2f}  OCAB={a6:.2f}")
    if a1:
        print(f"  RandomCurric penalty vs Direct: Swin={a4-a1:+.2f}  MDTA={a5-a1:+.2f}  OCAB={a6-a1:+.2f}")

# Phase B comparison
if all(k in results for k in ["B3_S5_Swin_CurricFwd", "B4_S5_Swin_CurricRev"]):
    b3 = results["B3_S5_Swin_CurricFwd"]
    b4 = results["B4_S5_Swin_CurricRev"]
    print(f"\nS5 Swin Curric: Fwd={b3:.2f}  Rev={b4:.2f}  Δ={b3-b4:+.2f}")
    if b3 > b4:
        print("  → Swin 下 Fwd 仍胜出, 'Fwd 翻转' 不适用于 Swin")

# Phase D comparison
if all(k in results for k in ["D1_D3_TrueFt", "D2_D3_RandomCurric", "D3_D3_Direct"]):
    d1 = results["D1_D3_TrueFt"]
    d2 = results["D2_D3_RandomCurric"]
    d3 = results["D3_D3_Direct"]
    print(f"\nD3: TrueFt={d1:.2f}  RandomCurric={d2:.2f}  Direct={d3:.2f}")
    print(f"  TrueFt vs Direct: {d1-d3:+.2f}")
    print(f"  RandomCurric vs Direct: {d2-d3:+.2f}")
    if d2 > d1 + 2:
        print("  → RandomCurric 收益来自梯度干扰减少, 非预训练迁移")
