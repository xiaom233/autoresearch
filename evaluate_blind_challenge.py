#!/usr/bin/env python3
"""
Evaluate blind degradation identification accuracy.

Compares the agent's prediction (predicted_params.json) against the hidden
ground truth (.ground_truth.json) and reports identification accuracy metrics.

Usage:
    uv run evaluate_blind_challenge.py
    uv run evaluate_blind_challenge.py --predicted blind_challenge/predicted_params.json
    uv run evaluate_blind_challenge.py --dir blind_challenge
"""

import argparse
import json
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------

def category_match(pred_func: str, truth_func: str, categories: dict) -> bool:
    """Check if two function names belong to the same degradation category."""
    for cat_funcs in categories.values():
        if pred_func in cat_funcs and truth_func in cat_funcs:
            return True
    return False


def get_category(func: str, categories: dict) -> str:
    """Return the category name for a degradation function."""
    for cat_name, cat_funcs in categories.items():
        if func in cat_funcs:
            return cat_name
    return "unknown"


# Degradation categories (must match blind_challenge.py and train.py)
DEG_CATEGORIES = {
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


# ---------------------------------------------------------------------------
# Alignment — match predicted steps to ground truth steps
# ---------------------------------------------------------------------------

def align_and_score(pred_pipeline: list, truth_pipeline: list) -> dict:
    """Align predicted steps to ground truth and compute per-step scores.

    Uses greedy alignment: for each truth step, find the best-matching
    unmatched prediction step. Handles missing/extra steps.
    """
    n_truth = len(truth_pipeline)
    n_pred = len(pred_pipeline)

    steps = []
    used_pred = set()

    # Forward alignment: match each truth step to best unmatched prediction
    for i, t_step in enumerate(truth_pipeline):
        t_func = t_step["function"]
        t_cat = t_step.get("category", get_category(t_func, DEG_CATEGORIES))
        t_sev = t_step["severity"]

        best_j = None
        best_score = -1
        best_detail = {}

        for j, p_step in enumerate(pred_pipeline):
            if j in used_pred:
                continue
            p_func = p_step["function"]
            p_cat = p_step.get("category", get_category(p_func, DEG_CATEGORIES))
            p_sev = p_step["severity"]

            # Scoring
            func_exact = (p_func == t_func)
            cat_ok = (p_cat == t_cat) or category_match(p_func, t_func, DEG_CATEGORIES)
            sev_ok = (p_sev == t_sev)
            sev_off_by_1 = abs(p_sev - t_sev) == 1

            score = 0
            if func_exact:
                score += 10  # exact function match
            elif cat_ok:
                score += 3   # category match only
            if sev_ok:
                score += 5   # exact severity
            elif sev_off_by_1:
                score += 2   # close severity

            if score > best_score:
                best_score = score
                best_j = j
                best_detail = {
                    "predicted_function": p_func,
                    "predicted_severity": p_sev,
                    "predicted_category": p_cat,
                    "function_exact": func_exact,
                    "category_match": cat_ok,
                    "severity_exact": sev_ok,
                    "severity_off_by_1": sev_off_by_1,
                }

        steps.append({
            "truth_step": i + 1,
            "truth_function": t_func,
            "truth_category": t_cat,
            "truth_severity": t_sev,
            "matched_prediction_step": best_j + 1 if best_j is not None else None,
            "score": best_score if best_j is not None else "miss",
            **best_detail,
        })
        if best_j is not None:
            used_pred.add(best_j)

    # Extra predictions (not matched to any truth step)
    extras = []
    for j, p_step in enumerate(pred_pipeline):
        if j not in used_pred:
            extras.append({
                "predicted_step": j + 1,
                "predicted_function": p_step["function"],
                "predicted_severity": p_step["severity"],
            })

    # Compute aggregate metrics
    n_matched = len([s for s in steps if s["score"] != "miss"])
    func_exact_count = sum(1 for s in steps if s.get("function_exact"))
    cat_match_count = sum(1 for s in steps if s.get("category_match"))
    sev_exact_count = sum(1 for s in steps if s.get("severity_exact"))
    sev_close_count = sum(1 for s in steps if s.get("severity_off_by_1") or s.get("severity_exact"))

    # Pipeline-level accuracy (all steps exactly correct, no extras, no misses)
    pipeline_exact = (
        n_truth == n_pred and
        func_exact_count == n_truth and
        sev_exact_count == n_truth and
        len(extras) == 0
    )

    # Overall score (0–100)
    max_possible = n_truth * 15  # 10 for exact func + 5 for exact severity
    actual_score = sum(s["score"] for s in steps if isinstance(s["score"], int))
    # Penalize extra predictions
    actual_score -= len(extras) * 3
    overall_pct = max(0, round(100 * actual_score / max(1, max_possible)))

    return {
        "per_step": steps,
        "extra_predictions": extras,
        "summary": {
            "total_truth_steps": n_truth,
            "total_predicted_steps": n_pred,
            "matched_steps": n_matched,
            "missed_steps": n_truth - n_matched,
            "extra_steps": len(extras),
            "function_exact": f"{func_exact_count}/{n_truth}",
            "function_exact_pct": round(100 * func_exact_count / max(1, n_truth)),
            "category_match": f"{cat_match_count}/{n_truth}",
            "category_match_pct": round(100 * cat_match_count / max(1, n_truth)),
            "severity_exact": f"{sev_exact_count}/{n_truth}",
            "severity_exact_pct": round(100 * sev_exact_count / max(1, n_truth)),
            "severity_close": f"{sev_close_count}/{n_truth}",
            "severity_close_pct": round(100 * sev_close_count / max(1, n_truth)),
            "pipeline_exact": pipeline_exact,
            "overall_score_pct": overall_pct,
            "grade": ("A" if overall_pct >= 90 else "B" if overall_pct >= 70
                      else "C" if overall_pct >= 50 else "D" if overall_pct >= 30 else "F"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate blind degradation identification accuracy")
    parser.add_argument("--dir", type=str, default="blind_challenge",
                        help="Challenge directory containing .ground_truth.json and predicted_params.json")
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground truth JSON")
    parser.add_argument("--predicted", type=str, default=None,
                        help="Path to predicted params JSON")
    args = parser.parse_args()

    # Resolve paths
    challenge_dir = args.dir
    truth_path = args.ground_truth or os.path.join(challenge_dir, ".ground_truth.json")
    pred_path = args.predicted or os.path.join(challenge_dir, "predicted_params.json")

    # Load ground truth
    if not os.path.exists(truth_path):
        print(f"ERROR: Ground truth not found at {truth_path}")
        print("Run 'uv run blind_challenge.py' first to generate a challenge.")
        sys.exit(1)

    with open(truth_path) as f:
        truth = json.load(f)
    truth_pipeline = truth["pipeline"]

    # Load prediction
    if not os.path.exists(pred_path):
        print(f"ERROR: Prediction not found at {pred_path}")
        print("Use image-degradation-simulator skill to analyze the degraded image,")
        print(f"then save the predicted pipeline to {pred_path}")
        sys.exit(1)

    with open(pred_path) as f:
        pred = json.load(f)
    pred_pipeline = pred.get("pipeline", [])

    # Evaluate
    results = align_and_score(pred_pipeline, truth_pipeline)

    # Display
    s = results["summary"]
    print("=" * 65)
    print("  BLIND DEGRADATION IDENTIFICATION — ACCURACY REPORT")
    print("=" * 65)
    print(f"\n  Challenge ID: {truth.get('challenge_id', 'unknown')}")
    print(f"  Clean source: {truth.get('clean_source', 'unknown')}")
    print()

    # Ground truth
    print("  Ground Truth Pipeline:")
    for t_step in truth_pipeline:
        print(f"    Step {t_step['step']}: {t_step['category']}/{t_step['function']} "
              f"(severity={t_step['severity']})")
    print()

    # Per-step comparison
    print("  Step-by-Step Comparison:")
    print(f"  {'Truth':<35} {'Prediction':<35} {'Match':<10}")
    print(f"  {'-'*35} {'-'*35} {'-'*10}")
    for step in results["per_step"]:
        truth_str = f"Step{step['truth_step']}: {step['truth_function']} (sev={step['truth_severity']})"
        if step["score"] == "miss":
            pred_str = "MISSED — no match found"
            match_str = "✗"
        else:
            pred_str = f"→ {step['predicted_function']} (sev={step['predicted_severity']})"
            func_ok = "✓" if step.get("function_exact") else "~" if step.get("category_match") else "✗"
            sev_ok = "✓" if step.get("severity_exact") else "~" if step.get("severity_off_by_1") else "✗"
            match_str = f"func={func_ok} sev={sev_ok}"
        print(f"  {truth_str:<35} {pred_str:<35} {match_str:<10}")

    for extra in results["extra_predictions"]:
        print(f"  {'(extra prediction)':<35} {extra['predicted_function']} "
              f"(sev={extra['predicted_severity']}):<35 {'EXTRA':<10}")

    # Summary
    print()
    print(f"  {'─'*60}")
    print(f"  Summary:")
    print(f"    Steps:           {s['total_truth_steps']} truth → "
          f"{s['matched_steps']} matched, {s['missed_steps']} missed, "
          f"{s['extra_steps']} extra")
    print(f"    Function exact:  {s['function_exact']} ({s['function_exact_pct']}%)")
    print(f"    Category match:  {s['category_match']} ({s['category_match_pct']}%)")
    print(f"    Severity exact:  {s['severity_exact']} ({s['severity_exact_pct']}%)")
    print(f"    Severity ±1:     {s['severity_close']} ({s['severity_close_pct']}%)")
    print(f"    Pipeline exact:  {'YES' if s['pipeline_exact'] else 'NO'}")
    print(f"    Overall score:   {s['overall_score_pct']}%  Grade: {s['grade']}")
    print()

    # Interpretation
    grade = s["grade"]
    if grade == "A":
        print("  VERDICT: Excellent — blind identification nearly perfect.")
        print("  The skill correctly identified degradation types, severities, AND order.")
    elif grade in ("B",):
        print("  VERDICT: Good — key degradations identified, some details off.")
        print("  Types likely correct, severities may need refinement.")
    elif grade == "C":
        print("  VERDICT: Partial — some degradations missed or wrong category.")
        print("  Specialist model may still help but won't be optimally targeted.")
    else:
        print("  VERDICT: Poor — blind identification failed.")
        print("  The skill could not identify the degradation. Check if the")
        print("  degradation is within the supported categories (blur/noise/compression).")

    print(f"\n  {'─'*60}")
    print(f"  Ground truth: {truth_path}")
    print(f"  Prediction:    {pred_path}")

    # Save detailed report
    report_path = os.path.join(challenge_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump({**results, "ground_truth": truth, "prediction": pred}, f, indent=2)
    print(f"  Full report:   {os.path.abspath(report_path)}")
    print()


if __name__ == "__main__":
    main()
