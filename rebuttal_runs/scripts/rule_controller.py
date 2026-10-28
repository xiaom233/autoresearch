#!/usr/bin/env python3
"""Deterministic rule controller for E1 (no LLM).

plan    : analyze 5 support pairs, aggregate signals, build R0 pipeline,
          pre-validate deterministic/noise candidates, write R0 tasks.
decide  : after a round trains, decide whether the next deterministic
          correction should be trained, and write the next task file.
"""
import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from x_distortion import add_distortion  # noqa: E402

CH = ROOT / "exp41" / "challenges" / "phase4"
RFA = ROOT / ".claude/skills/image-degradation-simulator/scripts/run_full_analysis.py"
VERIFY = ROOT / ".claude/skills/image-degradation-simulator/scripts/verify_signals.py"
OUT = ROOT / "rebuttal_runs" / "E1"
RULES = json.load(open(OUT / "rules_frozen.json"))
TH = RULES["thresholds"]
REF = RULES["reflection"]
PY = sys.executable

# Longest/most specific keys first: "GAUSSIAN" is a substring of
# "GAUSSIAN_YCRCB", so the RGB key must not be checked before YCrCb.
NOISE_FUNCS = {
    "GAUSSIAN_YCRC": "noise_gaussian_YCrCb",
    "NOISE_GAUSSIAN_YCRC": "noise_gaussian_YCrCb",
    "YCRC": "noise_gaussian_YCrCb",
    "GAUSSIAN_RGB": "noise_gaussian_RGB",
    "NOISE_GAUSSIAN_RGB": "noise_gaussian_RGB",
    "GAUSSIAN": "noise_gaussian_RGB",
    "NOISE_SPECKLE": "noise_speckle",
    "SPECKLE": "noise_speckle",
    "NOISE_POISSON": "noise_poisson",
    "POISSON": "noise_poisson",
    "NOISE_IMPULSE": "noise_impulse",
    "IMPULSE": "noise_impulse",
}
CAT_RANK = {"compression": 0, "noise": 1, "blur": 2, "contrast": 3, "brightness": 4, "saturation": 5}
PHYS_ORDER = {"noise": 0, "brightness": 0, "blur": 1, "contrast": 2, "saturation": 2, "compression": 3}


def parse_rfa(stdout):
    idx = stdout.find('{\n  "step1_compression"')
    if idx < 0:
        idx = stdout.rfind("\n{")
        if idx < 0:
            raise ValueError("no JSON in run_full_analysis output")
        idx += 1
    return json.loads(stdout[idx:])


def map_high(v, cuts):
    return 1 + sum(v > c for c in cuts)


def map_low(v, cuts):
    return 1 + sum(v < c for c in cuts)


def map_dev(v, cuts):
    d = max(v, 1.0 / v) if v > 0 else 1.0
    return 1 + sum(d > c for c in cuts)


def dc_from_one(d):
    steps = []
    c = d.get("step1_compression", {})
    s = c.get("signals", {})
    bb = s.get("block_boundary", 0.0)
    if bb > TH["compression"]["block_boundary_min"]:
        steps.append({"cat": "compression", "func": "compression_jpeg",
                      "sev": map_high(bb, [1.05, 1.15, 1.3, 1.5]),
                      "evidence": f"block_boundary={bb:.3f}"})

    n = d.get("step2_noise", {})
    ns = n.get("signals", {})
    sigma = ns.get("noise_prior_sigma", 0.0)
    nverdict = (n.get("verdict") or "").upper()
    impulse = "IMPULSE" in nverdict
    if impulse or sigma >= TH["noise"]["sigma_min"]:
        func = None
        for k, v in NOISE_FUNCS.items():
            if k in nverdict:
                func = v
                break
        if func is None:
            func = "noise_gaussian_RGB"
        sev = n.get("estimated_severity")
        if not isinstance(sev, int):
            sev = 3 if sigma < 8 else (4 if sigma < 16 else 5)
        if impulse and not n.get("estimated_severity"):
            sev = 5
        steps.append({"cat": "noise", "func": func, "sev": max(1, min(5, int(sev))),
                      "evidence": f"sigma={sigma:.1f}, verdict={nverdict}"})

    b = d.get("step3_blur", {})
    bs = b.get("signals", {})
    gm = bs.get("gm_ratio", 99.0)
    aniso = bs.get("FFT_anisotropy", 0.0)
    if gm < TH["blur"]["gm_ratio_max"]:
        func = "blur_motion" if aniso > TH["blur"]["anisotropy_motion"] else "blur_gaussian"
        steps.append({"cat": "blur", "func": func,
                      "sev": map_low(gm, [0.1, 0.2, 0.35, 0.5]),
                      "evidence": f"gm_ratio={gm:.3f}, anisotropy={aniso:.2f}"})

    g = d.get("step4_global", {})
    gs = g.get("signals", {})
    gverdict = (g.get("verdict") or "").upper()
    if not gverdict.startswith("NO_GLOBAL") and not impulse:
        vr = gs.get("variance_ratio", 1.0)
        ms = gs.get("brightness_mean_shift", 0.0)
        sat = gs.get("saturation_chroma_ratio", 1.0)
        if (gs.get("contrast_detected_raw") or gs.get("contrast_from_variance")) and \
                (vr < TH["global"]["variance_ratio_low"] or vr > TH["global"]["variance_ratio_high"]):
            func = "contrast_strengthen_scale" if vr > 1 else "contrast_weaken_scale"
            steps.append({"cat": "contrast", "func": func,
                          "sev": map_dev(vr, [1.15, 1.3, 1.6, 2.0]),
                          "evidence": f"variance_ratio={vr:.3f}"})
        if gs.get("brightness_detected_raw") and abs(ms) > TH["global"]["mean_shift_min"]:
            func = "brightness_brighten_shfit_RGB" if ms > 0 else "brightness_darken_shfit_RGB"
            if "GAMMA" in gverdict:
                func = func.replace("shfit_RGB", "gamma_RGB")
            steps.append({"cat": "brightness", "func": func,
                          "sev": map_dev(abs(ms) / 15.0, [1.0, 1.7, 2.7, 4.0]),
                          "evidence": f"mean_shift={ms:.1f}"})
        if gs.get("saturation_detected_raw") and \
                (sat < TH["global"]["saturation_ratio_low"] or sat > TH["global"]["saturation_ratio_high"]):
            func = "saturate_strengthen_HSV" if sat > 1 else "saturate_weaken_HSV"
            steps.append({"cat": "saturation", "func": func,
                          "sev": map_dev(sat, [1.1, 1.25, 1.5, 2.0]),
                          "evidence": f"chroma_ratio={sat:.3f}"})
    return steps


def _cpu_env():
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    return env


def analyze_one(task):
    sid, i, timeout = task
    cache = OUT / "cache" / f"{sid}_{i}.json"
    if cache.exists():
        d = json.load(open(cache))
        return {"sid": sid, "pair": i, "analysis": d, "steps": dc_from_one(d), "cached": True}
    target = CH / sid / f"degraded_{i}.png"
    clean = CH / sid / f"clean_{i}.png"
    try:
        r = subprocess.run([PY, str(RFA), "--target", str(target), "--clean", str(clean)],
                           capture_output=True, text=True, timeout=timeout, env=_cpu_env())
    except subprocess.TimeoutExpired:
        return {"sid": sid, "pair": i, "error": f"RFA timeout after {timeout}s"}
    except Exception as e:
        return {"sid": sid, "pair": i, "error": f"RFA failed: {e}"}
    if r.returncode != 0:
        return {"sid": sid, "pair": i, "error": (r.stderr or r.stdout)[-1000:]}
    try:
        d = parse_rfa(r.stdout)
    except Exception as e:
        return {"sid": sid, "pair": i, "error": f"parse: {e}"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return {"sid": sid, "pair": i, "analysis": d, "steps": dc_from_one(d)}


def aggregate(pair_steps):
    cats = ["compression", "noise", "blur", "contrast", "brightness", "saturation"]
    agg = {}
    for cat in cats:
        votes = [(st["func"], st["sev"]) for steps in pair_steps for st in steps if st["cat"] == cat]
        if len(votes) < 3:
            continue
        func, count = Counter(f for f, _ in votes).most_common(1)[0]
        if count < 3:
            continue
        sev = int(round(statistics.median([s for f, s in votes if f == func])))
        agg[cat] = {"func": func, "sev": max(1, min(5, sev)), "votes": count, "n": len(pair_steps)}
    keep = sorted(agg, key=lambda c: CAT_RANK[c])[: RULES["support"]["max_pipeline_steps"]]
    keep = sorted(keep, key=lambda c: (PHYS_ORDER[c], CAT_RANK[c]))
    return [(agg[c]["func"], agg[c]["sev"], c) for c in keep], agg


def rs_for(pipeline, agg, analysis_by_pair):
    has_motion = any(f == "blur_motion" for f, _, _ in pipeline)
    aniso = 0.0
    for a in analysis_by_pair:
        try:
            aniso = max(aniso, a["step3_blur"]["signals"].get("FFT_anisotropy", 0.0))
        except Exception:
            pass
    has_contrast = any(c == "contrast" for _, _, c in pipeline)
    has_struct = any(c in ("blur", "compression") for _, _, c in pipeline)
    vr = 1.0
    for a in analysis_by_pair:
        try:
            vr = a["step4_global"]["signals"].get("variance_ratio", 1.0)
        except Exception:
            pass
    if has_motion and aniso > TH["blur"]["anisotropy_motion"]:
        return "motion_ocab", {"attention_type": "ocab", "embed_dim": 30, "window_size": 16,
                               "use_simple_gate": 1}
    if has_contrast and has_struct and (vr < 0.85 or vr > 1.15):
        return "contrast_structured_gcm", {"attention_type": "swin", "embed_dim": 64,
                                           "window_size": 8, "use_gcm": 1, "use_simple_gate": 1}
    return "default_swin_sg", {"attention_type": "swin", "embed_dim": 64,
                               "window_size": 8, "use_simple_gate": 1}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def has_noise(pipeline):
    return any(f.startswith("noise_") for f, _ in pipeline)


def apply_pipeline(clean, pipeline, seed):
    random.seed(seed)
    np.random.seed(seed)
    img = clean.copy()
    for func, sev in pipeline:
        img = add_distortion(img, severity=sev, distortion_name=func)
    return img


def load_pairs(sid):
    out = []
    for i in range(5):
        clean = np.array(Image.open(CH / sid / f"clean_{i}.png").convert("RGB"), dtype=np.uint8)
        deg = np.array(Image.open(CH / sid / f"degraded_{i}.png").convert("RGB"), dtype=np.uint8)
        out.append((CH / sid / f"clean_{i}.png", CH / sid / f"degraded_{i}.png", clean, deg))
    return out


def score_pipeline(sid, pipeline, pairs, simdir):
    if has_noise(pipeline):
        vals = []
        ps = []
        for i, (clean_path, deg_path, clean, deg) in enumerate(pairs):
            sim = apply_pipeline(clean, pipeline, 1000 + i)
            sim_path = simdir / f"{sid}_{i}.png"
            Image.fromarray(sim).save(sim_path)
            pipe_str = ",".join(f"{f}:{s}" for f, s in pipeline)
            r = subprocess.run([PY, str(VERIFY), "--target", str(deg_path), "--clean",
                                str(clean_path), "--simulated", str(sim_path),
                                "--pipeline", pipe_str], capture_output=True, text=True,
                               timeout=120, env=_cpu_env())
            # The script prints a legend containing MATCH/PARTIAL/MISMATCH;
            # only the first "Verdict:" line is the actual result.
            m = re.search(r"Verdict:\s*(MATCH|PARTIAL|WEAK|MISMATCH)", r.stdout, re.I)
            v = {"MATCH": 1.0, "PARTIAL": 0.66, "WEAK": 0.33, "MISMATCH": 0.0}.get(
                m.group(1).upper() if m else "", 0.0)
            vals.append(v)
            ps.append(psnr(sim, deg))
        return {"mode": "signal", "score": float(np.mean(vals)), "psnr": float(np.mean(ps))}
    ps = []
    for i, (_, _, clean, deg) in enumerate(pairs):
        sim = apply_pipeline(clean, pipeline, 1000 + i)
        ps.append(psnr(sim, deg))
    return {"mode": "psnr", "score": float(np.mean(ps)), "psnr": float(np.mean(ps))}


def threshold_good(score):
    if score["mode"] == "psnr":
        return score["score"] >= REF["deterministic_score_good_db"]
    return score["score"] >= REF["noise_signal_match_score"]


def near_miss_candidates(analysis, pipeline):
    """Add a missed category when its signal is just below the hard threshold."""
    cats = {c for _, _, c in pipeline}
    out = []
    if len(pipeline) >= RULES["support"]["max_pipeline_steps"]:
        return out
    bbs = [a["step1_compression"]["signals"].get("block_boundary", 0.0) for a in analysis]
    gms = [a["step3_blur"]["signals"].get("gm_ratio", 99.0) for a in analysis]
    sigmas = [a["step2_noise"]["signals"].get("noise_prior_sigma", 0.0) for a in analysis]
    nvs = [(a["step2_noise"].get("verdict") or "").upper() for a in analysis]
    vrs = [a["step4_global"]["signals"].get("variance_ratio", 1.0) for a in analysis]
    mss = [a["step4_global"]["signals"].get("brightness_mean_shift", 0.0) for a in analysis]
    sats = [a["step4_global"]["signals"].get("saturation_chroma_ratio", 1.0) for a in analysis]
    base = [(f, s) for f, s, _ in pipeline]

    def add(func, cat):
        if cat not in cats:
            out.append(base + [(func, 1)])

    if "compression" not in cats and statistics.median(bbs) > 1.0:
        add("compression_jpeg", "compression")
    if "noise" not in cats:
        if any("IMPULSE" in v for v in nvs):
            add("noise_impulse", "noise")
        elif statistics.median(sigmas) >= 1.0:
            add("noise_gaussian_RGB", "noise")
    if "blur" not in cats and statistics.median(gms) < 1.0:
        add("blur_gaussian", "blur")
    if "contrast" not in cats:
        med = statistics.median(vrs)
        if med < 0.9:
            add("contrast_weaken_scale", "contrast")
        elif med > 1.1:
            add("contrast_strengthen_scale", "contrast")
    if "brightness" not in cats and abs(statistics.median(mss)) > 10.0:
        add("brightness_brighten_shfit_RGB" if statistics.median(mss) > 0
            else "brightness_darken_shfit_RGB", "brightness")
    if "saturation" not in cats:
        med = statistics.median(sats)
        if med < 0.9:
            add("saturate_weaken_HSV", "saturation")
        elif med > 1.1:
            add("saturate_strengthen_HSV", "saturation")
    return out


def candidates_for(pipeline, agg, sid, analysis=None):
    out = []
    if analysis:
        out.extend(near_miss_candidates(analysis, pipeline))
    if not agg:
        return out[: REF["max_candidate_tests_per_round"]]
    lowest = min(agg, key=lambda c: (agg[c]["votes"], -CAT_RANK[c]))
    base = [(f, s) for f, s, _ in pipeline]

    def add(cand_pipe):
        cand = [(f, int(max(1, min(5, s)))) for f, s in cand_pipe][: RULES["support"]["max_pipeline_steps"]]
        if cand and cand not in out and cand != base:
            out.append(cand)

    info = agg[lowest]
    func, sev = info["func"], info["sev"]
    # remove-step candidate
    add([(f, s) for f, s, c in pipeline if c != lowest])
    # severity sweep
    for d in (1, -1, 2, -2):
        add([(f, s + d if c == lowest else s) for f, s, c in pipeline])
    # subtype alternatives
    if lowest == "blur":
        for f in ("blur_gaussian", "blur_glass", "blur_lens", "blur_motion"):
            if f != func:
                add([(f, s) if c == lowest else (ff, s) for ff, s, c in pipeline])
    elif lowest == "noise":
        for f in ("noise_gaussian_RGB", "noise_gaussian_YCrCb", "noise_speckle",
                  "noise_poisson", "noise_impulse"):
            if f != func:
                add([(f, s) if c == lowest else (ff, s) for ff, s, c in pipeline])
    elif lowest == "contrast":
        for f in ("contrast_weaken_scale", "contrast_strengthen_scale",
                  "contrast_weaken_stretch", "contrast_strengthen_stretch"):
            if f != func:
                add([(f, s) if c == lowest else (ff, s) for ff, s, c in pipeline])
    elif lowest == "brightness":
        for f in ("brightness_darken_shfit_RGB", "brightness_brighten_shfit_RGB",
                  "brightness_darken_gamma_RGB", "brightness_brighten_gamma_RGB"):
            if f != func:
                add([(f, s) if c == lowest else (ff, s) for ff, s, c in pipeline])
    elif lowest == "saturation":
        for f in ("saturate_weaken_HSV", "saturate_strengthen_HSV",
                  "saturate_weaken_YCrCb", "saturate_strengthen_YCrCb"):
            if f != func:
                add([(f, s) if c == lowest else (ff, s) for ff, s, c in pipeline])
    return out[: REF["max_candidate_tests_per_round"]]


def write_params(path, pipeline):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"pipeline": [{"function": f, "severity": int(s)} for f, s in pipeline]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def arch_flags(arch):
    flags = {
        "AR_ATTENTION_TYPE": arch.get("attention_type", "swin"),
        "AR_EMBED_DIM": str(arch.get("embed_dim", 64)),
        "AR_WINDOW_SIZE": str(arch.get("window_size", 8)),
        "AR_USE_SIMPLE_GATE": str(arch.get("use_simple_gate", 0)),
        "AR_USE_GCM": str(arch.get("use_gcm", 0)),
    }
    return " ".join(f"{k}={v}" for k, v in flags.items())


def make_task(sid, rnd, arch):
    params = OUT / "params" / f"{sid}_{rnd}.json"
    ckpt = OUT / "experiments" / f"{sid}_{rnd}"
    log = OUT / "logs" / f"{sid}_{rnd}.log"
    env = " ".join([
        f"AR_PARAMS_PATH={params.relative_to(ROOT)}",
        f"AR_VAL_PARAMS_PATH={params.relative_to(ROOT)}",
        f"AR_CHALLENGE_DIR={('exp41/challenges/phase4/' + sid)}",
        "AR_EPOCH_BUDGET=2", "AR_LEARNING_RATE=0.0005", "AR_BATCH_SIZE=16",
        "AR_LOSS_FN=l1", "AR_SEED=42", "AR_SKIP_FINAL_VAL=1",
        arch_flags(arch),
        f"AR_CKPT_PREFIX={ckpt.relative_to(ROOT)}",
    ])
    cmd = f"CUDA_VISIBLE_DEVICES=GPU_ID {env} .venv/bin/python3 train.py > {log.relative_to(ROOT)} 2>&1"
    return {"id": f"E1_{sid}_{rnd}", "cmd": cmd,
            "check": str((ckpt / f"{sid}_{rnd}_results.json").relative_to(ROOT))}


def write_tasks(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def plan(args):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decisions").mkdir(parents=True, exist_ok=True)
    (OUT / "params").mkdir(parents=True, exist_ok=True)
    scenarios = ([s.strip() for s in args.scenarios.split(",")] if args.scenarios
                 else [f"blind_{i:04d}" for i in range(1, 25)])
    tasks = [(sid, i, args.timeout) for sid in scenarios for i in range(5)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(analyze_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"sid": t[0], "pair": t[1], "error": f"worker crashed: {e}"}
            results.setdefault(r["sid"], []).append(r)
            print(f"[rfa] {r['sid']} pair{r['pair']} "
                  f"{'ERROR' if 'error' in r else 'ok'}", flush=True)

    tasks_r0 = []
    for sid in scenarios:
        rs = sorted(results.get(sid, []), key=lambda x: x["pair"])
        ok = [r for r in rs if "analysis" in r]
        if len(ok) < 3:
            dec = {"scenario_id": sid, "status": "ANALYSIS_FAILED",
                   "errors": [r.get("error") for r in rs]}
            (OUT / "decisions" / f"{sid}.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
            continue
        agg_pipe, agg = aggregate([r["steps"] for r in ok])
        analysis = [r["analysis"] for r in ok]
        fallback_note = None
        if not agg_pipe:
            # Pre-registered conservative fallback when no category reaches 3/5.
            med_sigma = statistics.median([
                a["step2_noise"]["signals"].get("noise_prior_sigma", 0.0) for a in analysis])
            med_gm = statistics.median([
                a["step3_blur"]["signals"].get("gm_ratio", 99.0) for a in analysis])
            med_bb = statistics.median([
                a["step1_compression"]["signals"].get("block_boundary", 0.0) for a in analysis])
            if med_sigma >= 1.0:
                agg_pipe = [("noise_gaussian_RGB", 1, "noise")]
            elif med_gm < 1.0:
                agg_pipe = [("blur_gaussian", 1, "blur")]
            elif med_bb > 1.0:
                agg_pipe = [("compression_jpeg", 1, "compression")]
            else:
                agg_pipe = [("noise_gaussian_RGB", 1, "noise")]
            c = agg_pipe[0][2]
            agg = {c: {"func": agg_pipe[0][0], "sev": 1, "votes": 5, "n": 5}}
            fallback_note = (f"no category reached 3/5; fallback from medians "
                             f"sigma={med_sigma:.2f}, gm={med_gm:.2f}, bb={med_bb:.3f}")
        pipeline0 = [(f, s) for f, s, _ in agg_pipe]
        rule_id, arch = rs_for(agg_pipe, agg, analysis)
        pairs = load_pairs(sid)
        simdir = Path(tempfile.mkdtemp(prefix=f"e1_{sid}_"))
        score0 = score_pipeline(sid, pipeline0, pairs, simdir)
        good0 = threshold_good(score0)
        cands = candidates_for(agg_pipe, agg, sid, analysis)
        scored = []
        for c in cands:
            sc = score_pipeline(sid, c, pairs, simdir)
            scored.append({"pipeline": c, "score": sc})
        scored.sort(key=lambda x: x["score"]["score"], reverse=True)
        margin = (REF["min_deterministic_improvement_db"] if score0["mode"] == "psnr"
                  else REF["min_noise_signal_improvement"])
        chosen = None
        for s in scored:
            if s["score"]["score"] - score0["score"] > margin:
                chosen = s
                break
        dec = {
            "scenario_id": sid,
            "status": "ok",
            "fallback_note": fallback_note,
            "aggregated": agg,
            "pipeline_R0": pipeline0,
            "score_R0": score0,
            "low_confidence_R0": (not good0),
            "rs_rule": rule_id,
            "arch": arch,
            "candidates_tested": len(scored),
            "candidate_scores": scored,
            "pipeline_R1": chosen["pipeline"] if chosen else None,
            "score_R1_expected": chosen["score"] if chosen else None,
            "r1_improved_expected": bool(chosen),
        }
        # second-choice candidate for R2 (distinct from R1)
        alt = None
        for s in scored:
            if chosen is None or s["pipeline"] != chosen["pipeline"]:
                if s["score"]["score"] - score0["score"] > margin:
                    alt = s
                    break
        dec["pipeline_R2"] = alt["pipeline"] if alt else None
        dec["score_R2_expected"] = alt["score"] if alt else None
        (OUT / "decisions" / f"{sid}.json").write_text(
            json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
        write_params(OUT / "params" / f"{sid}_R0.json", pipeline0)
        if dec["pipeline_R1"]:
            write_params(OUT / "params" / f"{sid}_R1.json", dec["pipeline_R1"])
        if dec["pipeline_R2"]:
            write_params(OUT / "params" / f"{sid}_R2.json", dec["pipeline_R2"])
        tasks_r0.append(make_task(sid, "R0", arch))
        print(f"[plan] {sid} R0={pipeline0} score={score0['score']:.3f} "
              f"low={dec['low_confidence_R0']} R1={dec['pipeline_R1']}", flush=True)
    write_tasks(OUT / "tasks_R0.jsonl", tasks_r0)
    print(f"[plan] wrote {len(tasks_r0)} R0 tasks")


def parse_log(log):
    if not log.exists():
        return {"exists": False}
    txt = log.read_text(errors="ignore")
    out = {"exists": True, "nan": "PSNR_RGB=nan" in txt or "loss: nan" in txt,
           "best_support_psnr": None, "best_step": None, "training_seconds": None}
    import re
    m = re.search(r"Loaded best checkpoint: step (\d+) \(Set14 PSNR_RGB=([0-9.]+)\)", txt)
    if m:
        out["best_step"] = int(m.group(1))
        out["best_support_psnr"] = float(m.group(2))
    m = re.search(r"training_seconds:\s+([0-9.]+)", txt)
    if m:
        out["training_seconds"] = float(m.group(1))
    return out


def decide(args):
    rnd = args.round
    next_round = {"R0": "R1", "R1": "R2"}.get(rnd)
    if next_round is None:
        print("no next round for", rnd)
        return
    tasks = []
    for sid in [f"blind_{i:04d}" for i in range(1, 25)]:
        dec = json.load(open(OUT / "decisions" / f"{sid}.json"))
        if dec.get("status") != "ok":
            continue
        log = parse_log(OUT / "logs" / f"{sid}_{rnd}.log")
        dec[f"train_{rnd}"] = log
        cur_arch = dec.get(f"arch_{rnd}", dec["arch"])
        next_arch = dict(cur_arch)
        if log.get("nan") and (next_arch.get("use_gcm") or next_arch.get("use_freqmod")):
            next_arch = {"attention_type": "swin", "embed_dim": 64,
                         "window_size": 8, "use_simple_gate": 1}
            dec[f"arch_fallback_{next_round}"] = (
                f"{rnd} NaN with {cur_arch}; fallback to Swin+SimpleGate")
        dec[f"arch_{next_round}"] = next_arch

        if rnd == "R0":
            trigger = (dec["low_confidence_R0"] or log.get("nan", False)
                       or (log.get("best_support_psnr") is not None
                           and log["best_support_psnr"] < REF["train_trigger_support_psnr_below"]))
            nxt = dec.get("pipeline_R1")
        else:
            prev = dec.get("train_R0", {})
            improved = ((log.get("best_support_psnr") or 0) -
                        (prev.get("best_support_psnr") or 0)) > REF["continue_if_support_improvement_db"]
            trigger = improved and dec.get("low_confidence_R1", True) and not log.get("nan", False)
            nxt = dec.get("pipeline_R2")
        dec[f"trigger_{next_round}"] = bool(trigger and nxt)
        (OUT / "decisions" / f"{sid}.json").write_text(
            json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
        if trigger and nxt:
            tasks.append(make_task(sid, next_round, next_arch))
    write_tasks(OUT / f"tasks_{next_round}.jsonl", tasks)
    print(f"[decide] {next_round}: {len(tasks)} tasks")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--scenarios", default=None,
                   help="comma-separated subset, e.g. blind_0001,blind_0003")
    p.set_defaults(fn=plan)
    d = sub.add_parser("decide")
    d.add_argument("--round", choices=["R0", "R1"], required=True)
    d.set_defaults(fn=decide)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
