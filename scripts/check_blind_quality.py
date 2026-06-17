#!/usr/bin/env python3
"""
Phase 5 前置守门: 检查盲识别是否由 Agent 正确执行。
用法: .venv/bin/python3 scripts/check_blind_quality.py --exp expN

判断逻辑 (分层):
  Tier 1 (确定性单步): PSNR >= 40 → 可接受
  Tier 2 (噪声): 统计匹配检查通过 → 可接受
  Tier 3 (混合): 必须有 alternatives + coupling_analysis → UNCERTAIN 可接受

不通过 → 禁止进入 Phase 5 → Agent 重做盲识别。
"""
import json, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True, help='实验目录 (如 exp17)')
args = parser.parse_args()

exp = args.exp
phase4 = os.path.join(exp, 'challenges', 'phase4')

if not os.path.isdir(phase4):
    print(f'ERROR: {phase4} 不存在')
    sys.exit(1)

challenges = sorted(d for d in os.listdir(phase4) if os.path.isdir(os.path.join(phase4, d)))
if not challenges:
    print(f'ERROR: {phase4} 下无挑战目录')
    sys.exit(1)

# Degradation classification
deterministic = {'blur_gaussian','blur_motion','blur_lens','blur_zoom','blur_glass','blur_defocus',
    'blur_jitter','compression_jpeg','compression_jpeg_2000','contrast_weaken_stretch',
    'contrast_strengthen_scale','brightness_darken_gamma_RGB','brightness_brighten_shift_HSV',
    'saturate_weaken_HSV','saturate_strengthen_YCrCb','gamma_correct_RGB','quantization_median',
    'oversharpen'}
noise_funcs = {'noise_gaussian_RGB','noise_gaussian_YCrCb','noise_poisson','noise_speckle',
    'noise_impulse','noise_spatially_correlated'}

errors = []
warnings = []

for bid in challenges:
    d = os.path.join(phase4, bid)
    pf = os.path.join(d, 'predicted_params.json')
    rf = os.path.join(d, 'reflection.json')

    if not os.path.exists(pf):
        errors.append(f'{bid}: 缺少 predicted_params.json')
        continue

    try:
        pred = json.load(open(pf))
    except:
        errors.append(f'{bid}: predicted_params.json 格式错误')
        continue

    analysis = pred.get('analysis', {})
    verdict = analysis.get('verdict', 'N/A')
    tier = analysis.get('tier', 0)
    psnr = analysis.get('psnr', 0)
    psnr_gap = analysis.get('psnr_gap', 0)
    # Normalize PSNR (handle 'inf', None, etc.)
    try:
        psnr_num = float(psnr) if psnr not in (None, 'inf', '') else (999 if psnr == 'inf' else 0)
    except (ValueError, TypeError):
        psnr_num = 0
    try:
        psnr_gap_num = float(psnr_gap) if psnr_gap not in (None, 'inf', '') else (999 if psnr_gap == 'inf' else 0)
    except (ValueError, TypeError):
        psnr_gap_num = 0
    ci = analysis.get('ci_pass_rate', '0/10')
    try:
        ci_num = int(str(ci).split('/')[0])
    except:
        ci_num = -1

    # Classify degradation
    pipeline = pred.get('pipeline', [])
    funcs = [s.get('function','') for s in pipeline]
    has_noise = any(f in noise_funcs for f in funcs)
    has_deterministic = any(f in deterministic for f in funcs)
    n_steps = len(pipeline)

    # Auto-detect tier
    if n_steps >= 2 and has_noise:
        detected_tier = 3  # mixed with noise
    elif has_noise and n_steps == 1:
        detected_tier = 2  # noise only
    else:
        detected_tier = 1  # deterministic

    # Check reflection
    ref = None
    if not os.path.exists(rf):
        errors.append(f'{bid}: 缺少 reflection.json')
    else:
        try:
            ref = json.load(open(rf))
        except:
            errors.append(f'{bid}: reflection.json 格式错误')
            continue

    # 0: Thinking process must exist (anti-enumeration / anti-script safeguard)
    tp_file = os.path.join(d, 'thinking_process.json')
    if not os.path.exists(tp_file):
        errors.append(f'{bid}: 缺少 thinking_process.json (盲识别过程不可验证)')
    else:
        try:
            tp = json.load(open(tp_file))
            psnr_tests = tp.get('total_psnr_tests', 0)
            is_enum = tp.get('enumeration_used', False)
            path = tp.get('path', '?')
            if is_enum:
                errors.append(f'{bid}: thinking_process 标记为枚举 → 禁止')
            if path == 'A' and psnr_tests > 50:
                errors.append(f'{bid}: 路径A 但 PSNR测试{psnr_tests}次 > 50 → 疑似枚举')
            elif path == 'A' and psnr_tests > 20:
                warnings.append(f'{bid}: 路径A PSNR测试{psnr_tests}次 (20-50, 需确认非枚举)')
        except:
            errors.append(f'{bid}: thinking_process.json 格式错误')

    # === CHECKS ===

    # 1: Script garbage detection — CI=0 only suspicious if no PSNR, no stats, no reflection
    if ci_num <= 0 and verdict not in ('POOR', 'UNCERTAIN'):
        has_psnr = psnr_num > 0
        has_stats = analysis.get('statistical_match') is not None
        has_ref = ref is not None
        has_ranking = ref and ('psnr_ranking' in ref or 'iterations' in ref)
        is_tier2_noise = detected_tier == 2
        if not ((has_psnr or has_stats or is_tier2_noise) and has_ref):
            errors.append(f'{bid}: CI=0 + no evidence + no reflection → 脚本生成')
        elif not has_ranking:
            warnings.append(f'{bid}: Agent 生成但缺少 psnr_ranking')

    # 2: Tier 3 must have alternatives (in pred or reflection)
    if detected_tier == 3 or (detected_tier == 2 and n_steps > 1):
        alts = pred.get('alternatives', [])
        if not alts and ref:
            alts = ref.get('alternatives', ref.get('psnr_ranking', []))
            # Filter to just alt pipelines
            if isinstance(alts, list) and len(alts) > 1:
                alts = [a for a in alts[1:3] if isinstance(a, dict)]  # top 2 alternatives
        if not alts:
            errors.append(f'{bid}: Tier{detected_tier} 混合退化 缺少 alternatives')
        if ref and 'coupling_analysis' not in ref and detected_tier == 3:
            warnings.append(f'{bid}: Tier3 缺少 coupling_analysis')

    # 3: PSNR-based checks (Tier 1 only)
    if detected_tier == 1 and psnr_num > 0:
        if psnr_num < 30 and psnr_num != 999 and verdict not in ('POOR', 'UNCERTAIN'):
            errors.append(f'{bid}: Tier1 PSNR={psnr_num:.0f} < 30 但 verdict={verdict}')
        if psnr_gap_num < 10 and verdict in ('GOOD', 'LIKELY'):
            warnings.append(f'{bid}: PSNR gap={psnr_gap_num:.0f} < 10 但 verdict={verdict}')

    # 4: Must have psnr_ranking in reflection for multi-hypothesis testing
    if ref and 'psnr_ranking' not in ref and 'iterations' not in ref:
        errors.append(f'{bid}: 缺少 psnr_ranking 或 iterations (无多假设测试)')
    if ref and 'psnr_ranking' not in ref:
        # Check iterations for multi-hypothesis records
        has_multi = False
        for it in ref.get('iterations', []):
            ci_r = it.get('ci_result', '')
            if ',' in str(ci_r) or ':' in str(ci_r):
                has_multi = True
        if not has_multi:
            warnings.append(f'{bid}: 未找到多假设 PSNR/CI 对比记录')

    # 5: Noise checks (Tier 2)
    if detected_tier == 2 and ref:
        if 'statistical_checks' not in ref and 'noise' not in str(ref.get('initial_analysis', '')).lower():
            warnings.append(f'{bid}: Tier2 噪声退化 缺少 statistical_checks')

    # 6: Tier 3 GOOD verdict = suspicious
    if detected_tier == 3 and verdict == 'GOOD':
        errors.append(f'{bid}: Tier3 混合退化 不应标 GOOD (退化耦合无法可靠验证)')

# Report
print(f'检查 {len(challenges)} 挑战:')
print(f'  Tier分布: 1={sum(1 for b in challenges if not any(s.get("function","") in noise_funcs for s in json.load(open(os.path.join(phase4,b,"predicted_params.json")))["pipeline"]) and len(json.load(open(os.path.join(phase4,b,"predicted_params.json")))["pipeline"])==1)} | 2={sum(1 for b in challenges if any(s.get("function","") in noise_funcs for s in json.load(open(os.path.join(phase4,b,"predicted_params.json")))["pipeline"]) and len(json.load(open(os.path.join(phase4,b,"predicted_params.json")))["pipeline"])==1)} | 3={sum(1 for b in challenges if len(json.load(open(os.path.join(phase4,b,"predicted_params.json")))["pipeline"])>=2)}')
print(f'  错误: {len(errors)}')
print(f'  警告: {len(warnings)}')
print()

if errors:
    print('❌ 守门失败 — 禁止进入 Phase 5:')
    for e in errors:
        print(f'  - {e}')
    print()
    print('修复方法: Skill(skill="image-degradation-simulator", args="分析 <target> --clean <clean>")')
    sys.exit(1)

if warnings:
    print('⚠️ 守门通过 (有警告):')
    for w in warnings:
        print(f'  - {w}')
    print('  警告项可进入 Phase 5, 但训练后需 REFLECTION_MECHANISM 验证')
    print()

print('✅ 守门通过 — 可以进入 Phase 5')
