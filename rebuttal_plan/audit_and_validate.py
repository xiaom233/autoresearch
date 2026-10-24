"""只用标准库；审计公开材料/验证执行清单，不调用训练或隐藏GT目录。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys


def load_csv(path):
    text = path.read_text(encoding='utf-8-sig')
    return list(csv.DictReader(io.StringIO('\n'.join(
        line for line in text.splitlines()
        if line.strip() and not line.lstrip('"').startswith('#')))))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root):
    report = {'scope': '公开汇总材料；未读取隐藏GT目录；不是完整训练复现',
              'sources': {}, 'dc_recomputed': {}, 'warnings': []}
    for name in ['blind_id_accuracy.csv', 'blind_id_detail.csv', 'ablation.csv',
                 'all_methods_comparison.csv', 'exp/exp40/final_performance.md']:
        p = root / name
        report['sources'][name] = {'exists': p.is_file(),
                                 'sha256': digest(p) if p.is_file() else None}
    p = root / 'blind_id_detail.csv'
    if p.is_file():
        rows = load_csv(p)
        for col in ['Random_Type', 'Random_Spec', 'Qwen_Type', 'Qwen_Spec',
                    'R0_Type', 'R0_Spec', 'R1_Type', 'R1_Spec', 'Ours_Type', 'Ours_Spec']:
            values = []
            for row in rows:
                try:
                    n, d = map(int, row[col].split('/'))
                    if d <= 0 or not 0 <= n <= d:
                        raise ValueError('invalid fraction')
                    values.append((n, d))
                except (KeyError, ValueError, AttributeError):
                    report['warnings'].append(f'{col}: invalid row {row.get("Challenge")}')
            if values:
                n, d = sum(v[0] for v in values), sum(v[1] for v in values)
                report['dc_recomputed'][col] = {
                    'n': n, 'd': d, 'challenges': len(values),
                    'micro_pct': round(100 * n / d, 4),
                    'macro_pct': round(100 * statistics.mean(a / b for a, b in values), 4)}
        report['warnings'].append('Ours择优依据未核实；复算分数不证明择优协议有效。')
    p = root / 'exp/exp40/final_performance.md'
    if p.is_file() and 'exp42' in p.read_text(encoding='utf-8-sig').splitlines()[0]:
        report['warnings'].append('exp40目录下报告标题为exp42，须分开追溯。')
    p = root / 'ablation.csv'
    if p.is_file():
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            cells = next(csv.reader([line]))
            if cells and cells[0] == 'exp40-Best' and len(cells) >= 7:
                if cells[2] == cells[3] or cells[5] == cells[6]:
                    report['warnings'].append('ablation exp40-Best中SSIM/LPIPS列相等，检查列映射与原始指标。')
    return report


def validate(data):
    errors = []
    if not Path(data.get('code_root', '')).is_dir():
        errors.append('code_root不存在')
    if data.get('code_hash') in (None, '', 'UNKNOWN'):
        errors.append('尚未记录实际代码版本/哈希')
    for gate in ['history_audited', 'metrics_unified', 'no_test_feedback',
                 'controller_budgets_matched', 'safe_launcher_ready', 'five_pair_loader_verified']:
        if data.get('gates', {}).get(gate) is not True:
            errors.append(f'门禁未通过: {gate}')
    runs = data.get('runs', [])
    if not runs:
        errors.append('runs为空')
    seen = set()
    hash_cache = {}
    total_candidates = 0
    for run in runs:
        rid = run.get('run_id', '')
        if not rid or rid in seen:
            errors.append(f'run_id空或重复: {rid}')
        seen.add(rid)
        split_ids, split_hashes = {}, {}
        for split in ['support', 'selection', 'test']:
            pairs = run.get(split + '_pairs', [])
            ids, hashes = [], []
            if split == 'support' and len(pairs) != 5:
                errors.append(f'{rid}: support必须恰好5对')
            if split == 'test' and not pairs:
                errors.append(f'{rid}: test清单为空')
            if split == 'selection' and run.get('selection_split') == 'selection' and not pairs:
                errors.append(f'{rid}: 声明独立selection但清单为空')
            for pair in pairs:
                ids.append(pair.get('image_id'))
                for kind in ['clean', 'lq']:
                    p = Path(pair.get(kind, ''))
                    if not p.is_file():
                        errors.append(f'{rid}/{split}: {kind}文件不存在: {p}')
                    elif kind == 'clean':
                        key = str(p.resolve())
                        if key not in hash_cache:
                            hash_cache[key] = digest(p)
                        hashes.append(hash_cache[key])
            if None in ids or len(ids) != len(set(ids)):
                errors.append(f'{rid}/{split}: image_id缺失或重复')
            if len(hashes) != len(set(hashes)):
                errors.append(f'{rid}/{split}: clean内容重复')
            split_ids[split], split_hashes[split] = set(ids), set(hashes)
        for a, b in [('support', 'test'), ('selection', 'test'), ('support', 'selection')]:
            if split_ids[a] & split_ids[b] or split_hashes[a] & split_hashes[b]:
                errors.append(f'{rid}: {a}/{b}重叠')
        if run.get('selection_split') not in ('support', 'selection'):
            errors.append(f'{rid}: 禁止test参与选择')
        training = run.get('training', {})
        for field in ['batch', 'crop', 'lr', 'epoch_budget', 'steps']:
            if training.get(field) is None:
                errors.append(f'{rid}: 尚未确认training.{field}')
        candidates = run.get('max_candidate_trainings')
        if not isinstance(candidates, int) or candidates <= 0:
            errors.append(f'{rid}: 无效候选训练预算')
        else:
            total_candidates += candidates
            if candidates != data.get('budget', {}).get('candidate_trainings_per_run'):
                errors.append(f'{rid}: 候选预算与全局预算不一致')
        if not Path(run.get('params_path', '')).is_file():
            errors.append(f'{rid}: params_path不存在')
        output = run.get('output_dir', '')
        if not output or 'REPLACE' in output:
            errors.append(f'{rid}: 未指定新output_dir')
        elif Path(output).exists():
            errors.append(f'{rid}: output_dir已存在，人工核验完成状态后复用，不覆盖')
    budget = data.get('budget', {})
    rate = budget.get('expected_gpu_h_per_candidate')
    estimate = total_candidates * rate if isinstance(rate, (int, float)) and rate > 0 else None
    limit = budget.get('approved_gpu_h')
    if estimate is None or not isinstance(limit, (int, float)) or limit < estimate:
        errors.append('预算未确认或低于估计GPU-h；还需单列评估/API成本')
    return {'valid': not errors, 'errors': errors, 'runs': len(runs),
            'candidate_training_upper_bound': total_candidates, 'estimated_train_gpu_h': estimate,
            'limitations': '静态校验不验证图像语义近重复、源图裁剪关系、外部训练集重叠、控制器真实行为或协议声明真实性。'}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--paper-root', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('validate')
    v.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        if args.out.exists():
            parser.error('输出文件已存在；指定新路径以保留历史报告')
        result = audit(args.paper_root)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = validate(json.loads(args.manifest.read_text(encoding='utf-8-sig')))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if result['valid'] else 2)


if __name__ == '__main__':
    main()
