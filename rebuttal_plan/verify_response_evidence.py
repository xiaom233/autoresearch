"""复核回应稿使用的已提交数值，不运行训练。"""
import argparse
import json
import statistics
from pathlib import Path


def verify(repo):
    r = repo / 'rebuttal_runs'
    d = json.loads((r / 'E1E2_fixed_lq_summary.json').read_text(encoding='utf-8'))
    output = {'synthetic': {}, 'real': {}, 'limits': 'Checks submitted data consistency only; not training provenance, official interfaces, or experimental independence.'}
    for exp in ['e1', 'e2']:
        out = output['synthetic'][exp] = {}
        for method, datasets in d[exp]['full'].items():
            out[method] = {}
            for ds, values in datasets.items():
                assert values['n'] == 24
                if 'per_scenario' in values:
                    assert len(values['per_scenario']) == 24
                    assert abs(statistics.mean(values['per_scenario'].values()) - values['psnr']) < 1e-9
                out[method][ds] = values['psnr']
    output['e1_rule_minus_autoir'] = d['e1']['paired_full']
    output['e2_ft_best_minus_autoir'] = d['e2']['paired_full']['DFPIR-ft-best']
    assert all(v['ci'][0] < 0 < v['ci'][1] for v in d['e1']['paired_full'].values())
    assert d['e2']['paired_full']['DFPIR-ft-best']['LSDIR']['ci'][0] > 0
    for dataset, expected in [('sidd', 1120), ('uhdm', 495)]:
        output['real'][dataset] = {}
        summary_name = 'full_eval_summary_extended.json' if dataset == 'sidd' else 'full_eval_summary.json'
        summary = json.loads((r / f'E4/{dataset}' / summary_name).read_text(encoding='utf-8'))
        references = summary if dataset == 'sidd' else summary['methods']
        groups = {}
        for f in sorted((r / f'E4/{dataset}/full_eval').glob('*_gpu*.jsonl')):
            groups.setdefault(f.name.split('_gpu')[0], []).append(f)
        for method, files in groups.items():
            rows = [json.loads(line) for p in files for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
            assert len(rows) == expected, (dataset, method, len(rows))
            assert len({x['image_id'] for x in rows}) == expected, (dataset, method, 'duplicate image IDs')
            mean = statistics.mean(x['psnr_rgb'] for x in rows)
            assert abs(mean - references[method]['psnr']) < 1e-9
            output['real'][dataset][method] = {'n': len(rows), 'psnr': mean,
                'ssim': statistics.mean(x['ssim_rgb'] for x in rows),
                'lpips': statistics.mean(x['lpips_vgg'] for x in rows)}
    output['status'] = 'PASS'
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists; choose a new path')
    result = verify(a.repo)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: synthetic summaries, confidence intervals and all real-data image counts/means')


if __name__ == '__main__':
    main()
