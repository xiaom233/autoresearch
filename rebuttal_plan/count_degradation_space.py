"""不导入训练/退化模块，静态解析实际候选表并精确计数。"""
import argparse
import ast
import itertools
import json
import math
from pathlib import Path


def literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{path}: missing {name}')


def count(root):
    library = literal_assignment(root / 'x_distortion/__init__.py', 'distortions_dict')
    categories = literal_assignment(root / 'blind_challenge.py', 'DEG_CATEGORIES')
    blacklist = literal_assignment(root / 'blind_challenge.py', 'BLACKLIST')
    priority = literal_assignment(root / 'blind_challenge.py', 'PHYSICAL_ORDER_CATEGORY')
    active = {k: [f for f in v if f not in blacklist] for k, v in categories.items()}
    levels = 5  # add_distortion explicitly validates severity in [1,2,3,4,5]
    rows = []
    for n in range(1, len(active) + 1):
        types_unordered, types_ordered = 0, 0
        for cats in itertools.combinations(active, n):
            for funcs in itertools.product(*(active[c] for c in cats)):
                types_unordered += 1
                choices = [([0, 2] if c == 'global' and f.startswith('brightness') else [priority[c]])
                           for c, f in zip(cats, funcs)]
                valid_orders = set()
                for priorities in itertools.product(*choices):
                    for order in itertools.permutations(range(n)):
                        if all(priorities[order[i]] <= priorities[order[i+1]] for i in range(n-1)):
                            valid_orders.add(tuple(funcs[i] for i in order))
                types_ordered += len(valid_orders)
        rows.append({'steps': n, 'category_distinct_function_sets': types_unordered,
                     'unordered_type_severity': types_unordered * levels ** n,
                     'physical_order_type_chains': types_ordered,
                     'physical_order_type_severity': types_ordered * levels ** n})
    full = len(set(itertools.chain.from_iterable(library.values())))
    # These are sizes of current calibration scripts, not proof of all historic calibration.
    blur = root / 'scripts/calibrate_blur_kernel.py'
    b = {k: literal_assignment(blur, k) for k in ['BLUR_TYPES','BLUR_SEVS','NOISE_TYPES','NOISE_SEVS','ORDERS','SEEDS']}
    blur_configs = math.prod(len(b[k]) for k in b if k != 'SEEDS')
    jpeg = root / 'scripts/calibrate_jpeg_thresholds.py'
    j = {k: literal_assignment(jpeg, k) for k in ['JPEG_SEVS','NOISE_TYPES','NOISE_SEVS','BLUR_TYPES','BLUR_SEVS','ORDERS','SEEDS']}
    js, nt, ns, bt, bs, orders, seeds = [len(j[k]) for k in ['JPEG_SEVS','NOISE_TYPES','NOISE_SEVS','BLUR_TYPES','BLUR_SEVS','ORDERS','SEEDS']]
    jpeg_configs = js + js*nt*ns + js*bt*bs + js*bt*bs*nt*ns*orders
    jpeg_cases = js*10*seeds + js*nt*ns*10*seeds + js*bt*bs*10*seeds + js*bt*bs*nt*ns*5*seeds*orders
    return {'definition': 'distinct symbolic pipelines, not distinct pixel-level distributions; images/seeds not counted as configurations',
            'library_functions': full, 'active_categories': {k: len(v) for k,v in active.items()},
            'active_functions': sum(map(len, active.values())), 'severity_levels': levels,
            'rows': rows, 'total_1_to_3_unordered': sum(r['unordered_type_severity'] for r in rows[:3]),
            'total_1_to_3_physical_ordered': sum(r['physical_order_type_severity'] for r in rows[:3]),
            'full_library_unrestricted_1_to_3_upper_reference': sum((full*levels)**n for n in range(1,4)),
            'blur_calibration_script': {'unique_configs': blur_configs, 'seeds': len(b['SEEDS']), 'images': 1, 'cases': blur_configs * len(b['SEEDS'])},
            'jpeg_calibration_script': {'unique_configs': jpeg_configs, 'cases_from_declared_10_and_5_images': jpeg_cases},
            'historic_3840_unique_configs': None,
            'limits': ['Exact only for current generator: categories cannot repeat; brightness gets priority 0 or 2.',
                       'All five severity labels count even if two labels happen to have equivalent output.',
                       'Historic exp41 generator version and historic calibration manifest must be verified.',
                       'Small coverage does not prove calibration/evaluation independence.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; choose a new report path')
    result = count(args.code_root)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
