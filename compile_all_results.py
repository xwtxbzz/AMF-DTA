"""Complete summary of ALL experimental results"""
import json, os

datasets = ['davis', 'kiba', 'metz']

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# ========== Load all result sources ==========

# 1. Baseline all200 (9 paper models)
baseline_all200 = {}
for model in ['attentiondta','deepdta','widedta','fingerdta','drugban',
              'graphdta','mgraphdta','moltrans','ssmdta']:
    for ds in datasets:
        f = f'outputs/{ds}_{model}_all200/test_metrics.json'
        d = load_json(f)
        if d:
            baseline_all200[(ds, model)] = d

# 2. CSatDTA (only davis)
csat = load_json('outputs/davis_csatdta_all200/test_metrics.json')
if csat:
    baseline_all200[('davis', 'csatdta')] = csat

# 3. Three selected models (EDTA, CKDTA, DEAttentionDTA)
three_selected = {}
three_paths = {
    ('davis','edta'): 'outputs/davis_edta_three_selected_davis/test_metrics.json',
    ('davis','ckdta'): 'outputs/davis_ckdta_three_selected_davis/test_metrics.json',
    ('davis','deattentiondta'): 'outputs/davis_deattentiondta_three_selected_davis/test_metrics.json',
    ('kiba','edta'): 'outputs/kiba_edta_three_selected/test_metrics.json',
    ('kiba','ckdta'): 'outputs/kiba_ckdta_three_selected/test_metrics.json',
    ('kiba','deattentiondta'): 'outputs/kiba_deattentiondta_three_selected/test_metrics.json',
    ('metz','edta'): 'outputs/metz_edta_three_selected/test_metrics.json',
    ('metz','ckdta'): 'outputs/metz_ckdta_three_selected/test_metrics.json',
    ('metz','deattentiondta'): 'outputs/metz_deattentiondta_three_selected/test_metrics.json',
}
for key, path in three_paths.items():
    d = load_json(path)
    if d:
        three_selected[key] = d

# 4. DrugBAN metz all200_stable
d = load_json('outputs/metz_drugban_all200_stable/test_metrics.json')
if d:
    baseline_all200[('metz', 'drugban')] = d

# 5. DeepDTA alternative paths (200 epoch not all200)
deepdta_200 = {}
for ds in datasets:
    f = f'outputs/{ds}_deepdta_200/test_metrics.json'
    d = load_json(f)
    if d:
        deepdta_200[(ds, 'deepdta')] = d

# 6. Multi-seed results
multi_seed = {}
for model in ['fingerdta', 'deepdta', 'widedta']:
    for ds in datasets:
        for seed in ['52','62','72','82']:
            f = f'outputs/{ds}_{model}_seed{seed}_200/test_metrics.json'
            d = load_json(f)
            if d:
                multi_seed[(ds, model, seed)] = d

# 7. MambaTransDTA multi-seed
mamba_seeds = {}
for ds in datasets:
    for seed in ['42','52','62','72','82']:
        f = f'outputs/{ds}_mambatransdta_seed{seed}_200/test_metrics.json'
        d = load_json(f)
        if d:
            mamba_seeds[(ds, 'mambatransdta', seed)] = d

# ========== Build model name mapping ==========
model_display = {
    'fingerdta':'FingerDTA', 'deepdta':'DeepDTA', 'widedta':'WideDTA',
    'attentiondta':'AttentionDTA', 'deattentiondta':'DEAttentionDTA',
    'edta':'EDTA', 'ckdta':'CKDTA', 'drugban':'DrugBAN',
    'graphdta':'GraphDTA', 'mgraphdta':'MGraphDTA',
    'moltrans':'MolTrans', 'ssmdta':'SSMDTA',
    'csatdta':'CSatDTA', 'mambatransdta':'MambaTransDTA',
}

# ========== Compile complete result table ==========
print('='*140)
print('              所有实验结果完整汇总 (ALL EXPERIMENTAL RESULTS)')
print('='*140)

# Section A: Models with complete 3-dataset 200-epoch results
print('\n' + '='*140)
print('A. 三数据集完整 200 轮结果 (完赛模型)')
print('='*140)

complete_models = []
for model_id, display in model_display.items():
    ok = True
    for ds in datasets:
        # Check all sources
        found = False
        if (ds, model_id) in baseline_all200:
            found = True
        if (ds, model_id) in three_selected:
            found = True
        if (ds, model_id) in deepdta_200:
            found = True
        if len([k for k in mamba_seeds if k[0]==ds and k[1]==model_id]) > 0:
            found = True
        if not found:
            ok = False
            break
    if ok:
        complete_models.append(model_id)

print(f'\n共 {len(complete_models)} 个模型在三数据集均有 200 轮结果:\n')

header = f'{"模型":<18} {"数据源":<20} {"Davis RMSE":<12} {"Davis CI":<12} {"KIBA RMSE":<12} {"KIBA CI":<12} {"Metz RMSE":<12} {"Metz CI":<12}'
print(header)
print('-'*110)

for model_id in complete_models:
    display = model_display[model_id]
    source_labels = []
    row = []
    for ds in datasets:
        data = None
        seed_count = 0
        if (ds, model_id) in baseline_all200:
            data = baseline_all200[(ds, model_id)]
            seed_count = 1
        elif (ds, model_id) in three_selected:
            data = three_selected[(ds, model_id)]
            seed_count = 1
        elif (ds, model_id) in deepdta_200:
            data = deepdta_200[(ds, model_id)]
            seed_count = 1
        seeds_found = [k[2] for k in mamba_seeds if k[0]==ds and k[1]==model_id]
        if seeds_found:
            # Use multi-seed average
            n = len(seeds_found)
            avg = {}
            for m in ['rmse','ci','r2','rm2','mae']:
                vals = [mamba_seeds[(ds, model_id, s)][m] for s in seeds_found]
                avg[m] = sum(vals)/n
            data = avg
            seed_count = n
        if data:
            row.extend([f'{data["rmse"]:.4f}', f'{data["ci"]:.4f}'])
            if seed_count > 1:
                source_labels.append(f'{ds}={seed_count}种子')
            else:
                src = 'all200'
                if (ds, model_id) in three_selected:
                    src = '1runs'
                elif (ds, model_id) in deepdta_200:
                    src = '200r'
                source_labels.append(f'{ds}={src}')
    source_str = ', '.join(source_labels)
    print(f'{display:<18} {source_str:<20} {row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}')

# Section B: Cross-model ranking
print('\n' + '='*140)
print('B. 三数据集综合排名 (按RMSE升序)')
print('='*140)
print()

rankings = []
for model_id in complete_models:
    all_rmse = []
    all_ci = []
    for ds in datasets:
        data = None
        if (ds, model_id) in baseline_all200:
            data = baseline_all200[(ds, model_id)]
        elif (ds, model_id) in three_selected:
            data = three_selected[(ds, model_id)]
        elif (ds, model_id) in deepdta_200:
            data = deepdta_200[(ds, model_id)]
        seeds = [k[2] for k in mamba_seeds if k[0]==ds and k[1]==model_id]
        if seeds:
            avg = {}
            for m in ['rmse','ci','r2','rm2','mae']:
                vals = [mamba_seeds[(ds, model_id, s)][m] for s in seeds]
                avg[m] = sum(vals)/len(vals)
            data = avg
        if data:
            all_rmse.append(data['rmse'])
            all_ci.append(data['ci'])
    if len(all_rmse) == 3:
        rankings.append({
            'model': model_display[model_id],
            'avg_rmse': sum(all_rmse)/3,
            'avg_ci': sum(all_ci)/3,
        })

rankings.sort(key=lambda x: x['avg_rmse'])

print(f'{"排名":<6} {"模型":<18} {"平均RMSE↓":<14} {"平均CI↑":<14}')
print('-'*52)
for i, r in enumerate(rankings, 1):
    print(f'{i:<6} {r["model"]:<18} {r["avg_rmse"]:<14.4f} {r["avg_ci"]:<14.4f}')

# Section C: Less complete results
print('\n' + '='*140)
print('C. 数据集不完整的模型 (缺失部分数据集或仅有smoke)')
print('='*140)
print()

incomplete = [
    ('CSatDTA', 'csatdta', '只有Davis all200 (RMSE=0.4689, CI=0.8881)\n  KIBA/Metz 无结果'),
    ('DeepConvDTI', '', '三数据集结果均异常 (Davis RMSE=4.51, KIBA RMSE=10.76, Metz RMSE=5.31)\n  R2 大幅负值，模型训练失败'),
]
for name, mid, desc in incomplete:
    print(f'  {name:<20} {desc}')
    print()

# Section D: Smoke-only (original models that need full training)
print('='*140)
print('D. 仅有 Smoke 结果的原创模型 (待完成 200 轮完整训练)')
print('='*140)

smoke_results = {
    'CGBiLSTM': {'rmse':0.9222,'ci':0.7550,'r2':-0.160},
    'DeepGLSTM': {'rmse':0.8112,'ci':0.7790,'r2':0.102},
    'DrugMGR': {'rmse':1.7823,'ci':0.7616,'r2':-3.334},
    'MATT-DTI': {'rmse':0.7235,'ci':0.7786,'r2':0.286},
}

print(f'\n{"模型":<18} {"数据集":<10} {"RMSE↓":<10} {"CI↑":<10} {"R2":<10} {"状态":<20}')
print('-'*78)
for model, metrics in smoke_results.items():
    print(f'{model:<18} {"davis":<10} {metrics["rmse"]:<10.4f} {metrics["ci"]:<10.4f} {metrics["r2"]:<10.4f} {"smoke/1轮":<20}')
print()

# Section E: Models with zero results
print('='*140)
print('E. 项目中存在代码但无实验结果的模型')
print('='*140)
print(f'\n  {"HiFDTA":<20} 无任何 outputs 结果')
print(f'  {"LaProDTA":<20} 无任何 outputs 结果')
print(f'  {"ViDTA":<20} 无任何 outputs 结果')
print()

# Section F: Best single-seed per model (for reference)
print('='*140)
print('F. 每模型最佳单次运行结果 (跨数据集平均, 用于 Baseline 参考)')
print('='*140)
print()

# Use all200 baseline for all except where _200 is better
print(f'{"模型":<18} {"来源":<18} {"RMSE↓":<10} {"CI↑":<10} {"R2↑":<10} {"RM2↑":<10}')
print('-'*66)

# FingerDTA: all200 (default) for comparison
finger_src = baseline_all200[('davis','fingerdta')]
finger_src_k = baseline_all200[('kiba','fingerdta')]
finger_src_m = baseline_all200[('metz','fingerdta')]
avg_rmse = (finger_src['rmse'] + finger_src_k['rmse'] + finger_src_m['rmse'])/3
avg_ci = (finger_src['ci'] + finger_src_k['ci'] + finger_src_m['ci'])/3
avg_r2 = (finger_src['r2'] + finger_src_k['r2'] + finger_src_m['r2'])/3
avg_rm2 = (finger_src['rm2'] + finger_src_k['rm2'] + finger_src_m['rm2'])/3
print(f'{"FingerDTA":<18} {"all200(单种子)":<18} {avg_rmse:<10.4f} {avg_ci:<10.4f} {avg_r2:<10.4f} {avg_rm2:<10.4f}')

# Multi-seed average for FingerDTA
all_seeds = ['52','62','72','82']
f_avg = {}
for m in ['rmse','ci','r2','rm2']:
    vals = []
    for ds in datasets:
        for s in all_seeds:
            d = multi_seed[(ds, 'fingerdta', s)]
            vals.append(d[m])
    f_avg[m] = sum(vals)/len(vals)
print(f'{"FingerDTA":<18} {"4种子平均":<18} {f_avg["rmse"]:<10.4f} {f_avg["ci"]:<10.4f} {f_avg["r2"]:<10.4f} {f_avg["rm2"]:<10.4f}')

# DeepDTA: all200
for src_name, src_keys in [("all200(默认)", baseline_all200), ("200单次", deepdta_200)]:
    dd_src = {}
    for ds in datasets:
        if (ds, 'deepdta') in src_keys:
            dd_src[ds] = src_keys[(ds, 'deepdta')]
    if len(dd_src)==3:
        avg_rmse = sum(dd_src[ds]['rmse'] for ds in datasets)/3
        avg_ci = sum(dd_src[ds]['ci'] for ds in datasets)/3
        print(f'{"DeepDTA":<18} {src_name:<18} {avg_rmse:<10.4f} {avg_ci:<10.4f}')

# WideDTA: all200
wd_src = baseline_all200
avg_rmse = sum(wd_src[(ds,'widedta')]['rmse'] for ds in datasets)/3
avg_ci = sum(wd_src[(ds,'widedta')]['ci'] for ds in datasets)/3
print(f'{"WideDTA":<18} {"all200(默认)":<18} {avg_rmse:<10.4f} {avg_ci:<10.4f}')

# MambaTransDTA 5-seed average
m_avg = {}
for m in ['rmse','ci','r2','rm2']:
    vals = []
    for ds in datasets:
        for s in ['42','52','62','72','82']:
            d = mamba_seeds[(ds, 'mambatransdta', s)]
            vals.append(d[m])
    m_avg[m] = sum(vals)/len(vals)
print(f'{"MambaTransDTA":<18} {"5种子平均":<18} {m_avg["rmse"]:<10.4f} {m_avg["ci"]:<10.4f} {m_avg["r2"]:<10.4f} {m_avg["rm2"]:<10.4f}')
print()

# Section G: Model source file summary
print('='*140)
print('G. 模型代码与实验结果对照总表')
print('='*140)
print()
print(f'{"模型":<18} {"代码存在":<10} {"Davis":<10} {"KIBA":<10} {"Metz":<10} {"说明":<50}')
print('-'*108)

all_code_models = sorted([f.replace('.py','') for f in os.listdir('models') if f.endswith('.py') and f != '__init__.py'])
for m in all_code_models:
    code = 'Y'
    results_status = []
    for ds in datasets:
        found = 'N'
        for source in [baseline_all200, three_selected, deepdta_200]:
            if (ds, m.lower()) in source:
                d = source[(ds, m.lower())]
                found = f'{d["rmse"]:.4f}'
                break
        for seed_dict in [multi_seed, mamba_seeds]:
            for k in seed_dict:
                if k[0]==ds and k[1]==m.lower():
                    found = 'multi-sd'
                    break
        results_status.append(found)
    display = model_display.get(m.lower(), m)

    # Determine description
    desc = ''
    if all(r != '✗' for r in results_status):
        desc = '三数据集完成'
    elif results_status[0] != '✗' and results_status[1] == '✗' and results_status[2] == '✗':
        desc = '仅 Davis'
    elif all(r == '✗' for r in results_status):
        desc = '无结果'
    else:
        desc = '部分完成'

    # Special notes
    if m.lower() == 'deepconvdti':
        desc = '结果异常(RMSE>4)'
    if m.lower() in ['cgbilstm','deepglstm','drugmgr','mattdti']:
        desc = '仅smoke'

    print(f'{display:<18} {code:<10} {results_status[0]:<10} {results_status[1]:<10} {results_status[2]:<10} {desc:<50}')
