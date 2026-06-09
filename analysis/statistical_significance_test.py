"""
统计显著性检验：检验 DP-SGD 与 Baseline 的差异是否统计显著
使用 Mann-Whitney U test (非参数检验，适用于小样本)
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict


def load_all_metrics():
    """加载所有实验结果"""
    runs_dir = Path('outputs/runs')
    results = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.name.startswith('v2_'):
            continue
        
        metrics_path = run_dir / 'metrics.json'
        config_path = run_dir / 'config.json'
        
        if not metrics_path.exists() or not config_path.exists():
            continue
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        
        results.append({
            'run_id': run_dir.name,
            'config': config,
            'metrics': metrics
        })
    
    return results


def statistical_comparison(results):
    """进行统计比较"""
    # 按条件分组
    groups = defaultdict(list)
    
    for r in results:
        cfg = r['config']
        key = (cfg['dataset'], cfg['model'], cfg.get('model_variant') or 'N/A',
               cfg['train_defense'], cfg.get('eps'))
        groups[key].append(r['metrics'])
    
    # 提取 baseline 和 DP-SGD 的结果
    comparisons = []
    
    for dataset in ['OULAD']:
        for model in ['MLP']:
            for variant in ['small', 'large']:
                baseline_key = (dataset, model, variant, 'none', None)
                
                for eps in [1, 5, 10]:
                    dpsgd_key = (dataset, model, variant, 'DP-SGD', eps)
                    
                    if baseline_key not in groups or dpsgd_key not in groups:
                        continue
                    
                    baseline_metrics = groups[baseline_key]
                    dpsgd_metrics = groups[dpsgd_key]
                    
                    # 提取 Test AUC
                    baseline_auc = [m['test_auc'] for m in baseline_metrics if m.get('test_auc')]
                    dpsgd_auc = [m['test_auc'] for m in dpsgd_metrics if m.get('test_auc')]
                    
                    # 提取 TPR Gap
                    baseline_tpr = [m['worst_group_tpr_gap'] for m in baseline_metrics 
                                   if m.get('worst_group_tpr_gap') is not None]
                    dpsgd_tpr = [m['worst_group_tpr_gap'] for m in dpsgd_metrics 
                                if m.get('worst_group_tpr_gap') is not None]
                    
                    # Mann-Whitney U test
                    if len(baseline_auc) >= 3 and len(dpsgd_auc) >= 3:
                        auc_stat, auc_p = stats.mannwhitneyu(baseline_auc, dpsgd_auc, alternative='two-sided')
                    else:
                        auc_stat, auc_p = None, None
                    
                    if len(baseline_tpr) >= 3 and len(dpsgd_tpr) >= 3:
                        tpr_stat, tpr_p = stats.mannwhitneyu(baseline_tpr, dpsgd_tpr, alternative='two-sided')
                    else:
                        tpr_stat, tpr_p = None, None
                    
                    comparisons.append({
                        'dataset': dataset,
                        'model': model,
                        'variant': variant,
                        'eps': eps,
                        'baseline_auc': baseline_auc,
                        'dpsgd_auc': dpsgd_auc,
                        'auc_p_value': auc_p,
                        'baseline_tpr': baseline_tpr,
                        'dpsgd_tpr': dpsgd_tpr,
                        'tpr_p_value': tpr_p,
                        'n_baseline': len(baseline_auc),
                        'n_dpsgd': len(dpsgd_auc)
                    })
    
    return comparisons


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """计算 bootstrap 置信区间"""
    if len(data) < 3:
        return None, None
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return lower, upper


def compute_confidence_intervals(results):
    """计算所有条件的置信区间"""
    groups = defaultdict(list)
    
    for r in results:
        cfg = r['config']
        key = (cfg['dataset'], cfg['model'], cfg.get('model_variant') or 'N/A',
               cfg['train_defense'], cfg.get('eps'))
        groups[key].append(r['metrics'])
    
    ci_results = {}
    
    for key, metrics_list in groups.items():
        test_aucs = [m['test_auc'] for m in metrics_list if m.get('test_auc')]
        tpr_gaps = [m['worst_group_tpr_gap'] for m in metrics_list 
                   if m.get('worst_group_tpr_gap') is not None]
        
        if test_aucs:
            auc_lower, auc_upper = bootstrap_confidence_interval(test_aucs)
            ci_results[key] = {
                'test_auc_mean': np.mean(test_aucs),
                'test_auc_ci': (auc_lower, auc_upper),
                'test_auc_std': np.std(test_aucs),
                'n': len(test_aucs)
            }
        
        if tpr_gaps:
            tpr_lower, tpr_upper = bootstrap_confidence_interval(tpr_gaps)
            if key not in ci_results:
                ci_results[key] = {}
            ci_results[key]['tpr_gap_mean'] = np.mean(tpr_gaps)
            ci_results[key]['tpr_gap_ci'] = (tpr_lower, tpr_upper)
            ci_results[key]['tpr_gap_std'] = np.std(tpr_gaps)
    
    return ci_results


def format_p_value(p):
    """格式化 p-value"""
    if p is None:
        return "N/A"
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}ns"


if __name__ == "__main__":
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    results = load_all_metrics()
    print(f"\nLoaded {len(results)} experiment results")
    
    # 统计比较
    comparisons = statistical_comparison(results)
    
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U TEST: Baseline vs DP-SGD")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Model':<12} {'Eps':<6} {'Metric':<15} {'Baseline':<20} {'DP-SGD':<20} {'p-value':<15}")
    print("-" * 80)
    
    for comp in comparisons:
        baseline_auc_str = f"{np.mean(comp['baseline_auc']):.4f}±{np.std(comp['baseline_auc']):.4f}"
        dpsgd_auc_str = f"{np.mean(comp['dpsgd_auc']):.4f}±{np.std(comp['dpsgd_auc']):.4f}"
        
        print(f"{comp['dataset']:<10} {comp['model']}-{comp['variant']:<8} {comp['eps']:<6} {'Test AUC':<15} {baseline_auc_str:<20} {dpsgd_auc_str:<20} {format_p_value(comp['auc_p_value']):<15}")
        
        if comp['baseline_tpr'] and comp['dpsgd_tpr']:
            baseline_tpr_str = f"{np.mean(comp['baseline_tpr']):.4f}±{np.std(comp['baseline_tpr']):.4f}"
            dpsgd_tpr_str = f"{np.mean(comp['dpsgd_tpr']):.4f}±{np.std(comp['dpsgd_tpr']):.4f}"
            print(f"{'':10} {'':12} {'':6} {'TPR Gap':<15} {baseline_tpr_str:<20} {dpsgd_tpr_str:<20} {format_p_value(comp['tpr_p_value']):<15}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    
    # 置信区间
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 80)
    
    ci_results = compute_confidence_intervals(results)
    
    print(f"{'Dataset':<10} {'Model':<15} {'Defense':<15} {'Eps':<6} {'Test AUC':<30} {'TPR Gap':<30}")
    print("-" * 80)
    
    for key in sorted(ci_results.keys())[:10]:
        dataset, model, variant, defense, eps = key
        ci = ci_results[key]
        
        model_str = f"{model}-{variant}"
        def_str = defense if defense != 'none' else 'Baseline'
        eps_str = str(eps) if eps else "--"
        
        auc_ci = ci.get('test_auc_ci', (None, None))
        auc_str = f"{ci['test_auc_mean']:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]" if auc_ci[0] else "N/A"
        
        tpr_ci = ci.get('tpr_gap_ci', (None, None))
        tpr_str = f"{ci['tpr_gap_mean']:.4f} [{tpr_ci[0]:.4f}, {tpr_ci[1]:.4f}]" if tpr_ci and tpr_ci[0] else "N/A"
        
        print(f"{dataset:<10} {model_str:<15} {def_str:<15} {eps_str:<6} {auc_str:<30} {tpr_str:<30}")
    
    # 保存结果
    output_path = Path('outputs/reports/statistical_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'comparisons': [
                {k: (v if not isinstance(v, list) else [float(x) for x in v]) 
                 for k, v in comp.items()}
                for comp in comparisons
            ],
            'confidence_intervals': {
                str(k): {
                    'test_auc_mean': float(v['test_auc_mean']),
                    'test_auc_ci': [float(v['test_auc_ci'][0]), float(v['test_auc_ci'][1])] if v['test_auc_ci'][0] else None,
                    'tpr_gap_mean': float(v['tpr_gap_mean']) if 'tpr_gap_mean' in v else None,
                    'tpr_gap_ci': [float(v['tpr_gap_ci'][0]), float(v['tpr_gap_ci'][1])] if 'tpr_gap_ci' in v and v['tpr_gap_ci'][0] else None,
                }
                for k, v in ci_results.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
