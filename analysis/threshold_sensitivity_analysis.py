"""
阈值敏感性分析：计算不同阈值 (0.3, 0.5, 0.7) 下的 Fairness Gaps
使用已完成的实验结果中的预测文件
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


def load_predictions(run_dir):
    """加载预测结果"""
    pred_path = Path(run_dir) / 'predictions_released.npy'
    if not pred_path.exists():
        pred_path = Path(run_dir) / 'predictions_base.npy'
    
    if pred_path.exists():
        return np.load(pred_path)
    return None


def load_labels_and_groups(run_dir):
    """加载标签和分组信息"""
    run_path = Path(run_dir)
    
    # 尝试直接加载已保存的文件
    labels_path = run_path / 'test_labels.npy'
    groups_path = run_path / 'groups.npy'
    
    y_true = None
    groups = None
    
    if labels_path.exists():
        y_true = np.load(labels_path)
    
    if groups_path.exists():
        groups = np.load(groups_path)
    
    # 如果文件不存在，尝试从数据集重新加载
    if y_true is None or groups is None:
        config_path = run_path / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            dataset = config.get('dataset', '')
            seed = config.get('seed', 42)
            
            from src.data_loader import load_dataset
            try:
                X_train, X_test, y_train, y_test, groups_test = load_dataset(dataset, seed=seed)
                if y_true is None:
                    y_true = y_test
                if groups is None:
                    groups = groups_test
            except:
                pass
    
    return y_true, groups


def compute_fairness_metrics_at_threshold(y_true, y_pred_proba, groups, threshold=0.5):
    """计算指定阈值下的公平性指标"""
    if groups is None or y_true is None or y_pred_proba is None:
        return None, None, None
    
    # 确保输入是 numpy 数组
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    groups = np.asarray(groups)
    
    # 如果 y_pred_proba 是二维的 (n, 2)，取第二列作为正类概率
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # 二值化预测 - 确保是整数类型
    y_pred = (y_pred_proba >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)
    
    # 计算每组的 TPR, FPR, FNR
    unique_groups = np.unique(groups)
    tprs = []
    fprs = []
    fnrs = []
    
    for g in unique_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        
        y_true_g = y_true[mask].astype(np.int32)
        y_pred_g = y_pred[mask].astype(np.int32)
        
        # 计算混淆矩阵
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
            fnrs.append(fnr)
        except:
            continue
    
    if len(tprs) < 2:
        return None, None, None
    
    tpr_gap = max(tprs) - min(tprs)
    fpr_gap = max(fprs) - min(fprs)
    fnr_gap = max(fnrs) - min(fnrs)
    
    return tpr_gap, fpr_gap, fnr_gap


def analyze_threshold_sensitivity():
    """分析阈值敏感性"""
    runs_dir = Path('outputs/runs')
    
    thresholds = [0.3, 0.5, 0.7]
    results = []
    
    # 只分析 OULAD 实验（有 groups）
    for run_dir in runs_dir.iterdir():
        if not run_dir.name.startswith('v2_'):
            continue
        
        config_path = run_dir / 'config.json'
        if not config_path.exists():
            continue
        
        with open(config_path) as f:
            config = json.load(f)
        
        if config.get('dataset') != 'OULAD':
            continue
        
        # 加载预测和标签
        y_pred_proba = load_predictions(run_dir)
        y_true, groups = load_labels_and_groups(run_dir)
        
        if y_pred_proba is None or y_true is None or groups is None:
            continue
        
        # 确保维度匹配
        if len(y_pred_proba) != len(y_true):
            # 使用第二个列（正类概率）
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]
        
        if len(y_pred_proba) != len(y_true):
            continue
        
        # 计算不同阈值下的指标
        threshold_results = {}
        for tau in thresholds:
            tpr_gap, fpr_gap, fnr_gap = compute_fairness_metrics_at_threshold(
                y_true, y_pred_proba, groups, threshold=tau
            )
            threshold_results[tau] = {
                'tpr_gap': tpr_gap,
                'fpr_gap': fpr_gap,
                'fnr_gap': fnr_gap
            }
        
        results.append({
            'run_id': run_dir.name,
            'model': config.get('model', 'N/A'),
            'model_variant': config.get('model_variant', 'N/A'),
            'train_defense': config.get('train_defense', 'none'),
            'eps': config.get('eps'),
            'threshold_results': threshold_results
        })
    
    return results, thresholds


def aggregate_by_condition(results, thresholds):
    """按条件聚合结果"""
    from collections import defaultdict
    
    groups = defaultdict(lambda: {tau: [] for tau in thresholds})
    
    for r in results:
        key = (r['model'], r.get('model_variant') or 'N/A', r['train_defense'], r.get('eps'))
        
        for tau in thresholds:
            tr = r['threshold_results'].get(tau, {})
            if tr.get('tpr_gap') is not None:
                groups[key][tau].append(tr['tpr_gap'])
    
    # 计算平均值
    summary = {}
    for key, tau_data in groups.items():
        summary[key] = {}
        for tau, values in tau_data.items():
            if values:
                summary[key][tau] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n': len(values)
                }
    
    return summary


def generate_latex_table(summary, thresholds):
    """生成 LaTeX 表格"""
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Threshold Sensitivity Analysis: TPR Gap Across Decision Thresholds ($\\tau$)}")
    latex.append("\\label{tab:threshold_sensitivity}")
    latex.append("\\begin{tabular}{lllccc}")
    latex.append("\\toprule")
    latex.append("Model & Defense & Epsilon & $\\tau=0.3$ & $\\tau=0.5$ & $\\tau=0.7$ \\\\")
    latex.append("\\midrule")
    
    # 排序并输出
    sorted_keys = sorted(summary.keys(), key=lambda x: (x[0], x[2] or 0))
    
    for key in sorted_keys:
        model, variant, defense, eps = key
        model_str = f"{model}-{variant}" if variant else model
        def_str = defense if defense != 'none' else 'Baseline'
        eps_str = str(eps) if eps else "--"
        
        vals = []
        for tau in thresholds:
            s = summary[key].get(tau, {})
            if s:
                vals.append(f"${s['mean']:.3f} \\pm {s['std']:.3f}$")
            else:
                vals.append("N/A")
        
        latex.append(f"{model_str} & {def_str} & {eps_str} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


if __name__ == "__main__":
    print("Analyzing threshold sensitivity...")
    results, thresholds = analyze_threshold_sensitivity()
    
    print(f"\nLoaded {len(results)} experiment results with predictions")
    
    if results:
        summary = aggregate_by_condition(results, thresholds)
        
        print("\n" + "="*80)
        print("THRESHOLD SENSITIVITY RESULTS")
        print("="*80)
        
        for key, tau_data in sorted(summary.items()):
            model, variant, defense, eps = key
            print(f"\n{model} {variant or ''} | {defense} | eps={eps}")
            for tau, s in sorted(tau_data.items()):
                print(f"  tau={tau}: TPR Gap = {s['mean']:.4f} ± {s['std']:.4f} (n={s['n']})")
        
        print("\n" + "="*80)
        print("LATEX TABLE")
        print("="*80)
        latex_table = generate_latex_table(summary, thresholds)
        print(latex_table)
        
        # 保存结果
        output_path = Path('outputs/reports/threshold_sensitivity_analysis.json')
        with open(output_path, 'w') as f:
            json.dump({
                'n_experiments': len(results),
                'thresholds': thresholds,
                'summary': {str(k): v for k, v in summary.items()},
                'latex_table': latex_table
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No results available. Make sure experiments have completed and predictions are saved.")
