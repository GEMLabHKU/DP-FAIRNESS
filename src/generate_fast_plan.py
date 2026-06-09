"""
生成reviewer-proof fast实验计划（压缩到≤260 runs）
严格按照压缩策略：
- 固定 seeds=5 仅保留 core；diagnostic runs 保留 <=6 条并单独标记 appendix
- release defenses 只保留 2 个强度档（low/high）
- DP-SGD ε 只在 MLP-small + MLP-large 上做；LR/XGB 只跑 non-DP + release defenses
- HarvardX/UCI697 不需要跑 intersectional 相关设置
"""
import json
import os
from pathlib import Path
from collections import defaultdict

def generate_fast_plan():
    """生成fast计划（reviewer-proof，压缩到≤260）"""
    plan = []
    run_id = 0
    
    # ========== 1. 数据集（3个必须） ==========
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    
    # ========== 2. 模型配置（每数据集至少4个） ==========
    models_config = {
        "OULAD": [
            {"model": "LR", "variant": None},
            {"model": "XGBoost", "variant": None},
            {"model": "MLP", "variant": "small"},
            {"model": "MLP", "variant": "large"},
        ],
        "UCI697": [
            {"model": "LR", "variant": None},
            {"model": "XGBoost", "variant": None},
            {"model": "MLP", "variant": "small"},
            {"model": "MLP", "variant": "large"},
        ],
        "HarvardX_PersonCourse": [
            {"model": "LR", "variant": None},
            {"model": "XGBoost", "variant": None},
            {"model": "MLP", "variant": "small"},
            {"model": "MLP", "variant": "large"},
        ],
    }
    
    # ========== 3. 防御配置（压缩策略） ==========
    # OULAD: 
    #   - LR/XGB: none + release defenses (2档强度)
    #   - MLP: none + DP-SGD@ε∈{1,5,10} + release defenses (2档强度)
    # UCI697/HarvardX:
    #   - LR/XGB: none + release defenses (2档强度)
    #   - MLP: none + DP-SGD@ε∈{1,5,10} + release defenses (2档强度)
    
    defenses_config = {
        "OULAD": [
            # 训练端防御（主设置）
            {"train_defense": "none", "publish_defense": None, "eps": None},
            # DP-SGD（仅MLP，在get_fairness_seeds_config中过滤）
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 1},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 5},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 10},
            # 发布端防御：只保留2个强度档（low/high），只在MLP上做
            # Coarsening配置
            {"train_defense": "none", "publish_defense": "output_coarsening", "eps": None, 
             "coarsening_type": "label-only", "coarsening_step": 0.05, "intensity": "low"},
            {"train_defense": "none", "publish_defense": "output_coarsening", "eps": None,
             "coarsening_type": "rounding", "coarsening_step": 0.10, "intensity": "high"},
            # Perturbation配置（Gaussian和Laplace，scale=0.1）
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            # DP-SGD + release defense 组合
            {"train_defense": "DP-SGD", "publish_defense": "output_coarsening", "eps": 5,
             "coarsening_type": "label-only", "coarsening_step": 0.05, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
        ],
        "UCI697": [
            {"train_defense": "none", "publish_defense": None, "eps": None},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 1},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 5},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 10},
            # Perturbation配置（Gaussian和Laplace，scale=0.1）
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
        ],
        "HarvardX_PersonCourse": [
            {"train_defense": "none", "publish_defense": None, "eps": None},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 1},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 5},
            {"train_defense": "DP-SGD", "publish_defense": None, "eps": 10},
            # Perturbation配置（Gaussian和Laplace，scale=0.1）
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "none", "publish_defense": "output_perturbation", "eps": None,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 1,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 5,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "gaussian", "noise_scale": 0.1, "intensity": "low"},
            {"train_defense": "DP-SGD", "publish_defense": "output_perturbation", "eps": 10,
             "noise_type": "laplace", "noise_scale": 0.1, "intensity": "low"},
        ],
    }
    
    # ========== 4. Visibility和Q ==========
    visibility_options = ["full", "label-only"]
    Q = 5  # 固定
    
    # ========== 5. Fairness配置 ==========
    # OULAD必须：gender, disability, age_band, intersectional (gender×disability)
    # HarvardX/UCI：无demographic字段，fairness="NA"，不需要intersectional
    
    def get_fairness_seeds_config(dataset, model_info, defense_info, visibility, fairness_attr):
        """
        获取fairness和seeds配置（压缩策略）
        返回：(seeds, is_core, is_diagnostic)
        - is_core=True: 主设置，seeds=5，进入主结果表
        - is_diagnostic=True: 诊断设置，seeds=2，只进入附录表（最多6条）
        """
        if dataset == "OULAD":
            model_name = model_info["model"]
            model_var = model_info.get("variant")
            train_def = defense_info["train_defense"]
            pub_def = defense_info.get("publish_defense")
            eps = defense_info.get("eps")
            
            # ===== 压缩策略：LR/XGB 不跑 DP-SGD =====
            if train_def == "DP-SGD" and model_name in ["LR", "XGBoost"]:
                return None, False, False
            
            # ===== 主设置（seeds=5，进入主结果表） =====
            # 核心：OULAD × (LR, XGBoost, MLP-small, MLP-large) × (none, DP-SGD@ε∈{1,5,10}) × full × gender × seeds=5
            is_core_main = (
                model_name in ["LR", "XGBoost", "MLP"] and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps in [1, 5, 10]) and
                (train_def != "DP-SGD" or model_name == "MLP") and  # DP-SGD 仅MLP
                pub_def is None and
                visibility == "full" and
                fairness_attr == "gender"
            )
            
            if is_core_main:
                return 5, True, False
            
            # 扩展fairness（主设置）：OULAD × MLP × (none, DP@ε=5) × full × (disability, age_band) × seeds=5
            # 压缩：只保留 DP@ε=5（不跑所有ε值）
            if (
                model_name == "MLP" and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps == 5) and  # 只保留ε=5
                pub_def is None and
                visibility == "full" and
                fairness_attr in ["disability", "age_band"]
            ):
                return 5, True, False
            
            # 扩展intersectional（主设置）：OULAD × MLP × (none, DP@ε=5) × full × gender_x_disability × seeds=5
            if (
                model_name == "MLP" and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps == 5) and
                pub_def is None and
                visibility == "full" and
                fairness_attr == "gender_x_disability"
            ):
                return 5, True, False
            
            # 发布端防御（主设置）：OULAD × MLP × (none+DP@ε=5) × publish defenses × full × gender × seeds=5
            # 压缩：只在MLP上做release defenses（包括coarsening和perturbation）
            if (
                model_name == "MLP" and
                pub_def is not None and
                visibility == "full" and
                fairness_attr == "gender"
            ):
                return 5, True, False
            
            # 发布端防御（主设置）：OULAD × (LR, XGBoost) × none × perturbation × full × gender × seeds=5
            # 对于LR/XGBoost，只做none + perturbation（不做DP-SGD）
            if (
                model_name in ["LR", "XGBoost"] and
                train_def == "none" and
                pub_def == "output_perturbation" and
                visibility == "full" and
                fairness_attr == "gender"
            ):
                return 5, True, False
            
            # ===== 诊断设置（seeds=2，只进入附录表，最多6条） =====
            # 扩展visibility：OULAD × MLP × (none, DP@ε=5) × label-only × gender × seeds=2（保留2条）
            if (
                model_name == "MLP" and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps == 5) and
                pub_def is None and
                visibility == "label-only" and
                fairness_attr == "gender"
            ):
                return 2, False, True
            
            # 其他情况跳过
            return None, False, False
        
        elif dataset in ["UCI697", "HarvardX_PersonCourse"]:
            # UCI697/HarvardX：无demographic字段，fairness="NA"
            # 主设置：所有模型 × (none, DP@ε∈{1,5,10}) × full × NA × seeds=5
            # 压缩策略：LR/XGB 不跑 DP-SGD
            model_name = model_info["model"]
            train_def = defense_info["train_defense"]
            pub_def = defense_info.get("publish_defense")
            eps = defense_info.get("eps")
            
            # LR/XGB 不跑 DP-SGD
            if train_def == "DP-SGD" and model_name in ["LR", "XGBoost"]:
                return None, False, False
            
            if (
                model_name in ["LR", "XGBoost", "MLP"] and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps in [1, 5, 10]) and
                (train_def != "DP-SGD" or model_name == "MLP") and  # DP-SGD 仅MLP
                pub_def is None and
                visibility == "full" and
                fairness_attr == "NA"
            ):
                return 5, True, False
            
            # 发布端防御（主设置）：UCI697/HarvardX × (LR, XGBoost, MLP) × (none+DP-SGD) × perturbation × full × NA × seeds=5
            if (
                model_name in ["LR", "XGBoost", "MLP"] and
                train_def in ["none", "DP-SGD"] and
                (train_def != "DP-SGD" or eps in [1, 5, 10]) and
                (train_def != "DP-SGD" or model_name == "MLP") and  # DP-SGD 仅MLP
                pub_def == "output_perturbation" and
                visibility == "full" and
                fairness_attr == "NA"
            ):
                return 5, True, False
            
            return None, False, False
        
        return None, False, False
    
    # ========== 6. 生成计划条目 ==========
    diagnostic_config_count = 0
    max_diagnostic_configs = 3  # 最多3个diagnostic配置（每个seeds=2，共6 runs）
    
    for dataset in datasets:
        models = models_config[dataset]
        defenses = defenses_config[dataset]
        
        # Fairness属性配置
        if dataset == "OULAD":
            fairness_attrs = ["gender", "disability", "age_band", "gender_x_disability"]
        else:
            fairness_attrs = ["NA"]  # UCI697/HarvardX 不需要intersectional
        
        for model_info in models:
            for defense_info in defenses:
                # DP-SGD只对MLP（在get_fairness_seeds_config中已处理，这里双重保险）
                if defense_info["train_defense"] == "DP-SGD" and model_info["model"] != "MLP":
                    continue
                
                for visibility in visibility_options:
                    for fairness_attr in fairness_attrs:
                        seeds, is_core, is_diagnostic = get_fairness_seeds_config(
                            dataset, model_info, defense_info, visibility, fairness_attr
                        )
                        
                        if seeds is None:
                            continue
                        
                        # 诊断runs限制：最多3个配置（每个配置seeds=2，共6 runs）
                        if is_diagnostic:
                            if diagnostic_config_count >= max_diagnostic_configs:
                                continue
                            diagnostic_config_count += 1
                        
                        for seed in range(1, seeds + 1):
                            entry = {
                                "run_id": f"fast_{run_id:04d}",
                                "dataset": dataset,
                                "model": model_info["model"],
                                "model_variant": model_info.get("variant"),
                                "train_defense": defense_info["train_defense"],
                                "publish_defense": defense_info.get("publish_defense"),
                                "eps": defense_info.get("eps"),
                                "visibility": visibility,
                                "Q": Q,
                                "seed": seed,
                                "fairness_attribute": fairness_attr,
                                "coarsening_type": defense_info.get("coarsening_type"),
                                "coarsening_step": defense_info.get("coarsening_step"),
                                "noise_type": defense_info.get("noise_type"),
                                "noise_scale": defense_info.get("noise_scale"),
                                "intensity": defense_info.get("intensity"),
                                "is_core": is_core,
                                "is_diagnostic": is_diagnostic,
                            }
                            plan.append(entry)
                            run_id += 1
    
    # ========== 添加负控制runs ==========
    # Random Labels: 所有数据集和模型，seeds=5
    for dataset in datasets:
        for model_info in models_config[dataset]:
            for seed in range(1, 6):  # seeds 1-5
                # Windows路径问题：使用 "N" 而不是 "N/A"
                variant_str = model_info['variant'] or 'N'
                entry = {
                    "run_id": f"negative_control_random_labels_{dataset}_{model_info['model']}_{variant_str}_seed{seed}",
                    "dataset": dataset,
                    "model": model_info["model"],
                    "model_variant": model_info["variant"],
                    "train_defense": "none",
                    "publish_defense": None,
                    "eps": None,
                    "visibility": "full",
                    "Q": 5,
                    "seed": seed,
                    "fairness_attribute": "NA" if dataset != "OULAD" else "gender",
                    "coarsening_type": None,
                    "coarsening_step": None,
                    "noise_type": None,
                    "noise_scale": None,
                    "intensity": None,
                    "is_core": True,
                    "is_diagnostic": False,
                    "negative_control": "random_labels",
                }
                plan.append(entry)
                run_id += 1
    
    # Random Groups: 仅OULAD，所有模型，seeds=5
    for model_info in models_config["OULAD"]:
        for seed in range(1, 6):  # seeds 1-5
            # Windows路径问题：使用 "N" 而不是 "N/A"
            variant_str = model_info['variant'] or 'N'
            entry = {
                "run_id": f"negative_control_random_groups_OULAD_{model_info['model']}_{variant_str}_seed{seed}",
                "dataset": "OULAD",
                "model": model_info["model"],
                "model_variant": model_info["variant"],
                "train_defense": "none",
                "publish_defense": None,
                "eps": None,
                "visibility": "full",
                "Q": 5,
                "seed": seed,
                "fairness_attribute": "gender",
                "coarsening_type": None,
                "coarsening_step": None,
                "noise_type": None,
                "noise_scale": None,
                "intensity": None,
                "is_core": True,
                "is_diagnostic": False,
                "negative_control": "random_groups",
            }
            plan.append(entry)
            run_id += 1
    
    return plan

def main():
    """主函数"""
    plan = generate_fast_plan()
    
    # 保存JSON
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "experiment_plan_fast.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    from collections import Counter
    core_count = sum(1 for e in plan if e["is_core"])
    diagnostic_count = sum(1 for e in plan if e["is_diagnostic"])
    
    # 按数据集统计
    dataset_stats = Counter(e["dataset"] for e in plan)
    model_stats = Counter((e["dataset"], e["model"], e.get("model_variant")) for e in plan)
    
    # 按防御统计
    defense_stats = Counter((e["dataset"], e["train_defense"], e.get("publish_defense") or "none", e.get("eps")) for e in plan)
    
    # 生成plan_stats_fast.json
    plan_stats = {
        "total_runs": len(plan),
        "core_runs": core_count,
        "diagnostic_runs": diagnostic_count,
        "target_range": "180-260",
        "compression_strategy": {
            "seeds_fixed": 5,
            "diagnostic_max": 6,
            "release_defenses_intensity_levels": 2,
            "dp_sgd_only_mlp": True,
            "lr_xgb_no_dp_sgd": True,
            "harvardx_uci_no_intersectional": True,
        },
        "dataset_coverage": dict(dataset_stats),
        "model_coverage": {f"{d}_{m}_{v or 'N/A'}": c for (d, m, v), c in sorted(model_stats.items())},
        "defense_coverage": {f"{d}_{td}_{pd or 'none'}_eps{eps or 'N/A'}": c for (d, td, pd, eps), c in sorted(defense_stats.items())[:30]},
    }
    
    stats_path = output_dir / "plan_stats_fast.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(plan_stats, f, indent=2, ensure_ascii=False)
    
    # 保存Markdown版本
    md_path = output_dir / "experiment_plan_fast.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Fast Experiment Plan (Reviewer-Proof, Compressed)\n\n")
        f.write(f"**Total runs: {len(plan)}** (target: ≤260)\n\n")
        f.write(f"- Core runs (seeds=5, main tables): {core_count}\n")
        f.write(f"- Diagnostic runs (seeds=2, appendix only): {diagnostic_count}\n")
        f.write(f"\n")
        
        f.write("## Compression Strategy\n\n")
        f.write("- Fixed seeds=5 for core runs only\n")
        f.write("- Diagnostic runs: ≤6 entries, marked as appendix\n")
        f.write("- Release defenses: 2 intensity levels (low/high) only\n")
        f.write("- DP-SGD ε∈{1,5,10} only on MLP-small + MLP-large\n")
        f.write("- LR/XGB: non-DP + release defenses only\n")
        f.write("- HarvardX/UCI697: no intersectional fairness settings\n")
        f.write("\n")
        
        f.write("## Dataset Coverage\n\n")
        for dataset, count in sorted(dataset_stats.items()):
            f.write(f"- {dataset}: {count} runs\n")
        f.write("\n")
        
        f.write("## Model Coverage\n\n")
        f.write("| Dataset | Model | Variant | Count |\n")
        f.write("|---------|-------|---------|-------|\n")
        for (dataset, model, variant), count in sorted(model_stats.items()):
            variant_str = variant if variant else "N/A"
            f.write(f"| {dataset} | {model} | {variant_str} | {count} |\n")
        f.write("\n")
        
        f.write("## Plan Summary (First 100 entries)\n\n")
        f.write("| Run ID | Dataset | Model | Train Defense | Publish Defense | ε | Visibility | Seeds | Fairness | Core |\n")
        f.write("|--------|---------|-------|---------------|-----------------|---|------------|-------|----------|------|\n")
        
        for entry in plan[:100]:
            pub_def = entry.get("publish_defense") or "none"
            eps_str = str(entry.get("eps")) if entry.get("eps") else "N/A"
            seeds_str = "5" if entry["is_core"] else "2"
            core_str = "✓" if entry["is_core"] else "✗"
            f.write(f"| {entry['run_id']} | {entry['dataset']} | {entry['model']} | {entry['train_defense']} | {pub_def} | {eps_str} | {entry['visibility']} | {seeds_str} | {entry['fairness_attribute']} | {core_str} |\n")
        
        if len(plan) > 100:
            f.write(f"\n... and {len(plan) - 100} more entries\n")
    
    print(f"Generated fast plan with {len(plan)} runs")
    print(f"  Core runs (seeds=5): {core_count}")
    print(f"  Diagnostic runs (seeds=2): {diagnostic_count}")
    print(f"Saved to {json_path}")
    print(f"Saved to {md_path}")
    print(f"Saved to {stats_path}")
    
    # 验证覆盖充分性
    if len(plan) > 260:
        print(f"WARNING: Plan has {len(plan)} runs, target is <=260")
    elif len(plan) < 180:
        print(f"WARNING: Plan has only {len(plan)} runs, target is 180-260")
    else:
        print(f"OK: Plan size within target range (<=260)")

if __name__ == "__main__":
    main()
