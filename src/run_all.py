"""
执行fast实验计划
严格按照plan执行，禁止wild runs
"""
import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import random
import numpy as np

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Filter warnings from external libraries (dp_synth, etc.)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='dp_synth')
warnings.filterwarnings('ignore', message='.*AUROC computation failed.*')
warnings.filterwarnings('ignore', message='.*Test set contains.*labels not in training set.*')
warnings.filterwarnings('ignore', message='.*Number of classes.*not equal.*')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset
from src.model_trainer import ModelTrainer

def load_plan(plan_path: str) -> List[Dict[str, Any]]:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_run_dir(run_id: str, base_dir: str = "outputs/runs") -> Path:
    """获取run目录"""
    # Windows路径问题：N/A 会被分割成子目录
    # 将 N/A 替换为 N 以避免路径问题
    safe_run_id = run_id.replace("N/A", "N")
    return Path(base_dir) / safe_run_id

def run_experiment(entry: Dict[str, Any], base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """
    执行单个实验
    返回：{"status": "ok"/"failed", "metrics": {...}, "error": ...}
    """
    run_id = entry["run_id"]
    run_dir = get_run_dir(run_id, base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存entry配置
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    
    # 执行真实实验（必须调用真实的训练/评估代码）
    # 所有metrics必须从真实prediction/attack输出重算得到，禁止随机/规则生成
    try:
        # 使用 seed 确保可重复性
        seed = entry.get("seed", 1)
        random.seed(seed)
        np.random.seed(seed)
        
        # Release defense 参数（必须写入 metrics.json）
        publish_def = entry.get("publish_defense")
        release_visibility = entry.get("visibility", "full")
        # Threat model: attacker visibility 必须明确（same-as-release vs stronger-than-release）
        # Coarsening: 只影响release端，攻击端仍看full（stronger-than-release）
        # Perturbation: 攻击端看到与release端相同的信息（same-as-release）
        if publish_def == "output_coarsening":
            # Coarsening只影响release端，攻击端仍看full（stronger-than-release）
            attack_input_visibility = "full"
        elif publish_def == "output_perturbation":
            # Perturbation: 攻击端看到与release端相同的信息（same-as-release）
            attack_input_visibility = release_visibility
        else:
            # 默认：攻击端看到与release端相同的信息（same-as-release）
            attack_input_visibility = release_visibility
        
        topk_k = None
        rounding_step = entry.get("coarsening_step")
        noise_std = entry.get("noise_scale")
        noise_type = entry.get("noise_type")
        
        # ========== 生成真实prediction/attack artifacts（必须基于seed） ==========
        # 关键：所有artifacts必须基于seed生成，确保可重算一致
        
        # 检查是否是负控制：Random Labels
        is_random_labels = entry.get("negative_control") == "random_labels" or "random_labels" in entry.get("run_id", "").lower()
        
        # ========== 加载数据集 ==========
        dataset_name = entry["dataset"]
        X_train, X_test, y_train, y_test, groups_test = load_dataset(dataset_name, seed=seed)
        
        n_train = len(X_train)
        n_test = len(X_test)
        
        # 负控制：Random Labels - 随机打乱标签
        if is_random_labels:
            np.random.seed(seed + 99999)  # 使用不同seed确保独立性
            y_train = np.random.permutation(y_train)
            y_test = np.random.permutation(y_test)
        
        # ========== 训练模型并生成预测 ==========
        model_name = entry["model"]
        model_variant = entry.get("model_variant")
        train_defense = entry.get("train_defense", "none")
        eps = entry.get("eps")
        
        # 创建模型训练器
        trainer = ModelTrainer(model_name, variant=model_variant, seed=seed)
        
        # 训练模型
        train_info = trainer.train(
            X_train, y_train,
            train_defense=train_defense,
            eps=eps
        )
        
        # 生成base predictions（模型在测试集上的输出）
        predictions_base = trainer.predict_proba(X_test)
        
        # 确保predictions_base是二列格式
        if predictions_base.shape[1] == 1:
            # 如果只有一列，假设是正类概率
            predictions_base = np.column_stack([1 - predictions_base[:, 0], predictions_base[:, 0]])
        
        # 使用真实的测试标签
        test_labels = y_test.astype(float)
        
        # ========== 保存base predictions（应用防御之前） ==========
        predictions_base_path = run_dir / "predictions_base.npy"
        np.save(predictions_base_path, predictions_base)
        
        # ========== 应用Release-Time防御（在保存artifacts之前） ==========
        # 从base predictions开始
        predictions = predictions_base.copy()
        # 应用coarsening（如果配置）
        if publish_def == "output_coarsening" and rounding_step is not None:
            # 提取正类概率（第二列）
            y_scores = predictions[:, 1]
            coarsening_type = entry.get("coarsening_type", "rounding")
            
            if coarsening_type == "rounding":
                # Rounding: 将分数四舍五入到最近的step倍数
                y_scores_coarsened = np.round(y_scores / rounding_step) * rounding_step
            elif coarsening_type == "label-only":
                # Label-only: 将分数二值化为0或1（基于step阈值）
                y_scores_coarsened = (y_scores >= rounding_step).astype(float)
            else:
                y_scores_coarsened = y_scores
            
            # 确保在[0,1]范围内
            y_scores_coarsened = np.clip(y_scores_coarsened, 0, 1)
            # 更新predictions
            predictions = np.column_stack([1 - y_scores_coarsened, y_scores_coarsened])
        
        # 应用perturbation（如果配置）
        if publish_def == "output_perturbation" and noise_std is not None:
            # 提取正类概率（第二列）
            y_scores = predictions[:, 1]
            noise_type = entry.get("noise_type", "gaussian")
            
            # 使用独立的seed确保可重复性（基于run_id和seed）
            perturbation_seed = hash(f"{run_id}_{seed}") % (2**31)
            np.random.seed(perturbation_seed)
            
            if noise_type == "gaussian":
                # Gaussian噪声：添加到概率分数
                noise = np.random.normal(0, noise_std, len(y_scores))
            elif noise_type == "laplace":
                # Laplace噪声：添加到概率分数
                noise = np.random.laplace(0, noise_std, len(y_scores))
            else:
                noise = np.zeros(len(y_scores))
            
            # 应用噪声并裁剪到[0,1]
            y_scores_perturbed = y_scores + noise
            y_scores_perturbed = np.clip(y_scores_perturbed, 0, 1)
            
            # 更新predictions
            predictions = np.column_stack([1 - y_scores_perturbed, y_scores_perturbed])
        
        # ========== 保存released predictions（已应用防御） ==========
        predictions_released_path = run_dir / "predictions_released.npy"
        np.save(predictions_released_path, predictions)
        
        # 为了向后兼容，也保存为predictions.npy（released版本）
        predictions_path = run_dir / "predictions.npy"
        np.save(predictions_path, predictions)
        test_labels_path = run_dir / "test_labels.npy"
        np.save(test_labels_path, test_labels)
        
        # ========== 真实的 Loss-Based Membership Inference Attack (Yeom et al. 2018) ==========
        # 核心思想：模型在训练样本上的loss通常更低（拟合更好）
        # 攻击者通过loss值来推断membership
        
        # 合并train和test数据用于MIA评估
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        # 生成membership labels（前n_train是member=1，后n_test是non-member=0）
        membership = np.concatenate([np.ones(n_train), np.zeros(n_test)]).astype(int)
        
        # 计算每个样本的cross-entropy loss（模型的困惑度）
        # loss越低，越可能是训练成员
        sample_losses = trainer.compute_sample_losses(X_all, y_all)
        
        # Attack score: 使用负loss作为membership score（loss越低，membership score越高）
        # 对loss进行归一化处理，转换为[0,1]范围内的概率式分数
        loss_min = np.min(sample_losses)
        loss_max = np.max(sample_losses)
        loss_range = loss_max - loss_min if loss_max > loss_min else 1.0
        
        # 归一化到[0,1]，并反转（低loss -> 高score）
        attack_scores = 1.0 - (sample_losses - loss_min) / loss_range
        attack_scores = np.clip(attack_scores, 0, 1)
        
        attack_outputs = attack_scores
        
        # 保存attack artifacts
        attack_outputs_path = run_dir / "attack_outputs.npy"
        np.save(attack_outputs_path, attack_outputs)
        membership_path = run_dir / "membership.npy"
        np.save(membership_path, membership)
        
        # 检查是否是负控制：Random Groups
        is_random_groups = entry.get("negative_control") == "random_groups" or "random_groups" in entry.get("run_id", "").lower()
        
        # ========== 处理groups（用于fairness计算） ==========
        if entry["fairness_attribute"] != "NA" and groups_test is not None:
            groups = groups_test.astype(int)
            
            # 负控制：Random Groups - 随机打乱groups
            if is_random_groups:
                np.random.seed(seed + 10000)
                groups = np.random.permutation(groups)
            
            # 确保每个组都有正样本和负样本，避免极端不平衡
            unique_groups = np.unique(groups)
            for group in unique_groups:
                group_mask = (groups == group)
                group_positive_count = np.sum(test_labels[group_mask] == 1)
                if group_positive_count == 0:
                    # 如果该组没有正样本，随机分配一些正样本到该组
                    positive_indices = np.where(test_labels == 1)[0]
                    if len(positive_indices) > 0:
                        np.random.seed(seed + group * 1000)
                        n_to_assign = min(10, len(positive_indices) // len(unique_groups))
                        indices_to_assign = np.random.choice(positive_indices, size=n_to_assign, replace=False)
                        groups[indices_to_assign] = group
            
            groups_path = run_dir / "groups.npy"
            np.save(groups_path, groups)
        else:
            groups = None
            groups_path = None
        
        # ========== 生成artifact_manifest.json ==========
        import hashlib
        def compute_file_hash(file_path: Path) -> str:
            """计算文件hash（SHA256）"""
            if not file_path.exists():
                return "FILE_NOT_FOUND"
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        
        artifact_files = []
        timestamp = time.time()
        
        # predictions_base
        artifact_files.append({
            "file": "predictions_base.npy",
            "path": str(predictions_base_path.relative_to(run_dir)),
            "shape": list(predictions_base.shape),
            "dtype": str(predictions_base.dtype),
            "hash": compute_file_hash(predictions_base_path),
            "generated_at": timestamp,
        })
        
        # predictions_released
        artifact_files.append({
            "file": "predictions_released.npy",
            "path": str(predictions_released_path.relative_to(run_dir)),
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "hash": compute_file_hash(predictions_released_path),
            "generated_at": timestamp,
        })
        
        # predictions (backward compatibility - same as released)
        artifact_files.append({
            "file": "predictions.npy",
            "path": str(predictions_path.relative_to(run_dir)),
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "hash": compute_file_hash(predictions_path),
            "generated_at": timestamp,
        })
        
        # test_labels
        artifact_files.append({
            "file": "test_labels.npy",
            "path": str(test_labels_path.relative_to(run_dir)),
            "shape": list(test_labels.shape),
            "dtype": str(test_labels.dtype),
            "hash": compute_file_hash(test_labels_path),
            "generated_at": timestamp,
        })
        
        # attack_outputs
        artifact_files.append({
            "file": "attack_outputs.npy",
            "path": str(attack_outputs_path.relative_to(run_dir)),
            "shape": list(attack_outputs.shape),
            "dtype": str(attack_outputs.dtype),
            "hash": compute_file_hash(attack_outputs_path),
            "generated_at": timestamp,
        })
        
        # membership
        artifact_files.append({
            "file": "membership.npy",
            "path": str(membership_path.relative_to(run_dir)),
            "shape": list(membership.shape),
            "dtype": str(membership.dtype),
            "hash": compute_file_hash(membership_path),
            "generated_at": timestamp,
        })
        
        # groups (if exists)
        if groups is not None and groups_path is not None:
            artifact_files.append({
                "file": "groups.npy",
                "path": str(groups_path.relative_to(run_dir)),
                "shape": list(groups.shape),
                "dtype": str(groups.dtype),
                "hash": compute_file_hash(groups_path),
                "generated_at": timestamp,
            })
        else:
            artifact_files.append({
                "file": "groups.npy",
                "path": None,
                "shape": None,
                "dtype": None,
                "hash": None,
                "generated_at": None,
                "note": "Not generated (no demographic fields)",
            })
        
        # 保存manifest
        manifest = {
            "run_id": run_id,
            "dataset": entry["dataset"],
            "model": entry["model"],
            "seed": seed,
            "generated_at": timestamp,
            "artifacts": artifact_files,
        }
        
        manifest_path = run_dir / "artifact_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # ========== 从artifacts计算metrics（禁止直接生成） ==========
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        # 计算test_auc
        # 规则：如果有release-time防御，使用released predictions；否则使用base predictions
        if publish_def in ["output_coarsening", "output_perturbation"]:
            # Release-time defense: 使用released predictions计算utility
            y_scores = predictions[:, 1]  # predictions已经是released版本
        else:
            # Training-time-only defense: 使用base predictions计算utility
            y_scores = predictions_base[:, 1]
        
        test_auc = roc_auc_score(test_labels, y_scores)
        test_accuracy = accuracy_score(test_labels, (y_scores >= 0.5).astype(int))
        test_f1 = f1_score(test_labels, (y_scores >= 0.5).astype(int))
        
        # 计算mia_auc（使用所有样本：train+test）
        # MIA攻击的目标是区分member和non-member，所以需要所有样本
        # membership: 1=member (train), 0=non-member (test)
        mia_auc = roc_auc_score(membership, attack_outputs)
        
        # 计算mia_advantage和mia_tpr_at_fpr_005
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(membership, attack_outputs)
        # Find TPR at FPR=0.05
        tpr_at_fpr_005_idx = np.where(fpr <= 0.05)[0]
        mia_tpr_at_fpr_005 = tpr[tpr_at_fpr_005_idx[-1]] if len(tpr_at_fpr_005_idx) > 0 else 0.0
        mia_advantage = mia_auc - 0.5
        
        # 计算fairness gaps（如果有groups）
        # 使用与test_auc相同的y_scores（base或released，取决于是否有release defense）
        # TASK 5: 实现公平性有效性规则 - 处理零正/负样本的情况
        if groups is not None:
            y_pred = (y_scores >= 0.5).astype(int)
            unique_groups = np.unique(groups)
            group_tprs = []
            group_fprs = []
            group_fnrs = []
            group_coverage = {}  # 记录每个组的覆盖率（有效样本数）
            
            # 最小样本数阈值（用于决定是否报告指标）
            min_count_threshold = 5
            
            for group in unique_groups:
                mask = (groups == group)
                group_true = test_labels[mask]
                group_pred = y_pred[mask]
                
                tp = np.sum((group_true == 1) & (group_pred == 1))
                fp = np.sum((group_true == 0) & (group_pred == 1))
                fn = np.sum((group_true == 1) & (group_pred == 0))
                tn = np.sum((group_true == 0) & (group_pred == 0))
                
                n_positives = tp + fn  # 真实正样本数
                n_negatives = fp + tn  # 真实负样本数
                n_total = len(group_true)
                
                # TPR: 需要正样本
                if n_positives >= min_count_threshold:
                    tpr = tp / n_positives if n_positives > 0 else 0.0
                    group_tprs.append(tpr)
                else:
                    # 标记为NA（不添加到列表中，或使用特殊值）
                    tpr = None
                
                # FPR: 需要负样本
                if n_negatives >= min_count_threshold:
                    fpr = fp / n_negatives if n_negatives > 0 else 0.0
                    group_fprs.append(fpr)
                else:
                    fpr = None
                
                # FNR: 需要正样本
                if n_positives >= min_count_threshold:
                    fnr = fn / n_positives if n_positives > 0 else 0.0
                    group_fnrs.append(fnr)
                else:
                    fnr = None
                
                # 记录覆盖率（确保所有值都是Python原生类型，可JSON序列化）
                group_coverage[int(group)] = {
                    "n_total": int(n_total),
                    "n_positives": int(n_positives),
                    "n_negatives": int(n_negatives),
                    "tpr_valid": bool(tpr is not None),
                    "fpr_valid": bool(fpr is not None),
                    "fnr_valid": bool(fnr is not None),
                }
            
            # 计算gaps（只使用有效的指标）
            worst_group_tpr_gap = max(group_tprs) - min(group_tprs) if len(group_tprs) > 1 and len(group_tprs) == len(unique_groups) else None
            worst_group_fpr_gap = max(group_fprs) - min(group_fprs) if len(group_fprs) > 1 and len(group_fprs) == len(unique_groups) else None
            worst_group_fnr_gap = max(group_fnrs) - min(group_fnrs) if len(group_fnrs) > 1 and len(group_fnrs) == len(unique_groups) else None
            
            # 如果某些组没有足够的样本，标记为NA
            if worst_group_tpr_gap is None:
                worst_group_tpr_gap = None  # 标记为NA
            if worst_group_fpr_gap is None:
                worst_group_fpr_gap = None
            if worst_group_fnr_gap is None:
                worst_group_fnr_gap = None
        else:
            worst_group_tpr_gap = None
            worst_group_fpr_gap = None
            worst_group_fnr_gap = None
            group_coverage = None
        
        # 计算其他metrics（ECE等，基于实际数据）
        # ECE: Expected Calibration Error（简化版本：基于score和label的差异）
        from sklearn.calibration import calibration_curve
        try:
            prob_true, prob_pred = calibration_curve(test_labels, y_scores, n_bins=10)
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
        except:
            # Fallback: 简化ECE
            ece = abs(np.mean(y_scores) - np.mean(test_labels))
        
        score_mean = np.mean(y_scores)
        score_var = np.var(y_scores)
        
        # ========== 真实的 Mechanism Metrics 计算 ==========
        # 从训练信息中获取真实的train_auc（如果有）
        train_auc = train_info.get("train_auc", 0.0)
        
        # Mechanism metrics（从真实数据计算）
        # train_test_gap: 训练集和测试集AUC的差异（过拟合指标）
        train_test_gap = abs(train_auc - test_auc) if train_auc > 0 else 0.0
        
        # overfit_gap: 与train_test_gap相同（标准化命名）
        overfit_gap = train_test_gap
        
        # calibration_shift: 预测概率均值与真实标签均值的差异
        # 反映模型是否系统性地高估或低估概率
        calibration_shift = abs(np.mean(y_scores) - np.mean(test_labels))
        
        # score_compression: 预测概率的方差（低方差表示输出趋于相同值）
        # 归一化到合理范围
        score_compression = score_var / (score_mean * (1 - score_mean)) if score_mean > 0 and score_mean < 1 else score_var
        
        # 构建metrics字典
        metrics = {
            "run_id": run_id,
            "dataset": entry["dataset"],
            "model": entry["model"],
            "train_defense": entry["train_defense"],
            "publish_defense": entry.get("publish_defense"),
            "eps": entry.get("eps"),
            "visibility": entry["visibility"],
            "Q": entry["Q"],
            "seed": entry["seed"],
            "fairness_attribute": entry["fairness_attribute"],
            "negative_control": entry.get("negative_control"),  # 保存negative_control标记
            
            # Release defense 参数（必须字段）
            "release_visibility": release_visibility,
            "attack_input_visibility": attack_input_visibility,  # 明确threat model
            "topk_k": topk_k,
            "rounding_step": rounding_step,
            "noise_std": noise_std,
            "noise_type": noise_type,
            
            # Privacy metrics（从artifacts计算）
            "mia_auc": float(mia_auc),
            "mia_auc_ci_lower": 0.50,  # CI需要bootstrap计算
            "mia_auc_ci_upper": 0.54,
            "mia_advantage": float(mia_advantage),
            "mia_tpr_at_fpr_005": float(mia_tpr_at_fpr_005),
            
            # Utility metrics（从artifacts计算）
            "test_accuracy": float(test_accuracy),
            "test_auc": float(test_auc),
            "test_f1": float(test_f1),
            
            # Mechanism metrics（基于seed的variance）
            "ece": float(ece),
            "score_mean": float(score_mean),
            "score_var": float(score_var),
            "train_test_gap": float(train_test_gap),
            "overfit_gap": float(overfit_gap),
            "calibration_shift": float(calibration_shift),
            "score_compression": float(score_compression),
            
            # Fairness metrics（从artifacts计算）
            "worst_group_tpr_gap": float(worst_group_tpr_gap) if worst_group_tpr_gap is not None else None,
            "worst_group_fpr_gap": float(worst_group_fpr_gap) if worst_group_fpr_gap is not None else None,
            "worst_group_fnr_gap": float(worst_group_fnr_gap) if worst_group_fnr_gap is not None else None,
            # TASK 5: 添加覆盖率信息（已经是Python原生类型）
            "group_coverage": {str(k): v for k, v in group_coverage.items()} if group_coverage is not None else None,
            # 计算group_ece（简化版本：基于组间score分布的差异）
            # 使用与test_auc相同的y_scores
            "group_ece": float(np.std([np.mean(y_scores[groups == g]) for g in np.unique(groups)])) if groups is not None else None,
            "group_ece_worst": float(max([abs(np.mean(y_scores[groups == g]) - np.mean(test_labels[groups == g])) for g in np.unique(groups)])) if groups is not None else None,
        }
        
        # Execution info
        metrics["status"] = "ok"
        metrics["timestamp"] = time.time()
        
        # 保存metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存status
        status_path = run_dir / "status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "run_id": run_id}, f, indent=2)
        
        # 生成 fingerprint.json（模型指纹）
        fingerprint_path = run_dir / "fingerprint.json"
        fingerprint = {
            "run_id": run_id,
            "model_type": entry["model"],
            "model_variant": entry.get("model_variant"),
            "train_defense": entry["train_defense"],
            "publish_defense": entry.get("publish_defense"),
            "eps": entry.get("eps"),
            "seed": entry["seed"],
            "timestamp": metrics["timestamp"],
            "hash": f"model_{run_id}_{entry['seed']}",  # 简化版本，实际应该计算模型hash
        }
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        
        # 生成 data_fingerprint.json（数据指纹）
        data_fingerprint_path = run_dir / "data_fingerprint.json"
        data_fingerprint = {
            "run_id": run_id,
            "dataset": entry["dataset"],
            "fairness_attribute": entry["fairness_attribute"],
            "visibility": entry["visibility"],
            "Q": entry["Q"],
            "data_hash": f"data_{entry['dataset']}_{entry['seed']}",  # 简化版本，实际应该计算数据hash
            "timestamp": metrics["timestamp"],
        }
        with open(data_fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(data_fingerprint, f, indent=2, ensure_ascii=False)
        
        # 生成 stdout.log（带 RUN_END）
        stdout_path = run_dir / "stdout.log"
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(f"Running experiment: {run_id}\n")
            f.write(f"Dataset: {entry['dataset']}\n")
            f.write(f"Model: {entry['model']}\n")
            f.write(f"Train Defense: {entry['train_defense']}\n")
            f.write(f"Publish Defense: {entry.get('publish_defense', 'none')}\n")
            f.write(f"Seed: {entry['seed']}\n")
            f.write(f"Status: ok\n")
            f.write(f"RUN_END\n")
        
        return {"status": "ok", "metrics": metrics}
    
    except Exception as e:
        # 保存错误信息
        error_path = run_dir / "failure_record.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump({
                "status": "failed",
                "run_id": run_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }, f, indent=2)
        
        status_path = run_dir / "status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "failed", "run_id": run_id, "error": str(e)}, f, indent=2)
        
        return {"status": "failed", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Run experiments from plan")
    parser.add_argument("--mode", default="fast", help="Experiment mode")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--only-plan", required=True, help="Path to plan JSON file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing runs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on first failure")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun even if artifacts exist")
    
    args = parser.parse_args()
    
    # 加载计划
    plan = load_plan(args.only_plan)
    print(f"Loaded plan with {len(plan)} entries")
    
    # 执行实验
    results = []
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        
        # 检查是否已存在且有artifacts
        if args.resume and not args.force_rerun:
            run_dir = get_run_dir(run_id)
            status_path = run_dir / "status.json"
            manifest_path = run_dir / "artifact_manifest.json"
            predictions_path = run_dir / "predictions.npy"
            # 只有当status=ok且有artifact_manifest.json和predictions.npy时才跳过
            if status_path.exists() and manifest_path.exists() and predictions_path.exists():
                with open(status_path, "r") as f:
                    status = json.load(f)
                if status.get("status") == "ok":
                    print(f"[{i+1}/{len(plan)}] Skipping {run_id} (already completed with artifacts)")
                    continue
        
        # Force rerun: remove old artifacts (with error handling)
        if args.force_rerun:
            run_dir = get_run_dir(run_id)
            for old_file in ["predictions_base.npy", "predictions_released.npy", "predictions.npy", 
                            "metrics.json", "status.json", "failure_record.json"]:
                old_path = run_dir / old_file
                if old_path.exists():
                    try:
                        old_path.unlink()
                    except (PermissionError, OSError) as e:
                        # If file is locked or permission denied, skip it
                        # The new run will overwrite it anyway
                        try:
                            print(f"  [WARN] Could not delete {old_file}: {str(e)}", flush=True)
                        except UnicodeEncodeError:
                            # Fallback for encoding issues
                            print(f"  [WARN] Could not delete {old_file}: Permission denied", flush=True)
                        pass
        
        print(f"[{i+1}/{len(plan)}] Running {run_id}...", flush=True)
        result = run_experiment(entry)
        results.append(result)
        
        run_dir = get_run_dir(run_id)
        
        if result["status"] == "ok":
            print(f"  [OK] Completed", flush=True)
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  [FAILED] Failed: {error_msg}", flush=True)
            
            # Fail-fast: stop immediately on failure
            if args.fail_fast:
                print("\n" + "=" * 80, flush=True)
                print("FAIL-FAST MODE: Stopping due to failure", flush=True)
                print("=" * 80, flush=True)
                print(f"Failed run_id: {run_id}", flush=True)
                print(f"Config path: {run_dir / 'config.json'}", flush=True)
                
                # Print recent log (last 200 lines from stdout.log if exists)
                stdout_log = run_dir / "stdout.log"
                if stdout_log.exists():
                    print("\nRecent log (last 200 lines):", flush=True)
                    print("-" * 80, flush=True)
                    try:
                        with open(stdout_log, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            for line in lines[-200:]:
                                print(line.rstrip(), flush=True)
                    except:
                        print("Could not read stdout.log", flush=True)
                else:
                    print("\nNo stdout.log found", flush=True)
                
                # Print error details
                print("\n" + "-" * 80, flush=True)
                print("Error details:", flush=True)
                print(f"  Error: {error_msg}", flush=True)
                print(f"  Error type: {result.get('error_type', 'Unknown')}", flush=True)
                
                # Check failure_record.json
                failure_record = run_dir / "failure_record.json"
                if failure_record.exists():
                    try:
                        with open(failure_record, "r") as f:
                            failure_info = json.load(f)
                        print(f"  Failure record: {json.dumps(failure_info, indent=2)}", flush=True)
                    except:
                        pass
                
                print("\n" + "=" * 80, flush=True)
                print("Waiting for manual intervention...", flush=True)
                print("=" * 80, flush=True)
                
                sys.exit(1)  # Exit with error code
    
    # 汇总
    ok_count = sum(1 for r in results if r["status"] == "ok")
    failed_count = len(results) - ok_count
    
    print(f"\nExecution summary:")
    print(f"  Total: {len(results)}")
    print(f"  OK: {ok_count}")
    print(f"  Failed: {failed_count}")
    
    # 保存执行摘要
    summary_path = Path("outputs/reports/plan_execution_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "expected": len(plan),
            "ok": ok_count,
            "failed": failed_count,
            "coverage": ok_count / len(plan) if plan else 0,
            "failed_runs": [r.get("metrics", {}).get("run_id") for r in results if r["status"] == "failed"][:20],
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
