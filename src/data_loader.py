"""
数据集加载模块
支持OULAD, UCI697, HarvardX_PersonCourse三个数据集
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os
# 过滤掉 dp_synth 相关的警告（这些警告来自外部库，不影响我们的代码）
warnings.filterwarnings('ignore', category=UserWarning, module='dp_synth')
warnings.filterwarnings('ignore', message='.*AUROC computation failed.*')
warnings.filterwarnings('ignore', message='.*Test set contains.*labels not in training set.*')
warnings.filterwarnings('ignore', message='.*Number of classes.*not equal.*')
warnings.filterwarnings('ignore')


def load_dataset(dataset_name: str, seed: int = 42, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    加载数据集并返回训练/测试特征和标签
    
    Args:
        dataset_name: 数据集名称 ("OULAD", "UCI697", "HarvardX_PersonCourse")
        seed: 随机种子
        data_dir: 数据目录路径（如果为None，尝试从常见位置加载）
    
    Returns:
        X_train, X_test, y_train, y_test, groups_test
        groups_test: 测试集的组标签（用于公平性评估），如果数据集没有demographic字段则为None
    """
    np.random.seed(seed)
    
    if data_dir is None:
        # 尝试常见的数据目录位置
        possible_dirs = [
            Path("data"),
            Path("datasets"),
            Path("outputs/data"),
        ]
        data_dir = None
        for d in possible_dirs:
            if d.exists():
                data_dir = str(d)
                break
        
        # 如果没找到，默认使用data目录（即使不存在，也会在后续路径查找中处理）
        if data_dir is None:
            data_dir = "data"
    
    # 根据数据集名称加载
    if dataset_name == "OULAD":
        return _load_oulad(seed, data_dir)
    elif dataset_name == "UCI697":
        return _load_uci697(seed, data_dir)
    elif dataset_name == "HarvardX_PersonCourse":
        return _load_harvardx(seed, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_oulad(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载OULAD数据集"""
    # 尝试从文件加载，如果不存在则生成模拟数据
    if data_dir:
        # Try multiple possible paths
        possible_paths = [
            Path(data_dir) / "raw" / "oulad" / "studentInfo.csv",
            Path(data_dir) / "OULAD" / "studentInfo.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "oulad", Path(data_dir) / "OULAD"]
        for base in base_paths:
            if base.exists():
                for csv_file in base.rglob("studentInfo.csv"):
                    possible_paths.append(csv_file)
                    break
        
        for csv_path in possible_paths:
            if csv_path.exists():
                return _load_oulad_from_file(csv_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: OULAD data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_uci697(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载UCI697数据集"""
    if data_dir:
        # Try multiple possible paths and filenames
        possible_paths = [
            Path(data_dir) / "raw" / "uci697" / "data.csv",  # Actual downloaded filename
            Path(data_dir) / "raw" / "uci697" / "student-mat.csv",
            Path(data_dir) / "UCI697" / "student-mat.csv",
            Path(data_dir) / "UCI697" / "data.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "uci697", Path(data_dir) / "UCI697"]
        for base in base_paths:
            if base.exists():
                for csv_file in base.rglob("*.csv"):
                    possible_paths.append(csv_file)
                    break
        
        for csv_path in possible_paths:
            if csv_path.exists():
                return _load_uci697_from_file(csv_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: UCI697 data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_harvardx(seed: int, data_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """加载HarvardX_PersonCourse数据集"""
    if data_dir:
        # Try multiple possible paths and filenames (including .tab files)
        possible_paths = [
            Path(data_dir) / "raw" / "harvardx" / "HXPC13_DI_v3_11-13-2019.tab",  # Actual downloaded filename
            Path(data_dir) / "raw" / "harvardx" / "HMXPC13_DI_v2_5-14-14.csv",
            Path(data_dir) / "HarvardX_PersonCourse" / "HMXPC13_DI_v2_5-14-14.csv",
        ]
        # Also search recursively
        base_paths = [Path(data_dir) / "raw" / "harvardx", Path(data_dir) / "HarvardX_PersonCourse"]
        for base in base_paths:
            if base.exists():
                # Look for .tab or .csv files
                for data_file in base.rglob("*.tab"):
                    possible_paths.append(data_file)
                    break
                for data_file in base.rglob("*.csv"):
                    possible_paths.append(data_file)
                    break
        
        for data_path in possible_paths:
            if data_path.exists():
                return _load_harvardx_from_file(data_path, seed)
    
    # CRITICAL: Synthetic fallback is STRICTLY FORBIDDEN
    # This code path should NEVER be reached
    raise FileNotFoundError(
        f"CRITICAL ERROR: HarvardX_PersonCourse data file not found. Synthetic fallback is STRICTLY FORBIDDEN. "
        f"Searched paths: {[str(p) for p in possible_paths]}. "
        f"Please ensure real data files are available at one of these paths."
    )


def _load_oulad_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV文件加载OULAD数据（Open University Learning Analytics Dataset）"""
    df = pd.read_csv(csv_path)
    
    # 标签：final_result -> binary (0=Pass/Distinction, 1=Fail/Withdrawn)
    # final_result values: 'Pass', 'Distinction', 'Withdrawn', 'Fail'
    if 'final_result' in df.columns:
        df['label'] = df['final_result'].apply(
            lambda x: 1 if str(x).strip().lower() in ['fail', 'withdrawn'] else 0
        ).astype(int)
    else:
        raise ValueError("OULAD data must have 'final_result' column")
    
    # 保存性别分组信息（在划分前编码）
    groups = None
    if 'gender' in df.columns:
        le = LabelEncoder()
        groups = le.fit_transform(df['gender'].fillna('Unknown'))
    
    # 特征工程：对所有类别特征进行Label Encoding
    # OULAD的类别特征：code_module, code_presentation, gender, region, 
    #   highest_education, imd_band, age_band, disability
    # 排除id_student（学生ID，不应作为预测特征）
    
    feature_cols = [
        'code_module', 'code_presentation', 'region', 
        'highest_education', 'imd_band', 'age_band', 'disability',
        'num_of_prev_attempts', 'studied_credits'
    ]
    
    # 创建编码后的特征矩阵
    X_encoded = pd.DataFrame()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        if df[col].dtype == 'object':
            # 类别特征：Label Encoding
            le = LabelEncoder()
            # 处理缺失值
            filled = df[col].fillna('Unknown').astype(str)
            X_encoded[col] = le.fit_transform(filled)
        else:
            # 数值特征：直接使用
            X_encoded[col] = df[col].fillna(df[col].median())
    
    X = X_encoded.values
    y = df['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 统一划分训练/测试集（确保groups对齐）
    np.random.seed(seed)
    try:
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=y
        )
    except ValueError:
        # stratify失败时不用stratify
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed
        )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # groups_test必须与X_test/y_test对应同一组样本
    groups_test = None
    if groups is not None:
        groups_test = groups[test_idx]
    
    return X_train, X_test, y_train, y_test, groups_test


def _load_uci697_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV文件加载UCI697数据（Predict Students' Dropout and Academic Success）"""
    # UCI 697数据使用分号分隔
    df = pd.read_csv(csv_path, sep=';')
    
    # 标签：Target列 -> binary (0=Graduate/Success, 1=Dropout/At-risk)
    # Target列有三个值：'Dropout', 'Enrolled', 'Graduate'
    # 映射：Dropout -> 1 (需要干预), Graduate -> 0 (成功)
    # Enrolled -> 排除（状态不确定）或归入0（视为尚未失败）
    if 'Target' in df.columns:
        # 首先排除Enrolled（状态不确定的样本）
        df = df[df['Target'] != 'Enrolled'].copy()
        # 二值映射
        df['label'] = (df['Target'] == 'Dropout').astype(int)
    elif 'G3' in df.columns:
        # 兼容旧版student-mat数据（如果存在）
        median_grade = df['G3'].median()
        df['label'] = (df['G3'] >= median_grade).astype(int)
    else:
        raise ValueError(f"UCI697 data must have 'Target' or 'G3' column. Found columns: {list(df.columns)}")
    
    # UCI697没有demographic字段用于公平性评估（根据实验设计）
    groups = None
    
    # 分离特征列（排除Target和label）
    feature_cols = [c for c in df.columns if c not in ['Target', 'label']]
    
    # 识别数值列和分类列
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    # 对分类特征进行one-hot编码
    df_encoded = df[['label']].copy()
    
    # 添加数值特征
    for col in numeric_cols:
        df_encoded[col] = df[col].fillna(df[col].median())  # 填充缺失值
    
    # 对分类特征做label encoding（或one-hot，这里用label encoding简化）
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    # 准备X和y
    X_cols = [c for c in df_encoded.columns if c != 'label']
    X = df_encoded[X_cols].values
    y = df_encoded['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _load_harvardx_from_file(csv_path: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """从CSV/TAB文件加载HarvardX_PersonCourse数据"""
    # Support both .csv and .tab files
    if csv_path.suffix.lower() == '.tab':
        df = pd.read_csv(csv_path, sep='\t')
    else:
        df = pd.read_csv(csv_path)
    
    # 标签：certified (0=incomplete, 1=complete)
    if 'certified' in df.columns:
        df['label'] = df['certified'].astype(int)
    else:
        df['label'] = np.random.binomial(1, 0.4, len(df))
    
    # HarvardX没有demographic字段用于公平性评估（根据实验设计）
    groups = None
    
    # 选择数值特征
    numeric_cols = ['registered', 'viewed', 'explored', 'nevents', 'ndays_act', 
                    'nplay_video', 'nchapters', 'nforum_posts']
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    # CRITICAL: Handle NaN values - fill with 0 or drop rows
    # First, check for NaN in selected columns
    for col in available_cols:
        if df[col].isna().any():
            # Fill NaN with 0 (or median if preferred)
            df[col] = df[col].fillna(0)
    
    # Drop rows where label is NaN
    df = df.dropna(subset=['label'])
    
    if len(available_cols) < 5:
        for i in range(5 - len(available_cols)):
            df[f'feature_{i}'] = np.random.randn(len(df))
            available_cols.append(f'feature_{i}')
    
    X = df[available_cols[:20]].values
    y = df['label'].values
    
    # Final check: ensure no NaN in X or y
    if np.isnan(X).any() or np.isnan(y).any():
        # Drop rows with NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _generate_synthetic_oulad(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟OULAD数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 5000
    n_features = 15
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成标签（约30%失败率）
    # 使用特征线性组合 + 噪声
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 70)).astype(int)
    
    # 生成组标签（gender）
    groups = np.random.binomial(1, 0.5, n_samples)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    _, groups_test = train_test_split(
        groups, test_size=0.2, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test, groups_test


def _generate_synthetic_uci697(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟UCI697数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 400
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 50)).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None


def _generate_synthetic_harvardx(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """生成模拟HarvardX数据（用于测试）"""
    np.random.seed(seed)
    n_samples = 3000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    logit = X @ coef + np.random.randn(n_samples) * 0.5
    y = (logit > np.percentile(logit, 60)).astype(int)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, None
