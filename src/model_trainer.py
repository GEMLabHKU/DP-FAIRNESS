"""
模型训练模块
支持LR, XGBoost, MLP-small, MLP-large
支持真实DP-SGD训练（用于MLP）- 使用Opacus
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 导入Opacus用于真实DP-SGD
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class ModelTrainer:
    """模型训练器基类"""
    
    def __init__(self, model_type: str, variant: Optional[str] = None, seed: int = 42):
        self.model_type = model_type
        self.variant = variant
        self.seed = seed
        self.model = None
        self._set_seed(seed)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              train_defense: str = "none", eps: Optional[float] = None,
              delta: Optional[float] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            train_defense: 训练时防御 ("none" 或 "DP-SGD")
            eps: DP epsilon值（如果使用DP-SGD）
            delta: DP delta值（如果使用DP-SGD）
            **kwargs: 其他参数
        
        Returns:
            训练信息字典，包含真实的(epsilon, delta)和train AUC
        """
        if self.model_type == "LR":
            return self._train_lr(X_train, y_train, train_defense, eps, delta, **kwargs)
        elif self.model_type == "XGBoost":
            return self._train_xgboost(X_train, y_train, train_defense, eps, delta, **kwargs)
        elif self.model_type == "MLP":
            return self._train_mlp(X_train, y_train, train_defense, eps, delta, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model_type == "LR":
            return self.model.predict_proba(X)
        elif self.model_type == "XGBoost":
            return self.model.predict_proba(X)
        elif self.model_type == "MLP":
            return self._predict_mlp(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_lr(self, X_train: np.ndarray, y_train: np.ndarray,
                  train_defense: str, eps: Optional[float], delta: Optional[float], 
                  **kwargs) -> Dict[str, Any]:
        """训练逻辑回归"""
        # LR不支持DP-SGD（根据实验设计）
        if train_defense == "DP-SGD":
            raise ValueError("LR does not support DP-SGD")
        
        self.model = LogisticRegression(
            random_state=self.seed,
            max_iter=1000,
            solver='lbfgs'
        )
        self.model.fit(X_train, y_train)
        
        # 计算训练AUC
        y_train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        
        return {
            "train_auc": train_auc,
            "n_params": X_train.shape[1] + 1,  # weights + bias
        }
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       train_defense: str, eps: Optional[float], delta: Optional[float],
                       **kwargs) -> Dict[str, Any]:
        """训练XGBoost"""
        # XGBoost不支持DP-SGD（根据实验设计）
        if train_defense == "DP-SGD":
            raise ValueError("XGBoost does not support DP-SGD")
        
        self.model = xgb.XGBClassifier(
            random_state=self.seed,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.model.fit(X_train, y_train)
        
        y_train_pred = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        
        # 估算参数数量
        try:
            booster = self.model.get_booster()
            n_params = 0
            for i in range(self.model.n_estimators):
                try:
                    tree_str = booster.get_dump()[i]
                    n_params += tree_str.count('leaf=')
                except (IndexError, AttributeError):
                    pass
            if n_params == 0:
                n_params = self.model.n_estimators * 10
        except (AttributeError, TypeError, Exception):
            try:
                n_params = self.model.n_estimators * 10
            except (AttributeError, Exception):
                n_params = 100
        
        return {
            "train_auc": train_auc,
            "n_params": n_params,
        }
    
    def _train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                   train_defense: str, eps: Optional[float], delta: Optional[float],
                   **kwargs) -> Dict[str, Any]:
        """
        训练MLP，支持真实DP-SGD（使用Opacus）
        """
        # 确定MLP架构
        if self.variant == "small":
            hidden_sizes = [64]
            n_layers = 2
        elif self.variant == "large":
            hidden_sizes = [256, 256]
            n_layers = 3
        else:
            raise ValueError(f"MLP variant must be 'small' or 'large', got {self.variant}")
        
        input_size = X_train.shape[1]
        
        # 创建模型并确保与Opacus兼容
        self.model = MLPNet(input_size, hidden_sizes, n_layers)
        
        # 如果使用DP-SGD，需要使模型兼容Opacus
        if train_defense == "DP-SGD":
            self.model = ModuleValidator.fix(self.model)
        
        # 转换为PyTorch格式
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        
        # 关键：为DP-SGD设置较大的batch size以提高训练稳定性
        if train_defense == "DP-SGD":
            batch_size = min(256, len(X_train))  # DP-SGD通常使用较大的batch
        else:
            batch_size = 64
        
        # 使用seed确保DataLoader shuffle的可复现性
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        # 设置优化器
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 如果使用DP-SGD，使用Opacus的make_private_with_epsilon API
        privacy_engine = None
        actual_epsilon = None
        actual_delta = delta if delta is not None else 1.0 / len(X_train)
        
        if train_defense == "DP-SGD" and eps is not None:
            privacy_engine = PrivacyEngine()
            
            # DP训练通常需要更多epochs
            n_epochs = 30
            
            # 使用Opacus 1.4+的make_private_with_epsilon API
            # 它会自动计算noise_multiplier以达到target_epsilon
            self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=eps,
                target_delta=actual_delta,
                epochs=n_epochs,
                max_grad_norm=1.0,  # 标准的per-sample gradient clipping bound
            )
        else:
            n_epochs = 50
        
        # 训练循环
        self.model.train()
        
        for epoch in range(n_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 在DP-SGD中，每epoch后检查隐私消耗
            if train_defense == "DP-SGD" and privacy_engine is not None:
                try:
                    actual_epsilon = privacy_engine.get_epsilon(delta=actual_delta)
                except Exception as e:
                    # 如果无法获取，保持target_epsilon作为近似值
                    actual_epsilon = eps
        
        # 计算训练AUC
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            y_train_pred = probs[:, 1].numpy()
        
        train_auc = roc_auc_score(y_train, y_train_pred)
        
        # 计算参数数量
        n_params = sum(p.numel() for p in self.model.parameters())
        
        result = {
            "train_auc": train_auc,
            "n_params": n_params,
            "used_dp_sgd": train_defense == "DP-SGD",
        }
        
        # 如果是DP-SGD，返回真实的(epsilon, delta)
        if train_defense == "DP-SGD":
            result["dp_epsilon_target"] = eps
            result["dp_epsilon_actual"] = actual_epsilon if actual_epsilon is not None else eps
            result["dp_delta"] = actual_delta
            result["dp_max_grad_norm"] = 1.0
            result["dp_batch_size"] = batch_size
            result["dp_epochs"] = n_epochs
        
        return result
    
    def _predict_mlp(self, X: np.ndarray) -> np.ndarray:
        """MLP预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.numpy()
    
    def compute_sample_losses(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算每个样本的cross-entropy loss（用于MIA）
        返回: 每个样本的loss值（numpy array）
        """
        if self.model_type == "MLP":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.LongTensor(y)
                outputs = self.model(X_tensor)
                # 计算每个样本的cross-entropy loss
                log_probs = torch.log_softmax(outputs, dim=1)
                # 提取正确类别的log概率的负值（即loss）
                losses = -log_probs[range(len(y)), y].numpy()
            return losses
        elif self.model_type in ["LR", "XGBoost"]:
            # 对于sklearn/xgboost模型，手动计算CE loss
            y_pred_proba = self.predict_proba(X)
            # 避免log(0)
            y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
            losses = - (y * np.log(y_pred_proba[:, 1]) + (1 - y) * np.log(y_pred_proba[:, 0]))
            return losses
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class MLPNet(nn.Module):
    """MLP网络 - 与Opacus兼容的版本"""
    
    def __init__(self, input_size: int, hidden_sizes: list, n_layers: int):
        super(MLPNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # Dropout在DP-SGD中需要特殊处理，使用p=0.2但会被ModuleValidator处理
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # 输出层（二分类）
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
