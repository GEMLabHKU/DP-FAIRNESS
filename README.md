# Evaluating Private Training in Educational Prediction

[![Paper](https://img.shields.io/badge/Paper-EDM%202026-blue)](https://github.com/yourusername/dp-fairness-edm)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-OULAD%2FUCI697-orange)](data/)

> **Beyond Utility and Privacy Audit: Group-Level Error Patterns in Differentially Private Educational Prediction Models**

<p align="center">
  <img src="https://img.shields.io/badge/Privacy-DP--SGD-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Fairness-Group--Level-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Attack-MIA-teal?style=for-the-badge" />
</p>

## 📋 Overview

This repository contains the official implementation for our EDM 2026 paper **"Evaluating Private Training in Educational Prediction Beyond Utility and Privacy Audit"**. 

We investigate whether privacy-preserving training (using **DP-SGD**) changes not only predictive quality and measured privacy leakage, but also the distribution of prediction errors across student groups.



### Key Findings

- **RQ1**: DP-SGD reduces predictive quality more clearly than it changes measured privacy-audit outcomes
- **RQ2**: Private training increases group-level error gaps (TPR & FPR gaps) even when average performance remains acceptable

> ⚠️ **Implication**: Evaluating private training only through utility and privacy audit can miss important fairness-related deployment effects.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

```
torch>=1.12.0
opacus>=1.4.0
scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
```

---

## 📁 Repository Structure

```
.
├── src/                          # Core implementation
│   ├── data_loader.py           # Dataset loading (OULAD, UCI697)
│   ├── model_trainer.py         # Training with DP-SGD support
│   └── run_all.py               # Experiment orchestration
│
├── analysis/                     # Analysis & table generation
│   ├── build_paper_tables.py    # Generate LaTeX-ready tables
│   ├── final_tables.py          # Final result aggregation
│   ├── analyze_all_final.py     # Comprehensive analysis
│   ├── statistical_significance_test.py  # Statistical testing
│   └── threshold_sensitivity_analysis.py # Threshold robustness
│
├── paper/                        # Paper materials
│   ├── EDM_Article_Submission.tex  # LaTeX source
│   ├── sigproc.bib              # Bibliography
│   └── figures/                 # Paper figures
│
└── data/                         # Datasets (not included, see below)
    ├── OULAD/
    └── UCI697/
```

---

## 🔬 Reproducing the Paper

### Step 1: Data Preparation

Download the datasets:

- **OULAD**: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- **UCI697**: [Predict Students' Dropout](https://archive.ics.uci.edu/ml/datasets/Predict+Students+Dropout+and+Academic+Success)

Place them in the `data/` directory:

```bash
data/
├── OULAD/studentInfo.csv
└── UCI697/data.csv
```

### Step 2: Run Experiments

```bash
# Generate experiment plan
python src/generate_fast_plan.py --dataset OULAD --output plan.json

# Run all experiments
python src/run_all.py --only-plan plan.json --mode fast
```

### Step 3: Generate Tables

```bash
# Generate all paper tables
python analysis/build_paper_tables.py

# Generate LaTeX-ready tables
python analysis/final_tables.py
```

---

## 📊 Main Results

### Table 1: OULAD Main Results (Privacy-Utility-Fairness Trade-offs)

| Condition | Test AUC | MIA AUC | TPR Gap | FPR Gap |
|-----------|----------|---------|---------|---------|
| LR Baseline | 0.640 ± 0.002 | 0.501 ± 0.001 | 0.063 ± 0.010 | 0.075 ± 0.015 |
| MLP-small Baseline | 0.661 ± 0.004 | 0.506 ± 0.003 | 0.016 ± 0.009 | 0.009 ± 0.002 |
| **MLP-small DP-SGD ε=1** | 0.644 ± 0.002 | 0.503 ± 0.001 | **0.045 ± 0.014** | **0.064 ± 0.014** |
| MLP-small DP-SGD ε=5 | 0.649 ± 0.002 | 0.503 ± 0.002 | 0.031 ± 0.016 | 0.046 ± 0.012 |
| MLP-small DP-SGD ε=10 | 0.649 ± 0.002 | 0.503 ± 0.002 | 0.028 ± 0.019 | 0.042 ± 0.011 |

### Table 2: UCI697 Cross-Dataset Validation

| Condition | Test AUC | MIA AUC |
|-----------|----------|---------|
| Baseline | 0.952 ± 0.008 | 0.502 ± 0.013 |
| DP-SGD ε=1 | 0.919 ± 0.012 | 0.495 ± 0.013 |
| DP-SGD ε=5 | 0.938 ± 0.012 | 0.495 ± 0.014 |
| DP-SGD ε=10 | 0.941 ± 0.012 | 0.494 ± 0.013 |

---

## 🔧 Implementation Details

### DP-SGD Training

We use [Opacus](https://opacus.ai/) for differentially private training:

```python
from src.model_trainer import ModelTrainer

# Train with DP-SGD (ε=1)
trainer = ModelTrainer("MLP", variant="small", seed=42)
result = trainer.train(
    X_train, y_train,
    train_defense="DP-SGD",
    eps=1.0,
    delta=1/len(X_train)
)
```

### Membership Inference Attack

Loss-based MIA following Yeom et al. (2018):

```python
# Compute per-sample losses
losses = trainer.compute_sample_losses(X, y)

# Attack score: lower loss → higher membership score
attack_scores = 1.0 - normalize(losses)
```

### Fairness Metrics

Worst-group error gaps:

```
Δ_TPR = max_g TPR_g - min_g TPR_g
Δ_FPR = max_g FPR_g - min_g FPR_g
```

---

## 📚 Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{meng2026evaluating,
  title={Evaluating Private Training in Educational Prediction Beyond Utility and Privacy Audit},
  author={Meng, Xianghui and Zhang, Yujing and Chen, Xian and Lin, Jionghao},
  booktitle={Proceedings of the 16th International Conference on Educational Data Mining},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgments

This work was supported by the Faculty Research Fund and the URC Grant (No. 2401102970) at The University of Hong Kong.

We thank the creators of:
- [OULAD Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Opacus Library](https://opacus.ai/)

---

## 📮 Contact

For questions or issues, please open an issue on GitHub or contact:

- Xianghui Meng: margretmeng1020@gmail.com
- Jionghao Lin (Corresponding): jionghao@hku.hk

---

<p align="center">
  <b>⭐ Star this repository if you find it helpful!</b>
</p>
