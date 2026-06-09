# Data Directory

This directory should contain the following datasets:

## OULAD (Open University Learning Analytics Dataset)

**Source**: https://analyse.kmi.open.ac.uk/open_dataset

**Required file**:
- `OULAD/studentInfo.csv`

**Description**: 32,593 student-course records with course outcomes and demographic attributes (gender, region, age band, education level).

**Citation**:
```
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). 
Open University Learning Analytics dataset. 
Scientific Data, 4, 170171.
```

## UCI697 (Predict Students' Dropout and Academic Success)

**Source**: https://archive.ics.uci.edu/ml/datasets/Predict+Students+Dropout+and+Academic+Success

**Required file**:
- `UCI697/data.csv`

**Description**: 4,424 higher education records with 36 features related to enrollment background and academic progress.

**Citation**:
```
Realinho, V., Machado, J., Baptista, A., et al. (2022).
Predict students dropout and academic success.
UCI Machine Learning Repository.
```

## Data Privacy Notice

These are public educational datasets used for research purposes. Please refer to the original sources for data usage agreements and privacy policies.

## Preprocessing

The `src/data_loader.py` script automatically handles:
- Feature encoding (Label Encoding for categorical variables)
- Missing value imputation
- StandardScaler normalization
- Train/test splitting (80/20 with stratification)
