# Domain Name Classification & Price Prediction
### Machine Learning Pipeline for Domain Valuation using Structural and Linguistic Features

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ML-based domain name classification across 5 pricing categories using feature engineering and ensembling techniques.

---

## Overview

This project implements a machine learning pipeline to predict domain name pricing categories based on structural and linguistic features. Using **8.7 million domain dataset**, the system extracts 16 engineered features and applies three classification algorithms with hyperparameter optimisation.

**The Result?** Random Forest achieved **65% accuracy** on 5-class imbalanced dataset, with **+4% improvement** through systematic hyperparameter tuning.

---

## Key Results

### Model Performance Comparison

| Model | Default Accuracy | Tuned Accuracy | Improvement |
|-------|------------------|----------------|-------------|
| **Logistic Regression** | 56.0% | 58.0% | +2.0% |
| **Decision Tree** | 59.0% | 62.0% | +3.0% |
| **Random Forest** | 61.0% | **65.0%** ✅ | **+4.0%** |

### Best Model: Random Forest (Tuned)
**Optimal Hyperparameters:**
- `n_estimators=200` (trees in ensemble)
- `max_depth=20` (tree depth)
- `min_samples_split=2` (split threshold)

**Per-Category Performance (F1-Scores):**
- Category 1 (Expensive): **0.96**
- Category 2: **0.47**
- Category 3: **0.37**
- Category 4: **0.29**
- Category 5 (Cheap): **0.22**

---

## Implementation

### Three-Stage Pipeline:

**Stage 1: Feature Engineering**
- Extracted 16 features from domain structure (length, TLD, entropy, character composition).
- Selected top 10 features via correlation analysis.
- Applied StandardScaler normalisation.

**Stage 2: Baseline Models**
- Trained 3 algorithms with default parameters.
- Evaluated on stratified 70-30 train-test split.
- Identified Random Forest as best performer (61%).

**Stage 3: Hyperparameter Optimisation**
- Grid Search with 3-fold cross-validation.
- Tested 54 parameter combinations (Random Forest).
- Final accuracy: 65% (+4% improvement).

---

## Project Highlights

- **Massive Scale**: Handled 8.7M domain dataset with efficient sampling strategy.
- **Feature Engineering**: 16 structural & linguistic features (TLD, entropy, character patterns).
- **Systematic Optimisation**: GridSearchCV across 3 algorithms with CV.
- **Measurable Gains**: +4% accuracy improvement through tuning.
- **Imbalance Handling**: Stratified sampling maintains class distribution.
- **Visualisation**: Confusion matrices and performance comparisons.
- **Reproducible**: Complete pipeline with `random_state` set throughout.

---

## Tech Stack

- **Algorithms**: Logistic Regression, Decision Tree, Random Forest.
- **Feature Engineering**: TLD extraction (tldextract), Shannon entropy, character analysis.
- **Optimisation**: GridSearchCV with 3-fold CV.
- **Evaluation**: Precision, Recall, F1-Score, Confusion Matrices.
- **Dataset**: 8.7M domains across 5 pricing categories.
- **Framework**: scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.

---

## Limitations & Challenges

### Class Imbalance
- **Issue**: Category 1 (57% of data) vs Category 5 (2% of data).
- **Impact**: Model biased toward expensive domains (Cat 1: 96% F1 vs Cat 5: 22% F1).
- **Current Mitigation**: Stratified sampling, per-class metrics.
- **Future Improvement**: SMOTE oversampling, class_weight='balanced'.

### Computational Constraints
- **Issue**: 8.7M rows require significant compute time.
- **Current Approach**: 10% random sampling (609K samples) for training.
- **Impact**: Potential 2-4% accuracy loss vs full dataset.
- **Future Improvement**: Train on full dataset with cloud GPU resources.

### Feature Limitations
- **Issue**: Only structural features (no domain age, backlinks, traffic data).
- **Impact**: Missing key pricing indicators.
- **Future Improvement**: Integrate external domain metadata.
