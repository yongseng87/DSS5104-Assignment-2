# DSS5104 Assignment 2 – Deep Learning for Tabular Data

A critical exploration and evaluation of deep learning methods versus classical machine learning methods for tabular data prediction. Six models are benchmarked across three real-world datasets covering regression and classification tasks.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Experiments](#running-the-experiments)
- [References](#references)

---

## Overview

This project benchmarks two deep learning methods — **Tabular ResNet** and **FT-Transformer** — against four classical machine learning baselines on three tabular datasets. Each experiment follows a consistent pipeline:

1. Exploratory Data Analysis (EDA)
2. Preprocessing (feature scaling, encoding)
3. Train/Validation/Test split (60 / 20 / 20)
4. Hyperparameter tuning with **Optuna** (20 trials per model)
5. Multi-seed training and evaluation (3 seeds)
6. Recording of predictive performance metrics and training/inference times

---

## Repository Structure

```
DSS5104-Assignment-2/
├── data/
│   ├── adult.csv                                   # UCI Adult Income dataset
│   └── porto-seguro-safe-driver-prediction.zip     # Porto Seguro dataset (Kaggle)
├── dataset1_results/                               # Output charts for Dataset 1
├── dataset2_results/                               # Output charts for Dataset 2
├── dataset3_results/                               # Output charts for Dataset 3
├── dataset1_california_housing.ipynb         # Notebook 1: California Housing (Regression)
├── dataset2_adult_income.ipynb               # Notebook 2: Adult Income (Classification)
├── dataset3_porto_seguro.ipynb               # Notebook 3: Porto Seguro (Classification)
├── requirements.txt
└── README.md
```

---

## Datasets

| # | Dataset | Task | Source | Features |
|---|---------|------|--------|----------|
| 1 | [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | Regression | `scikit-learn` (auto-downloaded) | 8 numerical |
| 2 | [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | Binary Classification | Included in `data/adult.csv` | Mix of numerical & categorical |
| 3 | [Porto Seguro Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data) | Binary Classification (imbalanced) | Included in `data/porto-seguro-safe-driver-prediction.zip` | Mix of numerical, binary & categorical |

---

## Models

Six models are trained and evaluated in every notebook:

| Model | Type | Library |
|-------|------|---------|
| Tabular ResNet | Deep Learning | `rtdl_revisiting_models` |
| FT-Transformer | Deep Learning | `rtdl_revisiting_models` |
| XGBoost | Gradient Boosting | `xgboost` |
| LightGBM | Gradient Boosting | `lightgbm` |
| Random Forest | Ensemble | `scikit-learn` |
| Ridge Regression / Logistic Regression | Linear | `scikit-learn` |

All models are tuned with **Optuna** and trained across **3 random seeds** for robust evaluation.

### Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression (Dataset 1) | RMSE, MAE, R² |
| Classification (Datasets 2 & 3) | Accuracy, AUC-ROC, F1 (Dataset 3 also reports PR-AUC) |

---

## Results

Experiment outputs (comparison charts, training curves, and feature importance plots) are saved automatically to the respective results folders:

- `dataset1_results/` — California Housing
- `dataset2_results/` — Adult Income
- `dataset3_results/` — Porto Seguro

Each folder contains:
- `*_model_comparison_results.png` — predictive performance comparison across models
- `*_model_comparison_timing(s).png` — training and inference time comparison
- `*_resnet_training_curves.png` — ResNet epoch-level training/validation curves
- `*_ft_transformer_training_curves.png` — FT-Transformer epoch-level curves
- `Feature Importance/` — SHAP or model-native feature importance plots

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional but recommended for deep learning models)

All Python dependencies are listed in `requirements.txt`.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yongseng87/DSS5104-Assignment-2.git
cd DSS5104-Assignment-2
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset setup

| Dataset | Action required |
|---------|----------------|
| **California Housing** | Automatically downloaded by `scikit-learn` at runtime. No action needed. |
| **Adult Income** | Already included at `data/adult.csv`. No action needed. |
| **Porto Seguro** | Already included at `data/porto-seguro-safe-driver-prediction.zip`. The notebook extracts it automatically at runtime. |

---

## Running the Experiments

Launch Jupyter and open each notebook:

```bash
jupyter notebook
```

Then run the notebooks in order (or independently — they are self-contained):

| Notebook | Dataset | Task |
|----------|---------|------|
| `dataset1_california_housing.ipynb` | California Housing | Regression |
| `dataset2_adult_income.ipynb` | Adult Income | Binary Classification |
| `dataset3_porto_seguro.ipynb` | Porto Seguro | Binary Classification |

To run a notebook non-interactively from the command line:

```bash
jupyter nbconvert --to notebook --execute dataset1_california_housing.ipynb --output dataset1_california_housing_executed.ipynb
```

> **Note:** Running all experiments (Optuna tuning × 6 models × 3 seeds) can take a significant amount of time, especially for deep learning models. A GPU is strongly recommended. Estimated runtimes vary by hardware.

---

## References

- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. [[arXiv]](https://arxiv.org/abs/2106.11959) [[GitHub]](https://github.com/yandex-research/rtdl)
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
