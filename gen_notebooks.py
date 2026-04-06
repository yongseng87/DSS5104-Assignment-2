import nbformat as nbf

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 1 – California Housing  (Regression)
# ─────────────────────────────────────────────────────────────────────────────

nb1 = nbf.v4.new_notebook()

nb1.cells = [

# ── 1. Title ─────────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('''# Notebook 1 – California Housing (Regression)

## Overview
We benchmark five tabular-learning methods on the California Housing dataset:
**TabNet**, **FT-Transformer**, **XGBoost**, **Ridge Regression**, and
**Random Forest**.  
Each model is tuned with **Optuna** (20 trials) and evaluated across 3 random
seeds.  Metrics: **RMSE**, **MAE**, **R²**.
'''),

# ── 2. Install ────────────────────────────────────────────────────────────────
nbf.v4.new_code_cell(
'!pip install pytorch-tabnet "rtdl==0.0.13" optuna xgboost lightgbm ucimlrepo scikit-learn pandas numpy matplotlib seaborn shap'
),

# ── 3. Imports ────────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Imports'),
nbf.v4.new_code_cell('''import warnings
warnings.filterwarnings('ignore')

import random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

import rtdl
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from pytorch_tabnet.tab_model import TabNetRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
'''),

# ── 4. Configuration ──────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Configuration'),
nbf.v4.new_code_cell('''SEEDS          = [42, 123, 456]
N_OPTUNA_TRIALS = 20
TEST_SIZE       = 0.20
VAL_FRAC        = 0.25   # fraction of train+val to use as val  → 60/20/20 split
'''),

# ── 5. Data Loading & EDA ─────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Data Loading & EDA'),
nbf.v4.new_code_cell('''from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df   = data.frame

print("Shape:", df.shape)
print()
print(df.dtypes)
print()
print(df.describe())
print()
print("Missing values:")
print(df.isnull().sum())
'''),
nbf.v4.new_code_cell('''fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=40, edgecolor='k', alpha=0.7)
    axes[i].set_title(col)
plt.suptitle("California Housing – feature distributions", fontsize=14)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
'''),

# ── 6. Preprocessing ──────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Preprocessing'),
nbf.v4.new_code_cell('''feature_cols = [c for c in df.columns if c != 'MedHouseVal']
target_col   = 'MedHouseVal'

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

print(f"Features shape: {X.shape},  Target shape: {y.shape}")
'''),

# ── 7. Data Splitting 60/20/20 ────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Data Splitting (60 / 20 / 20)'),
nbf.v4.new_code_cell('''X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=VAL_FRAC, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape}, Val: {X_val_sc.shape}, Test: {X_test_sc.shape}")
'''),

# ── 8. Helper Functions ───────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Helper Functions'),
nbf.v4.new_code_cell('''def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_ft_transformer(model, X_num_tr, X_cat_tr, y_tr,
                          X_num_v, X_cat_v, y_v,
                          lr=1e-3, n_epochs=100, batch_size=256,
                          task='regression', device_='cpu'):
    model = model.to(device_)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()

    X_num_tr_t = torch.FloatTensor(X_num_tr).to(device_)
    X_cat_tr_t = torch.LongTensor(X_cat_tr).to(device_) if X_cat_tr is not None else None
    y_tr_t     = torch.FloatTensor(y_tr).to(device_)
    X_num_v_t  = torch.FloatTensor(X_num_v).to(device_)
    X_cat_v_t  = torch.LongTensor(X_cat_v).to(device_) if X_cat_v is not None else None
    y_v_t      = torch.FloatTensor(y_v).to(device_)

    train_losses, val_losses = [], []
    best_val  = float('inf')
    best_state = None
    patience  = 20
    pat_cnt   = 0

    for epoch in range(n_epochs):
        model.train()
        n   = len(X_num_tr_t)
        idx = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            b  = idx[i:i+batch_size]
            xn = X_num_tr_t[b]
            xc = X_cat_tr_t[b] if X_cat_tr_t is not None else None
            yb = y_tr_t[b]
            optimizer.zero_grad()
            out  = model(xn, xc).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vout  = model(X_num_v_t, X_cat_v_t).squeeze(-1)
            vloss = criterion(vout, y_v_t).item()
        train_losses.append(ep_loss / n)
        val_losses.append(vloss)
        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict_ft_transformer(model, X_num, X_cat, device_, batch_size=512):
    model.eval()
    model    = model.to(device_)
    X_num_t  = torch.FloatTensor(X_num).to(device_)
    X_cat_t  = torch.LongTensor(X_cat).to(device_) if X_cat is not None else None
    preds    = []
    with torch.no_grad():
        for i in range(0, len(X_num_t), batch_size):
            xn  = X_num_t[i:i+batch_size]
            xc  = X_cat_t[i:i+batch_size] if X_cat_t is not None else None
            out = model(xn, xc).squeeze(-1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)
'''),

# ── 9a. TabNet ─────────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Model 1: TabNet'),
nbf.v4.new_code_cell('''all_results = []

def tabnet_reg_objective(trial):
    n_d    = trial.suggest_int('n_d', 8, 64)
    n_a    = trial.suggest_int('n_a', 8, 64)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma  = trial.suggest_float('gamma', 1.0, 2.0)
    lr     = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    set_seed(42)
    m = TabNetRegressor(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                        optimizer_params={'lr': lr}, verbose=0, seed=42,
                        device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_train_sc, y_train, eval_set=[(X_val_sc, y_val)],
          eval_name=['val'], eval_metric=['rmse'],
          patience=15, max_epochs=100, batch_size=1024, virtual_batch_size=256)
    preds = m.predict(X_val_sc).flatten()
    return np.sqrt(mean_squared_error(y_val, preds))

study_tn = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_tn.optimize(tabnet_reg_objective, n_trials=N_OPTUNA_TRIALS)
best_tn = study_tn.best_params
print(f"Best TabNet params: {best_tn}")
'''),
nbf.v4.new_code_cell('''print("Training TabNet across seeds...")
for seed in SEEDS:
    set_seed(seed)
    m = TabNetRegressor(
        n_d=best_tn['n_d'], n_a=best_tn['n_a'], n_steps=best_tn['n_steps'],
        gamma=best_tn['gamma'], optimizer_params={'lr': best_tn['lr']},
        verbose=0, seed=seed,
        device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_train_sc, y_train, eval_set=[(X_val_sc, y_val)],
          eval_name=['val'], eval_metric=['rmse'],
          patience=20, max_epochs=200, batch_size=1024, virtual_batch_size=256)
    preds = m.predict(X_test_sc).flatten()
    rmse, mae, r2 = compute_regression_metrics(y_test, preds)
    all_results.append({'method': 'TabNet', 'seed': seed,
                        'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
'''),

# ── 9b. FT-Transformer ────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Model 2: FT-Transformer'),
nbf.v4.new_code_cell('''n_num = X_train_sc.shape[1]

def ft_reg_objective(trial):
    d_token  = trial.suggest_categorical('d_token', [64, 128, 192])
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    attn_drop = trial.suggest_float('attention_dropout', 0.0, 0.3)
    ffn_drop  = trial.suggest_float('ffn_dropout', 0.0, 0.3)
    lr        = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    set_seed(42)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num,
        cat_cardinalities=[],
        d_token=d_token,
        n_blocks=n_blocks,
        attention_dropout=attn_drop,
        ffn_d_hidden=int(d_token * 4 / 3),
        ffn_dropout=ffn_drop,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, _, _ = train_ft_transformer(
        model, X_train_sc, None, y_train,
        X_val_sc, None, y_val,
        lr=lr, n_epochs=50, batch_size=256,
        task='regression', device_=str(device))
    preds = predict_ft_transformer(model, X_val_sc, None, str(device))
    return np.sqrt(mean_squared_error(y_val, preds))

study_ft = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_ft.optimize(ft_reg_objective, n_trials=N_OPTUNA_TRIALS)
best_ft = study_ft.best_params
print(f"Best FT-Transformer params: {best_ft}")
'''),
nbf.v4.new_code_cell('''print("Training FT-Transformer across seeds...")
ft_train_curves = {}
for seed in SEEDS:
    set_seed(seed)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num,
        cat_cardinalities=[],
        d_token=best_ft['d_token'],
        n_blocks=best_ft['n_blocks'],
        attention_dropout=best_ft['attention_dropout'],
        ffn_d_hidden=int(best_ft['d_token'] * 4 / 3),
        ffn_dropout=best_ft['ffn_dropout'],
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, tr_l, va_l = train_ft_transformer(
        model, X_train_sc, None, y_train,
        X_val_sc, None, y_val,
        lr=best_ft['lr'], n_epochs=100, batch_size=256,
        task='regression', device_=str(device))
    ft_train_curves[seed] = (tr_l, va_l)
    preds = predict_ft_transformer(model, X_test_sc, None, str(device))
    rmse, mae, r2 = compute_regression_metrics(y_test, preds)
    all_results.append({'method': 'FT-Transformer', 'seed': seed,
                        'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
'''),

# ── 9c. XGBoost ────────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Model 3: XGBoost'),
nbf.v4.new_code_cell('''def xgb_reg_objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'max_depth':     trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'random_state':  42
    }
    set_seed(42)
    m = xgb.XGBRegressor(**params, verbosity=0)
    m.fit(X_train_sc, y_train, eval_set=[(X_val_sc, y_val)], verbose=False)
    preds = m.predict(X_val_sc)
    return np.sqrt(mean_squared_error(y_val, preds))

study_xgb = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_reg_objective, n_trials=N_OPTUNA_TRIALS)
best_xgb = study_xgb.best_params
print(f"Best XGBoost params: {best_xgb}")
'''),
nbf.v4.new_code_cell('''print("Training XGBoost across seeds...")
xgb_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = xgb.XGBRegressor(**best_xgb, random_state=seed, verbosity=0)
    m.fit(X_train_sc, y_train)
    preds = m.predict(X_test_sc)
    rmse, mae, r2 = compute_regression_metrics(y_test, preds)
    all_results.append({'method': 'XGBoost', 'seed': seed,
                        'rmse': rmse, 'mae': mae, 'r2': r2})
    xgb_model_last = m
    print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
'''),

# ── 9d. Ridge ─────────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Model 4: Ridge Regression'),
nbf.v4.new_code_cell('''def ridge_objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
    set_seed(42)
    m = Ridge(alpha=alpha)
    m.fit(X_train_sc, y_train)
    preds = m.predict(X_val_sc)
    return np.sqrt(mean_squared_error(y_val, preds))

study_ridge = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
study_ridge.optimize(ridge_objective, n_trials=N_OPTUNA_TRIALS)
best_ridge = study_ridge.best_params
print(f"Best Ridge params: {best_ridge}")
'''),
nbf.v4.new_code_cell('''print("Training Ridge across seeds...")
for seed in SEEDS:
    set_seed(seed)
    m = Ridge(alpha=best_ridge['alpha'])
    m.fit(X_train_sc, y_train)
    preds = m.predict(X_test_sc)
    rmse, mae, r2 = compute_regression_metrics(y_test, preds)
    all_results.append({'method': 'Ridge', 'seed': seed,
                        'rmse': rmse, 'mae': mae, 'r2': r2})
    print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
'''),

# ── 9e. Random Forest ─────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Model 5: Random Forest'),
nbf.v4.new_code_cell('''def rf_reg_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state':    42
    }
    set_seed(42)
    m = RandomForestRegressor(**params, n_jobs=-1)
    m.fit(X_train_sc, y_train)
    preds = m.predict(X_val_sc)
    return np.sqrt(mean_squared_error(y_val, preds))

study_rf = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_reg_objective, n_trials=N_OPTUNA_TRIALS)
best_rf = study_rf.best_params
print(f"Best RF params: {best_rf}")
'''),
nbf.v4.new_code_cell('''print("Training Random Forest across seeds...")
rf_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = RandomForestRegressor(**best_rf, random_state=seed, n_jobs=-1)
    m.fit(X_train_sc, y_train)
    preds = m.predict(X_test_sc)
    rmse, mae, r2 = compute_regression_metrics(y_test, preds)
    all_results.append({'method': 'RandomForest', 'seed': seed,
                        'rmse': rmse, 'mae': mae, 'r2': r2})
    rf_model_last = m
    print(f"  Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
'''),

# ── 10. Results ───────────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Results'),
nbf.v4.new_code_cell('''df_res = pd.DataFrame(all_results)
summary = df_res.groupby('method').agg(
    rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
    mae_mean=('mae', 'mean'),   mae_std=('mae', 'std'),
    r2_mean=('r2', 'mean'),     r2_std=('r2', 'std')
).round(4)

summary['RMSE'] = summary['rmse_mean'].astype(str) + ' +/- ' + summary['rmse_std'].astype(str)
summary['MAE']  = summary['mae_mean'].astype(str)  + ' +/- ' + summary['mae_std'].astype(str)
summary['R2']   = summary['r2_mean'].astype(str)   + ' +/- ' + summary['r2_std'].astype(str)
print(summary[['RMSE', 'MAE', 'R2']])
'''),

# ── 11. Visualizations ────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('## Visualizations'),
nbf.v4.new_code_cell('''methods = summary.index.tolist()
rmse_means = summary['rmse_mean'].values
rmse_stds  = summary['rmse_std'].values

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(methods, summary['rmse_mean'].values, yerr=summary['rmse_std'].values,
            capsize=5, color='steelblue', alpha=0.8)
axes[0].set_title('RMSE (lower is better)')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(methods, summary['mae_mean'].values, yerr=summary['mae_std'].values,
            capsize=5, color='darkorange', alpha=0.8)
axes[1].set_title('MAE (lower is better)')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=30)

axes[2].bar(methods, summary['r2_mean'].values, yerr=summary['r2_std'].values,
            capsize=5, color='forestgreen', alpha=0.8)
axes[2].set_title('R² (higher is better)')
axes[2].set_ylabel('R²')
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle('California Housing – Model Comparison', fontsize=14)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# FT-Transformer training curves
fig, axes = plt.subplots(1, len(SEEDS), figsize=(15, 4))
for ax, seed in zip(axes, SEEDS):
    tr_l, va_l = ft_train_curves[seed]
    ax.plot(tr_l, label='Train')
    ax.plot(va_l, label='Val')
    ax.set_title(f'Seed {seed}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
plt.suptitle('FT-Transformer Training Curves', fontsize=13)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# XGBoost feature importance
if xgb_model_last is not None:
    fi = xgb_model_last.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi}).sort_values(
        'importance', ascending=True)
    fi_df.plot.barh(x='feature', y='importance', figsize=(8, 5), legend=False,
                    color='teal')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()
'''),
nbf.v4.new_code_cell('''# Random Forest feature importance
if rf_model_last is not None:
    fi = rf_model_last.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi}).sort_values(
        'importance', ascending=True)
    fi_df.plot.barh(x='feature', y='importance', figsize=(8, 5), legend=False,
                    color='coral')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()
'''),

# ── 12. Conclusions ───────────────────────────────────────────────────────────
nbf.v4.new_markdown_cell('''## Analysis & Conclusions

### Summary
We compared five methods on the California Housing regression task.

- **Tree-based methods** (XGBoost, Random Forest) typically deliver strong
  performance on tabular data with pure numerical features.
- **TabNet** and **FT-Transformer** are deep-learning alternatives that can
  match or exceed gradient-boosting methods when properly tuned.
- **Ridge Regression** serves as a linear baseline; it is competitive when
  features are already well-scaled.

### Observations
- The 3-seed evaluation reduces variance in performance estimates.
- Optuna's TPE sampler efficiently navigates the hyperparameter space in
  only 20 trials.
- FT-Transformer training curves illustrate early stopping in action.

### Next Steps
- Feature engineering (e.g., geographic clustering via Lat/Lon).
- Ensemble or stacking of the best individual models.
- SHAP-based explanations for tree ensembles.
'''),
]  # end nb1.cells

with open('notebook1_california_housing.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("notebook1_california_housing.ipynb written.")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 2 – Adult Income  (Binary Classification)
# ─────────────────────────────────────────────────────────────────────────────

nb2 = nbf.v4.new_notebook()

nb2.cells = [

nbf.v4.new_markdown_cell('''# Notebook 2 – Adult Income (Binary Classification)

## Overview
We benchmark six tabular-learning methods on the UCI Adult Income dataset:
**TabNet**, **FT-Transformer**, **XGBoost**, **LightGBM**, **Random Forest**,
and **Logistic Regression**.  
Each model is tuned with **Optuna** (20 trials) and evaluated across 3 seeds.  
Metrics: **Accuracy**, **AUC-ROC**, **F1**.
'''),

nbf.v4.new_code_cell(
'!pip install pytorch-tabnet "rtdl==0.0.13" optuna xgboost lightgbm ucimlrepo scikit-learn pandas numpy matplotlib seaborn shap'
),

nbf.v4.new_markdown_cell('## Imports'),
nbf.v4.new_code_cell('''import warnings
warnings.filterwarnings('ignore')

import random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

import rtdl
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from pytorch_tabnet.tab_model import TabNetClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
'''),

nbf.v4.new_markdown_cell('## Configuration'),
nbf.v4.new_code_cell('''SEEDS           = [42, 123, 456]
N_OPTUNA_TRIALS = 20
TEST_SIZE       = 0.20
VAL_FRAC        = 0.25
'''),

nbf.v4.new_markdown_cell('## Data Loading & EDA'),
nbf.v4.new_code_cell('''adult = fetch_ucirepo(id=2)
X_raw = adult.data.features.copy()
y_raw = adult.data.targets.copy()

print("Features shape:", X_raw.shape)
print()
print(X_raw.dtypes)
print()
print(X_raw.describe())
'''),
nbf.v4.new_code_cell('''# Target distribution
target_series = y_raw.iloc[:, 0].astype(str).str.strip()
print("Unique target values:", target_series.unique())
print(target_series.value_counts())
'''),
nbf.v4.new_code_cell('''# Missing value overview
print("Missing values (including '?'):")
for col in X_raw.columns:
    n = (X_raw[col].astype(str).str.strip() == '?').sum()
    if n > 0:
        print(f"  {col}: {n}")
'''),
nbf.v4.new_code_cell('''# Distribution plots for numeric columns
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), num_cols):
    ax.hist(X_raw[col].dropna(), bins=40, edgecolor='k', alpha=0.7)
    ax.set_title(col)
plt.suptitle("Adult Income – numerical feature distributions", fontsize=13)
plt.tight_layout()
plt.show()
'''),

nbf.v4.new_markdown_cell('## Preprocessing'),
nbf.v4.new_code_cell('''num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
            'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']

# Binarise target
y_series = y_raw.iloc[:, 0].astype(str).str.strip()
y = np.where(y_series.str.contains('>50K'), 1, 0).astype(np.int64)
print(f"Positive rate: {y.mean():.3%}")

# Replace '?' with NaN
X_proc = X_raw.copy()
for col in cat_cols:
    X_proc[col] = X_proc[col].astype(str).str.strip().replace('?', np.nan)

# Impute
for col in num_cols:
    med = X_proc[col].median()
    X_proc[col] = X_proc[col].fillna(med)
for col in cat_cols:
    mode_val = X_proc[col].mode()[0]
    X_proc[col] = X_proc[col].fillna(mode_val)

print("Missing after imputation:", X_proc.isnull().sum().sum())
'''),
nbf.v4.new_code_cell('''# Encode
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_enc = ord_enc.fit_transform(X_proc[cat_cols]).astype(np.int64)

scaler_num = StandardScaler()
X_num_enc  = scaler_num.fit_transform(X_proc[num_cols].values.astype(np.float32))

cat_cardinalities = [len(cats) for cats in ord_enc.categories_]
print("Cat cardinalities:", cat_cardinalities)
print("X_num shape:", X_num_enc.shape, "  X_cat shape:", X_cat_enc.shape)
'''),

nbf.v4.new_markdown_cell('## Data Splitting (60 / 20 / 20)'),
nbf.v4.new_code_cell('''# Combined array for sklearn models
X_all = np.concatenate([X_num_enc, X_cat_enc.astype(np.float32)], axis=1)

idx = np.arange(len(y))
idx_tv, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=42, stratify=y)
idx_train, idx_val = train_test_split(idx_tv, test_size=VAL_FRAC, random_state=42,
                                       stratify=y[idx_tv])

X_train_all, X_val_all, X_test_all = X_all[idx_train], X_all[idx_val], X_all[idx_test]
X_train_num, X_val_num, X_test_num = X_num_enc[idx_train], X_num_enc[idx_val], X_num_enc[idx_test]
X_train_cat, X_val_cat, X_test_cat = X_cat_enc[idx_train], X_cat_enc[idx_val], X_cat_enc[idx_test]
y_train_clf, y_val_clf, y_test_clf  = y[idx_train], y[idx_val], y[idx_test]

# StandardScaler on combined (for TabNet)
sc2 = StandardScaler()
X_train_sc = sc2.fit_transform(X_train_all)
X_val_sc   = sc2.transform(X_val_all)
X_test_sc  = sc2.transform(X_test_all)

print(f"Train: {X_train_sc.shape}, Val: {X_val_sc.shape}, Test: {X_test_sc.shape}")
'''),

nbf.v4.new_markdown_cell('## Helper Functions'),
nbf.v4.new_code_cell('''def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return acc, auc, f1


def train_ft_transformer(model, X_num_tr, X_cat_tr, y_tr,
                          X_num_v, X_cat_v, y_v,
                          lr=1e-3, n_epochs=100, batch_size=256,
                          task='regression', device_='cpu'):
    model = model.to(device_)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()

    X_num_tr_t = torch.FloatTensor(X_num_tr).to(device_)
    X_cat_tr_t = torch.LongTensor(X_cat_tr).to(device_) if X_cat_tr is not None else None
    y_tr_t     = torch.FloatTensor(y_tr.astype(np.float32)).to(device_)
    X_num_v_t  = torch.FloatTensor(X_num_v).to(device_)
    X_cat_v_t  = torch.LongTensor(X_cat_v).to(device_) if X_cat_v is not None else None
    y_v_t      = torch.FloatTensor(y_v.astype(np.float32)).to(device_)

    train_losses, val_losses = [], []
    best_val   = float('inf')
    best_state = None
    patience   = 20
    pat_cnt    = 0

    for epoch in range(n_epochs):
        model.train()
        n   = len(X_num_tr_t)
        idx = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            b  = idx[i:i+batch_size]
            xn = X_num_tr_t[b]
            xc = X_cat_tr_t[b] if X_cat_tr_t is not None else None
            yb = y_tr_t[b]
            optimizer.zero_grad()
            out  = model(xn, xc).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vout  = model(X_num_v_t, X_cat_v_t).squeeze(-1)
            vloss = criterion(vout, y_v_t).item()
        train_losses.append(ep_loss / n)
        val_losses.append(vloss)
        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict_ft_transformer(model, X_num, X_cat, device_, batch_size=512):
    model.eval()
    model   = model.to(device_)
    X_num_t = torch.FloatTensor(X_num).to(device_)
    X_cat_t = torch.LongTensor(X_cat).to(device_) if X_cat is not None else None
    preds   = []
    with torch.no_grad():
        for i in range(0, len(X_num_t), batch_size):
            xn  = X_num_t[i:i+batch_size]
            xc  = X_cat_t[i:i+batch_size] if X_cat_t is not None else None
            out = model(xn, xc).squeeze(-1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)
'''),

nbf.v4.new_markdown_cell('## Model 1: TabNet'),
nbf.v4.new_code_cell('''all_results = []

def tabnet_clf_objective(trial):
    n_d     = trial.suggest_int('n_d', 8, 64)
    n_a     = trial.suggest_int('n_a', 8, 64)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma   = trial.suggest_float('gamma', 1.0, 2.0)
    lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    set_seed(42)
    m = TabNetClassifier(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         optimizer_params={'lr': lr}, verbose=0, seed=42,
                         device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_train_sc, y_train_clf.astype(int),
          eval_set=[(X_val_sc, y_val_clf.astype(int))],
          eval_name=['val'], eval_metric=['auc'],
          patience=15, max_epochs=100, batch_size=1024, virtual_batch_size=256)
    prob = m.predict_proba(X_val_sc)[:, 1]
    return -roc_auc_score(y_val_clf, prob)

study_tn = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_tn.optimize(tabnet_clf_objective, n_trials=N_OPTUNA_TRIALS)
best_tn = study_tn.best_params
print(f"Best TabNet params: {best_tn}")
'''),
nbf.v4.new_code_cell('''print("Training TabNet across seeds...")
for seed in SEEDS:
    set_seed(seed)
    m = TabNetClassifier(
        n_d=best_tn['n_d'], n_a=best_tn['n_a'], n_steps=best_tn['n_steps'],
        gamma=best_tn['gamma'], optimizer_params={'lr': best_tn['lr']},
        verbose=0, seed=seed,
        device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_train_sc, y_train_clf.astype(int),
          eval_set=[(X_val_sc, y_val_clf.astype(int))],
          eval_name=['val'], eval_metric=['auc'],
          patience=20, max_epochs=200, batch_size=1024, virtual_batch_size=256)
    preds = m.predict(X_test_sc)
    probs = m.predict_proba(X_test_sc)[:, 1]
    acc, auc, f1 = compute_classification_metrics(y_test_clf, preds, probs)
    all_results.append({'method': 'TabNet', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 2: FT-Transformer'),
nbf.v4.new_code_cell('''n_num_ft = X_train_num.shape[1]

def ft_clf_objective(trial):
    d_token   = trial.suggest_categorical('d_token', [64, 128, 192])
    n_blocks  = trial.suggest_int('n_blocks', 1, 3)
    attn_drop = trial.suggest_float('attention_dropout', 0.0, 0.3)
    ffn_drop  = trial.suggest_float('ffn_dropout', 0.0, 0.3)
    lr        = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    set_seed(42)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_ft,
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_dropout=attn_drop,
        ffn_d_hidden=int(d_token * 4 / 3),
        ffn_dropout=ffn_drop,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, _, _ = train_ft_transformer(
        model, X_train_num, X_train_cat, y_train_clf,
        X_val_num, X_val_cat, y_val_clf,
        lr=lr, n_epochs=50, batch_size=256,
        task='classification', device_=str(device))
    raw = predict_ft_transformer(model, X_val_num, X_val_cat, str(device))
    prob = torch.sigmoid(torch.tensor(raw)).numpy()
    return -roc_auc_score(y_val_clf, prob)

study_ft = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_ft.optimize(ft_clf_objective, n_trials=N_OPTUNA_TRIALS)
best_ft = study_ft.best_params
print(f"Best FT-Transformer params: {best_ft}")
'''),
nbf.v4.new_code_cell('''print("Training FT-Transformer across seeds...")
ft_train_curves = {}
for seed in SEEDS:
    set_seed(seed)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_ft,
        cat_cardinalities=cat_cardinalities,
        d_token=best_ft['d_token'],
        n_blocks=best_ft['n_blocks'],
        attention_dropout=best_ft['attention_dropout'],
        ffn_d_hidden=int(best_ft['d_token'] * 4 / 3),
        ffn_dropout=best_ft['ffn_dropout'],
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, tr_l, va_l = train_ft_transformer(
        model, X_train_num, X_train_cat, y_train_clf,
        X_val_num, X_val_cat, y_val_clf,
        lr=best_ft['lr'], n_epochs=100, batch_size=256,
        task='classification', device_=str(device))
    ft_train_curves[seed] = (tr_l, va_l)
    raw  = predict_ft_transformer(model, X_test_num, X_test_cat, str(device))
    prob = torch.sigmoid(torch.tensor(raw)).numpy()
    pred = (prob >= 0.5).astype(int)
    acc, auc, f1 = compute_classification_metrics(y_test_clf, pred, prob)
    all_results.append({'method': 'FT-Transformer', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 3: XGBoost'),
nbf.v4.new_code_cell('''def xgb_clf_objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'max_depth':     trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'random_state':  42, 'use_label_encoder': False, 'eval_metric': 'auc'
    }
    set_seed(42)
    m = xgb.XGBClassifier(**params, verbosity=0)
    m.fit(X_train_all, y_train_clf)
    prob = m.predict_proba(X_val_all)[:, 1]
    return -roc_auc_score(y_val_clf, prob)

study_xgb = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_clf_objective, n_trials=N_OPTUNA_TRIALS)
best_xgb = study_xgb.best_params
print(f"Best XGBoost params: {best_xgb}")
'''),
nbf.v4.new_code_cell('''print("Training XGBoost across seeds...")
xgb_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = xgb.XGBClassifier(**best_xgb, random_state=seed, verbosity=0,
                           use_label_encoder=False, eval_metric='auc')
    m.fit(X_train_all, y_train_clf)
    preds = m.predict(X_test_all)
    probs = m.predict_proba(X_test_all)[:, 1]
    acc, auc, f1 = compute_classification_metrics(y_test_clf, preds, probs)
    all_results.append({'method': 'XGBoost', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    xgb_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 4: LightGBM'),
nbf.v4.new_code_cell('''def lgb_objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'num_leaves':    trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'random_state':  42, 'verbose': -1
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train_all, y_train_clf)
    prob = m.predict_proba(X_val_all)[:, 1]
    return -roc_auc_score(y_val_clf, prob)

study_lgb = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS)
best_lgb = study_lgb.best_params
print(f"Best LightGBM params: {best_lgb}")
'''),
nbf.v4.new_code_cell('''print("Training LightGBM across seeds...")
lgb_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1)
    m.fit(X_train_all, y_train_clf)
    preds = m.predict(X_test_all)
    probs = m.predict_proba(X_test_all)[:, 1]
    acc, auc, f1 = compute_classification_metrics(y_test_clf, preds, probs)
    all_results.append({'method': 'LightGBM', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    lgb_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 5: Random Forest'),
nbf.v4.new_code_cell('''def rf_clf_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state':    42
    }
    set_seed(42)
    m = RandomForestClassifier(**params, n_jobs=-1)
    m.fit(X_train_all, y_train_clf)
    prob = m.predict_proba(X_val_all)[:, 1]
    return -roc_auc_score(y_val_clf, prob)

study_rf = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_clf_objective, n_trials=N_OPTUNA_TRIALS)
best_rf = study_rf.best_params
print(f"Best RF params: {best_rf}")
'''),
nbf.v4.new_code_cell('''print("Training Random Forest across seeds...")
rf_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = RandomForestClassifier(**best_rf, random_state=seed, n_jobs=-1)
    m.fit(X_train_all, y_train_clf)
    preds = m.predict(X_test_all)
    probs = m.predict_proba(X_test_all)[:, 1]
    acc, auc, f1 = compute_classification_metrics(y_test_clf, preds, probs)
    all_results.append({'method': 'RandomForest', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    rf_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 6: Logistic Regression'),
nbf.v4.new_code_cell('''def lr_objective(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    m = LogisticRegression(C=C, max_iter=1000, random_state=42)
    m.fit(X_train_all, y_train_clf)
    prob = m.predict_proba(X_val_all)[:, 1]
    return -roc_auc_score(y_val_clf, prob)

study_lr = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_lr.optimize(lr_objective, n_trials=N_OPTUNA_TRIALS)
best_lr = study_lr.best_params
print(f"Best LR params: {best_lr}")
'''),
nbf.v4.new_code_cell('''print("Training Logistic Regression across seeds...")
for seed in SEEDS:
    set_seed(seed)
    m = LogisticRegression(C=best_lr['C'], max_iter=1000, random_state=seed)
    m.fit(X_train_all, y_train_clf)
    preds = m.predict(X_test_all)
    probs = m.predict_proba(X_test_all)[:, 1]
    acc, auc, f1 = compute_classification_metrics(y_test_clf, preds, probs)
    all_results.append({'method': 'LogisticRegression', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'f1': f1})
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Results'),
nbf.v4.new_code_cell('''df_res = pd.DataFrame(all_results)
summary = df_res.groupby('method').agg(
    acc_mean=('accuracy', 'mean'), acc_std=('accuracy', 'std'),
    auc_mean=('auc', 'mean'),      auc_std=('auc', 'std'),
    f1_mean=('f1', 'mean'),        f1_std=('f1', 'std')
).round(4)

summary['Accuracy'] = summary['acc_mean'].astype(str) + ' +/- ' + summary['acc_std'].astype(str)
summary['AUC-ROC']  = summary['auc_mean'].astype(str) + ' +/- ' + summary['auc_std'].astype(str)
summary['F1']       = summary['f1_mean'].astype(str)  + ' +/- ' + summary['f1_std'].astype(str)
print(summary[['Accuracy', 'AUC-ROC', 'F1']])
'''),

nbf.v4.new_markdown_cell('## Visualizations'),
nbf.v4.new_code_cell('''methods = summary.index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(methods, summary['acc_mean'].values, yerr=summary['acc_std'].values,
            capsize=5, color='steelblue', alpha=0.8)
axes[0].set_title('Accuracy (higher is better)')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(methods, summary['auc_mean'].values, yerr=summary['auc_std'].values,
            capsize=5, color='darkorange', alpha=0.8)
axes[1].set_title('AUC-ROC (higher is better)')
axes[1].tick_params(axis='x', rotation=30)

axes[2].bar(methods, summary['f1_mean'].values, yerr=summary['f1_std'].values,
            capsize=5, color='forestgreen', alpha=0.8)
axes[2].set_title('F1 (higher is better)')
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle('Adult Income – Model Comparison', fontsize=14)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# FT-Transformer training curves
fig, axes = plt.subplots(1, len(SEEDS), figsize=(15, 4))
for ax, seed in zip(axes, SEEDS):
    tr_l, va_l = ft_train_curves[seed]
    ax.plot(tr_l, label='Train')
    ax.plot(va_l, label='Val')
    ax.set_title(f'Seed {seed}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.legend()
plt.suptitle('FT-Transformer Training Curves', fontsize=13)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# LightGBM feature importance
if lgb_model_last is not None:
    all_cols = num_cols + cat_cols
    fi = lgb_model_last.feature_importances_
    fi_df = pd.DataFrame({'feature': all_cols, 'importance': fi}).sort_values(
        'importance', ascending=True)
    fi_df.plot.barh(x='feature', y='importance', figsize=(8, 6),
                    legend=False, color='mediumpurple')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.show()
'''),

nbf.v4.new_markdown_cell('''## Analysis & Conclusions

### Summary
We compared six methods on the UCI Adult Income binary classification task.

- **LightGBM** and **XGBoost** typically achieve top AUC-ROC on this dataset
  thanks to their ability to handle mixed numerical/categorical features.
- **FT-Transformer** leverages embeddings for categorical features, which can
  yield competitive AUC with sufficient training.
- **Logistic Regression** provides a fast and interpretable baseline.
- **TabNet** and **Random Forest** round out the comparison.

### Observations
- The dataset is moderately imbalanced (~24% positive); AUC-ROC is the most
  reliable metric here.
- OrdinalEncoding + StandardScaling is a straightforward preprocessing pipeline
  that works across all methods.
- 3-seed evaluation reveals model stability under random initialization.

### Next Steps
- One-hot encoding for tree methods vs. ordinal for neural methods.
- Class-weight adjustment to improve F1 for the minority class.
- SHAP explanations for the best-performing model.
'''),
]  # end nb2.cells

with open('notebook2_adult_income.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("notebook2_adult_income.ipynb written.")


# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK 3 – Porto Seguro  (Binary Classification, Imbalanced)
# ─────────────────────────────────────────────────────────────────────────────

nb3 = nbf.v4.new_notebook()

nb3.cells = [

nbf.v4.new_markdown_cell('''# Notebook 3 – Porto Seguro Safe Driver Prediction (Classification)

## Overview
We benchmark five tabular-learning methods on the Porto Seguro dataset:
**TabNet**, **FT-Transformer**, **XGBoost**, **LightGBM**, and **Random Forest**.  
Each model is tuned with **Optuna** (20 trials) and evaluated across 3 seeds.  
Metrics: **Accuracy**, **AUC-ROC**, **Normalized Gini** (= 2·AUC − 1), **F1**.

> **Note:** The dataset is heavily imbalanced (~3.6 % positives).  
> If the Kaggle API is not configured, a synthetic dataset matching the Porto
> Seguro schema is generated automatically.
'''),

nbf.v4.new_code_cell(
'!pip install pytorch-tabnet "rtdl==0.0.13" optuna xgboost lightgbm ucimlrepo scikit-learn pandas numpy matplotlib seaborn shap'
),

nbf.v4.new_markdown_cell('## Imports'),
nbf.v4.new_code_cell('''import warnings
warnings.filterwarnings('ignore')

import random, os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

import rtdl
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from pytorch_tabnet.tab_model import TabNetClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
'''),

nbf.v4.new_markdown_cell('## Configuration'),
nbf.v4.new_code_cell('''SEEDS           = [42, 123, 456]
N_OPTUNA_TRIALS = 20
TEST_SIZE       = 0.20
VAL_FRAC        = 0.25
'''),

nbf.v4.new_markdown_cell('## Data Loading'),
nbf.v4.new_code_cell('''os.makedirs('data/porto_seguro', exist_ok=True)
df = None

try:
    import kaggle
    kaggle.api.competition_download_files(
        'porto-seguro-safe-driver-prediction', path='data/porto_seguro'
    )
    zip_path = 'data/porto_seguro/porto-seguro-safe-driver-prediction.zip'
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('data/porto_seguro')
    df = pd.read_csv('data/porto_seguro/train.csv')
    print(f"Loaded real data: {df.shape}")
except Exception as e:
    print(f"Kaggle download failed: {e}")
    print("Please manually download train.csv from:")
    print("https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data")
    print("and place it at data/porto_seguro/train.csv")
    df = None
'''),
nbf.v4.new_code_cell('''if df is None:
    # Try to load from disk if previously downloaded
    local_path = 'data/porto_seguro/train.csv'
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        print(f"Loaded from disk: {df.shape}")
'''),
nbf.v4.new_code_cell('''if df is None:
    print("Creating synthetic demo dataset matching Porto Seguro schema...")
    np.random.seed(42)
    n_rows = 50000

    cols = {}
    cols['id'] = np.arange(n_rows)

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        cols[f'ps_ind_{i:02d}'] = np.random.randint(0, 5, n_rows)
    for i in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        cols[f'ps_ind_{i:02d}_bin'] = np.random.randint(0, 2, n_rows)
    cols['ps_ind_02_cat'] = np.random.randint(-1, 5, n_rows)
    cols['ps_ind_04_cat'] = np.random.randint(-1, 3, n_rows)
    cols['ps_ind_05_cat'] = np.random.randint(-1, 7, n_rows)

    for i in [1, 2, 3]:
        cols[f'ps_reg_0{i}'] = np.random.uniform(0, 2, n_rows).astype(np.float32)

    for i in range(1, 16):
        cols[f'ps_car_{i:02d}_cat'] = np.random.randint(-1, 10, n_rows)
    for i in [12, 13, 14, 15]:
        cols[f'ps_car_{i:02d}'] = np.random.uniform(0, 10, n_rows).astype(np.float32)
    cols['ps_car_11'] = np.random.randint(0, 4, n_rows)

    for i in range(1, 21):
        cols[f'ps_calc_{i:02d}'] = np.random.uniform(0, 1, n_rows).astype(np.float32)
    for i in range(1, 21):
        cols[f'ps_calc_{i:02d}_bin'] = np.random.randint(0, 2, n_rows)

    cols['target'] = (np.random.random(n_rows) < 0.036).astype(int)
    df = pd.DataFrame(cols)
    print(f"Synthetic dataset: {df.shape}, positive rate: {df['target'].mean():.3%}")
    print("NOTE: Results on synthetic data are for demonstration only!")
'''),

nbf.v4.new_markdown_cell('## EDA'),
nbf.v4.new_code_cell('''print("Shape:", df.shape)
print()
print("Target distribution:")
print(df['target'].value_counts())
print(f"Positive rate: {df['target'].mean():.3%}")
'''),
nbf.v4.new_code_cell('''# -1 values as missing indicator
n_neg1 = (df == -1).sum()
print("Columns with -1 (missing sentinel):")
print(n_neg1[n_neg1 > 0])
'''),
nbf.v4.new_code_cell('''# Sample correlation heat-map (first 20 cols)
sample_cols = [c for c in df.columns if c not in ['id', 'target']][:20]
plt.figure(figsize=(12, 10))
sns.heatmap(df[sample_cols].corr(), cmap='coolwarm', center=0)
plt.title('Porto Seguro – correlation (first 20 features)')
plt.tight_layout()
plt.show()
'''),

nbf.v4.new_markdown_cell('## Preprocessing'),
nbf.v4.new_code_cell('''df_proc = df.drop(columns=['id']).copy()
y = df_proc['target'].values.astype(np.int64)
df_feat = df_proc.drop(columns=['target'])

cat_cols_ps  = [c for c in df_feat.columns if c.endswith('_cat')]
num_cols_ps  = [c for c in df_feat.columns if c not in cat_cols_ps]

print(f"Numerical features: {len(num_cols_ps)}")
print(f"Categorical features: {len(cat_cols_ps)}")

# Replace -1 with NaN
df_feat[cat_cols_ps] = df_feat[cat_cols_ps].replace(-1, np.nan)
df_feat[num_cols_ps] = df_feat[num_cols_ps].replace(-1, np.nan)

# Impute
for col in cat_cols_ps:
    mode_val = df_feat[col].mode()
    if len(mode_val) > 0:
        df_feat[col] = df_feat[col].fillna(mode_val[0])
    else:
        df_feat[col] = df_feat[col].fillna(0)

for col in num_cols_ps:
    df_feat[col] = df_feat[col].fillna(df_feat[col].median())

print("Missing after imputation:", df_feat.isnull().sum().sum())
'''),
nbf.v4.new_code_cell('''# Encode
ord_enc_ps = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat_ps   = ord_enc_ps.fit_transform(
    df_feat[cat_cols_ps].astype(str)).astype(np.int64)

scaler_ps = StandardScaler()
X_num_ps  = scaler_ps.fit_transform(
    df_feat[num_cols_ps].values.astype(np.float32))

cat_cards_ps = [len(cats) for cats in ord_enc_ps.categories_]
print("Cat cardinalities (first 5):", cat_cards_ps[:5])

# Combined array
X_all_ps = np.concatenate([X_num_ps, X_cat_ps.astype(np.float32)], axis=1)
print("X_all shape:", X_all_ps.shape)
'''),

nbf.v4.new_markdown_cell('## Data Splitting (60 / 20 / 20)'),
nbf.v4.new_code_cell('''idx = np.arange(len(y))
idx_tv, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=42, stratify=y)
idx_train, idx_val = train_test_split(idx_tv, test_size=VAL_FRAC, random_state=42,
                                       stratify=y[idx_tv])

X_tr_all, X_v_all, X_te_all = X_all_ps[idx_train], X_all_ps[idx_val], X_all_ps[idx_test]
X_tr_num, X_v_num, X_te_num = X_num_ps[idx_train], X_num_ps[idx_val], X_num_ps[idx_test]
X_tr_cat, X_v_cat, X_te_cat = X_cat_ps[idx_train], X_cat_ps[idx_val], X_cat_ps[idx_test]
y_train_ps, y_val_ps, y_test_ps = y[idx_train], y[idx_val], y[idx_test]

sc_ps = StandardScaler()
X_tr_sc = sc_ps.fit_transform(X_tr_all)
X_v_sc  = sc_ps.transform(X_v_all)
X_te_sc = sc_ps.transform(X_te_all)

spw = float((y_train_ps == 0).sum()) / float((y_train_ps == 1).sum())
print(f"Train: {X_tr_sc.shape}, Val: {X_v_sc.shape}, Test: {X_te_sc.shape}")
print(f"scale_pos_weight = {spw:.2f}")
'''),

nbf.v4.new_markdown_cell('## Helper Functions'),
nbf.v4.new_code_cell('''def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_porto_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return acc, auc, gini, f1


def train_ft_transformer(model, X_num_tr, X_cat_tr, y_tr,
                          X_num_v, X_cat_v, y_v,
                          lr=1e-3, n_epochs=100, batch_size=256,
                          task='regression', device_='cpu'):
    model = model.to(device_)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()

    X_num_tr_t = torch.FloatTensor(X_num_tr).to(device_)
    X_cat_tr_t = torch.LongTensor(X_cat_tr).to(device_) if X_cat_tr is not None else None
    y_tr_t     = torch.FloatTensor(y_tr.astype(np.float32)).to(device_)
    X_num_v_t  = torch.FloatTensor(X_num_v).to(device_)
    X_cat_v_t  = torch.LongTensor(X_cat_v).to(device_) if X_cat_v is not None else None
    y_v_t      = torch.FloatTensor(y_v.astype(np.float32)).to(device_)

    train_losses, val_losses = [], []
    best_val   = float('inf')
    best_state = None
    patience   = 20
    pat_cnt    = 0

    for epoch in range(n_epochs):
        model.train()
        n   = len(X_num_tr_t)
        idx_e = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            b  = idx_e[i:i+batch_size]
            xn = X_num_tr_t[b]
            xc = X_cat_tr_t[b] if X_cat_tr_t is not None else None
            yb = y_tr_t[b]
            optimizer.zero_grad()
            out  = model(xn, xc).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vout  = model(X_num_v_t, X_cat_v_t).squeeze(-1)
            vloss = criterion(vout, y_v_t).item()
        train_losses.append(ep_loss / n)
        val_losses.append(vloss)
        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def predict_ft_transformer(model, X_num, X_cat, device_, batch_size=512):
    model.eval()
    model   = model.to(device_)
    X_num_t = torch.FloatTensor(X_num).to(device_)
    X_cat_t = torch.LongTensor(X_cat).to(device_) if X_cat is not None else None
    preds   = []
    with torch.no_grad():
        for i in range(0, len(X_num_t), batch_size):
            xn  = X_num_t[i:i+batch_size]
            xc  = X_cat_t[i:i+batch_size] if X_cat_t is not None else None
            out = model(xn, xc).squeeze(-1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)
'''),

nbf.v4.new_markdown_cell('## Model 1: TabNet'),
nbf.v4.new_code_cell('''all_results = []

def tabnet_porto_objective(trial):
    n_d     = trial.suggest_int('n_d', 8, 64)
    n_a     = trial.suggest_int('n_a', 8, 64)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma   = trial.suggest_float('gamma', 1.0, 2.0)
    lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    set_seed(42)
    m = TabNetClassifier(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                         optimizer_params={'lr': lr}, verbose=0, seed=42,
                         device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_tr_sc, y_train_ps.astype(int),
          eval_set=[(X_v_sc, y_val_ps.astype(int))],
          eval_name=['val'], eval_metric=['auc'],
          patience=15, max_epochs=100, batch_size=1024, virtual_batch_size=256)
    prob = m.predict_proba(X_v_sc)[:, 1]
    return -roc_auc_score(y_val_ps, prob)

study_tn = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_tn.optimize(tabnet_porto_objective, n_trials=N_OPTUNA_TRIALS)
best_tn = study_tn.best_params
print(f"Best TabNet params: {best_tn}")
'''),
nbf.v4.new_code_cell('''print("Training TabNet across seeds...")
for seed in SEEDS:
    set_seed(seed)
    m = TabNetClassifier(
        n_d=best_tn['n_d'], n_a=best_tn['n_a'], n_steps=best_tn['n_steps'],
        gamma=best_tn['gamma'], optimizer_params={'lr': best_tn['lr']},
        verbose=0, seed=seed,
        device_name='cuda' if torch.cuda.is_available() else 'cpu')
    m.fit(X_tr_sc, y_train_ps.astype(int),
          eval_set=[(X_v_sc, y_val_ps.astype(int))],
          eval_name=['val'], eval_metric=['auc'],
          patience=20, max_epochs=200, batch_size=1024, virtual_batch_size=256)
    preds = m.predict(X_te_sc)
    probs = m.predict_proba(X_te_sc)[:, 1]
    acc, auc, gini, f1 = compute_porto_metrics(y_test_ps, preds, probs)
    all_results.append({'method': 'TabNet', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'gini': gini, 'f1': f1})
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, Gini={gini:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 2: FT-Transformer'),
nbf.v4.new_code_cell('''n_num_ps = X_tr_num.shape[1]

def ft_porto_objective(trial):
    d_token   = trial.suggest_categorical('d_token', [64, 128, 192])
    n_blocks  = trial.suggest_int('n_blocks', 1, 3)
    attn_drop = trial.suggest_float('attention_dropout', 0.0, 0.3)
    ffn_drop  = trial.suggest_float('ffn_dropout', 0.0, 0.3)
    lr        = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    set_seed(42)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_ps,
        cat_cardinalities=cat_cards_ps,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_dropout=attn_drop,
        ffn_d_hidden=int(d_token * 4 / 3),
        ffn_dropout=ffn_drop,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, _, _ = train_ft_transformer(
        model, X_tr_num, X_tr_cat, y_train_ps,
        X_v_num, X_v_cat, y_val_ps,
        lr=lr, n_epochs=50, batch_size=256,
        task='classification', device_=str(device))
    raw  = predict_ft_transformer(model, X_v_num, X_v_cat, str(device))
    prob = torch.sigmoid(torch.tensor(raw)).numpy()
    return -roc_auc_score(y_val_ps, prob)

study_ft = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_ft.optimize(ft_porto_objective, n_trials=N_OPTUNA_TRIALS)
best_ft = study_ft.best_params
print(f"Best FT-Transformer params: {best_ft}")
'''),
nbf.v4.new_code_cell('''print("Training FT-Transformer across seeds...")
ft_train_curves = {}
for seed in SEEDS:
    set_seed(seed)
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_ps,
        cat_cardinalities=cat_cards_ps,
        d_token=best_ft['d_token'],
        n_blocks=best_ft['n_blocks'],
        attention_dropout=best_ft['attention_dropout'],
        ffn_d_hidden=int(best_ft['d_token'] * 4 / 3),
        ffn_dropout=best_ft['ffn_dropout'],
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=1,
    )
    model, tr_l, va_l = train_ft_transformer(
        model, X_tr_num, X_tr_cat, y_train_ps,
        X_v_num, X_v_cat, y_val_ps,
        lr=best_ft['lr'], n_epochs=100, batch_size=256,
        task='classification', device_=str(device))
    ft_train_curves[seed] = (tr_l, va_l)
    raw  = predict_ft_transformer(model, X_te_num, X_te_cat, str(device))
    prob = torch.sigmoid(torch.tensor(raw)).numpy()
    pred = (prob >= 0.5).astype(int)
    acc, auc, gini, f1 = compute_porto_metrics(y_test_ps, pred, prob)
    all_results.append({'method': 'FT-Transformer', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'gini': gini, 'f1': f1})
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, Gini={gini:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 3: XGBoost'),
nbf.v4.new_code_cell('''def xgb_porto_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 8),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'scale_pos_weight': spw,
        'random_state':    42
    }
    set_seed(42)
    m = xgb.XGBClassifier(**params, verbosity=0, use_label_encoder=False,
                           eval_metric='auc')
    m.fit(X_tr_all, y_train_ps)
    prob = m.predict_proba(X_v_all)[:, 1]
    return -roc_auc_score(y_val_ps, prob)

study_xgb = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_porto_objective, n_trials=N_OPTUNA_TRIALS)
best_xgb = study_xgb.best_params
print(f"Best XGBoost params: {best_xgb}")
'''),
nbf.v4.new_code_cell('''print("Training XGBoost across seeds...")
xgb_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = xgb.XGBClassifier(**best_xgb, scale_pos_weight=spw,
                           random_state=seed, verbosity=0,
                           use_label_encoder=False, eval_metric='auc')
    m.fit(X_tr_all, y_train_ps)
    preds = m.predict(X_te_all)
    probs = m.predict_proba(X_te_all)[:, 1]
    acc, auc, gini, f1 = compute_porto_metrics(y_test_ps, preds, probs)
    all_results.append({'method': 'XGBoost', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'gini': gini, 'f1': f1})
    xgb_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, Gini={gini:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 4: LightGBM'),
nbf.v4.new_code_cell('''def lgb_porto_objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 500),
        'num_leaves':    trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'random_state':  42, 'verbose': -1, 'is_unbalance': True
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr_all, y_train_ps)
    prob = m.predict_proba(X_v_all)[:, 1]
    return -roc_auc_score(y_val_ps, prob)

study_lgb = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_porto_objective, n_trials=N_OPTUNA_TRIALS)
best_lgb = study_lgb.best_params
print(f"Best LightGBM params: {best_lgb}")
'''),
nbf.v4.new_code_cell('''print("Training LightGBM across seeds...")
lgb_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1,
                            is_unbalance=True)
    m.fit(X_tr_all, y_train_ps)
    preds = m.predict(X_te_all)
    probs = m.predict_proba(X_te_all)[:, 1]
    acc, auc, gini, f1 = compute_porto_metrics(y_test_ps, preds, probs)
    all_results.append({'method': 'LightGBM', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'gini': gini, 'f1': f1})
    lgb_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, Gini={gini:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Model 5: Random Forest'),
nbf.v4.new_code_cell('''def rf_porto_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'random_state':    42
    }
    set_seed(42)
    m = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1)
    m.fit(X_tr_all, y_train_ps)
    prob = m.predict_proba(X_v_all)[:, 1]
    return -roc_auc_score(y_val_ps, prob)

study_rf = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_porto_objective, n_trials=N_OPTUNA_TRIALS)
best_rf = study_rf.best_params
print(f"Best RF params: {best_rf}")
'''),
nbf.v4.new_code_cell('''print("Training Random Forest across seeds...")
rf_model_last = None
for seed in SEEDS:
    set_seed(seed)
    m = RandomForestClassifier(**best_rf, random_state=seed,
                                class_weight='balanced', n_jobs=-1)
    m.fit(X_tr_all, y_train_ps)
    preds = m.predict(X_te_all)
    probs = m.predict_proba(X_te_all)[:, 1]
    acc, auc, gini, f1 = compute_porto_metrics(y_test_ps, preds, probs)
    all_results.append({'method': 'RandomForest', 'seed': seed,
                        'accuracy': acc, 'auc': auc, 'gini': gini, 'f1': f1})
    rf_model_last = m
    print(f"  Seed {seed}: Acc={acc:.4f}, AUC={auc:.4f}, Gini={gini:.4f}, F1={f1:.4f}")
'''),

nbf.v4.new_markdown_cell('## Results'),
nbf.v4.new_code_cell('''df_res = pd.DataFrame(all_results)
summary = df_res.groupby('method').agg(
    acc_mean=('accuracy', 'mean'),  acc_std=('accuracy', 'std'),
    auc_mean=('auc', 'mean'),       auc_std=('auc', 'std'),
    gini_mean=('gini', 'mean'),     gini_std=('gini', 'std'),
    f1_mean=('f1', 'mean'),         f1_std=('f1', 'std')
).round(4)

summary['Accuracy']      = summary['acc_mean'].astype(str)  + ' +/- ' + summary['acc_std'].astype(str)
summary['AUC-ROC']       = summary['auc_mean'].astype(str)  + ' +/- ' + summary['auc_std'].astype(str)
summary['Norm. Gini']    = summary['gini_mean'].astype(str) + ' +/- ' + summary['gini_std'].astype(str)
summary['F1']            = summary['f1_mean'].astype(str)   + ' +/- ' + summary['f1_std'].astype(str)
print(summary[['Accuracy', 'AUC-ROC', 'Norm. Gini', 'F1']])
'''),

nbf.v4.new_markdown_cell('## Visualizations'),
nbf.v4.new_code_cell('''methods = summary.index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(methods, summary['acc_mean'].values, yerr=summary['acc_std'].values,
               capsize=5, color='steelblue', alpha=0.8)
axes[0, 0].set_title('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=30)

axes[0, 1].bar(methods, summary['auc_mean'].values, yerr=summary['auc_std'].values,
               capsize=5, color='darkorange', alpha=0.8)
axes[0, 1].set_title('AUC-ROC')
axes[0, 1].tick_params(axis='x', rotation=30)

axes[1, 0].bar(methods, summary['gini_mean'].values, yerr=summary['gini_std'].values,
               capsize=5, color='forestgreen', alpha=0.8)
axes[1, 0].set_title('Normalized Gini')
axes[1, 0].tick_params(axis='x', rotation=30)

axes[1, 1].bar(methods, summary['f1_mean'].values, yerr=summary['f1_std'].values,
               capsize=5, color='firebrick', alpha=0.8)
axes[1, 1].set_title('F1 Score')
axes[1, 1].tick_params(axis='x', rotation=30)

plt.suptitle('Porto Seguro – Model Comparison', fontsize=14)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# FT-Transformer training curves
fig, axes = plt.subplots(1, len(SEEDS), figsize=(15, 4))
for ax, seed in zip(axes, SEEDS):
    tr_l, va_l = ft_train_curves[seed]
    ax.plot(tr_l, label='Train')
    ax.plot(va_l, label='Val')
    ax.set_title(f'Seed {seed}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.legend()
plt.suptitle('FT-Transformer Training Curves', fontsize=13)
plt.tight_layout()
plt.show()
'''),
nbf.v4.new_code_cell('''# XGBoost feature importance (top 20)
if xgb_model_last is not None:
    fi = xgb_model_last.feature_importances_
    all_feat_names = num_cols_ps + cat_cols_ps
    fi_df = pd.DataFrame({'feature': all_feat_names[:len(fi)], 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=True).tail(20)
    fi_df.plot.barh(x='feature', y='importance', figsize=(9, 7),
                    legend=False, color='teal')
    plt.title('XGBoost Top-20 Feature Importance')
    plt.tight_layout()
    plt.show()
'''),
nbf.v4.new_code_cell('''# LightGBM feature importance (top 20)
if lgb_model_last is not None:
    fi = lgb_model_last.feature_importances_
    all_feat_names = num_cols_ps + cat_cols_ps
    fi_df = pd.DataFrame({'feature': all_feat_names[:len(fi)], 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=True).tail(20)
    fi_df.plot.barh(x='feature', y='importance', figsize=(9, 7),
                    legend=False, color='mediumpurple')
    plt.title('LightGBM Top-20 Feature Importance')
    plt.tight_layout()
    plt.show()
'''),

nbf.v4.new_markdown_cell('''## Analysis & Conclusions

### Summary
We benchmarked five models on the Porto Seguro safe-driver prediction task.

- The dataset is **severely imbalanced** (~3.6 % positive), so **Normalized Gini**
  (= 2·AUC − 1) is the key competition metric.
- **LightGBM** with `is_unbalance=True` and **XGBoost** with `scale_pos_weight`
  are well-suited for this imbalanced setting.
- **FT-Transformer** uses learned categorical embeddings, which can capture
  non-linear interactions between the many categorical features.
- **TabNet** provides built-in feature selection via sparse attention.
- **Random Forest** with `class_weight='balanced'` offers a strong tree baseline.

### Observations
- Gini scores on **synthetic** data are near 0 because the synthetic labels are
  random (independent of features). Real data results will differ substantially.
- The preprocessing pipeline — replacing -1 sentinels, mode/median imputation,
  OrdinalEncoding — mirrors common Kaggle solutions for this competition.
- 3-seed evaluation reduces variance in reported Gini.

### Next Steps
- Feature engineering: polynomial interactions among `ps_reg` features.
- Calibrated probability outputs for better threshold tuning.
- Stacking / blending of LightGBM + XGBoost predictions.
- SHAP analysis to understand which feature groups drive predictions.
'''),
]  # end nb3.cells

with open('notebook3_porto_seguro.ipynb', 'w') as f:
    nbf.write(nb3, f)
print("notebook3_porto_seguro.ipynb written.")

print()
print("All three notebooks created successfully.")
