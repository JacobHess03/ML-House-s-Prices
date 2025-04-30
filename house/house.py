import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
from xgboost import XGBRegressor

from preprocess import preprocess
from features import feature

# ---------------------------------------------------------------------------
# 1) Caricamento dati e preprocessing iniziale
# ---------------------------------------------------------------------------
file_path = r'house/house_data.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Errore: file non trovato a questo percorso: {file_path}")
    exit()

print("========================= Preprocessing Iniziale =========================")
X_initial, y_initial = preprocess(df)
print(f"X_initial shape: {X_initial.shape}")
print(f"y_initial shape: {y_initial.shape}")

# ---------------------------------------------------------------------------
# 2) Feature engineering, VIF, scaling e split
# ---------------------------------------------------------------------------
print("\n================ Feature Engineering & Split =================")
X_train, X_test, y_train, y_test = feature(X_initial, y_initial)
print(f"X_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")



# ---------------------------------------------------------------------------
# 4) Addestramento e valutazione dei modelli di base
# ---------------------------------------------------------------------------
print("\n================ Addestramento Modelli =================")

# -- Linear Regression
print("\n--- Linear Regression ---")
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)
r2_lin   = r2_score(y_test, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print(f"R² (Linear): {r2_lin:.4f}")
print(f"RMSE (Linear): {rmse_lin:.2f}")

# -- Ridge Regression
print("\n--- Ridge Regression (L2) ---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge   = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"R² (Ridge): {r2_ridge:.4f}")
print(f"RMSE (Ridge): {rmse_ridge:.2f}")

# -- Lasso Regression
print("\n--- Lasso Regression (L1) ---")
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
r2_lasso   = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"R² (Lasso): {r2_lasso:.4f}")
print(f"RMSE (Lasso): {rmse_lasso:.2f}")

# -- Gradient Boosting Regressor
print("\n--- Gradient Boosting Regressor ---")
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=73
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb   = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print(f"R² (GB): {r2_gb:.4f}")
print(f"RMSE (GB): {rmse_gb:.2f}")

# ---------------------------------------------------------------------------
# 5) XGBoost Regressor con grid search manuale su reg_alpha/reg_lambda
# ---------------------------------------------------------------------------
print("\n--- XGBoost Regressor: Grid Search su alpha e reg_lambda ---")

# Prepara DMatrix su training (senza .values)
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)

# Parametri fissi
xgb_base_params = {
    'objective': 'reg:squarederror',
    'eta':        0.1,
    'max_depth':  4,
}

# Griglia
param_grid = {
    'alpha':     [0, 0.1, 0.5, 1],
    'reg_lambda':[0, 0.1, 0.5, 1]
}

best_rmse = float("inf")
best_params = {}

for alpha in param_grid['alpha']:
    for reg_l in param_grid['reg_lambda']:
        params = xgb_base_params.copy()
        params['reg_alpha']  = alpha
        params['reg_lambda'] = reg_l

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=100,
            nfold=5,
            metrics="rmse",
            early_stopping_rounds=10,
            seed=42,
            verbose_eval=False
        )
        mean_rmse = cv_results['test-rmse-mean'].min()
        print(f" alpha={alpha:<4} reg_lambda={reg_l:<4} → RMSE: {mean_rmse:.4f}")

        if mean_rmse < best_rmse:
            best_rmse   = mean_rmse
            best_params = {'reg_alpha': alpha, 'reg_lambda': reg_l}

print(f"\nMigliori XGB params: {best_params} → RMSE CV: {best_rmse:.4f}")

# Addestra il modello finale con i parametri ottimali
xgb_model = XGBRegressor(
    objective    = 'reg:squarederror',
    eta          = 0.1,
    max_depth    = 4,
    reg_alpha    = best_params['reg_alpha'],
    reg_lambda   = best_params['reg_lambda'],
    n_estimators = 100,
    random_state = 73
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

r2_xgb   = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"\nR² (XGBoost): {r2_xgb:.4f}")
print(f"RMSE (XGBoost): {rmse_xgb:.2f}")

# ---------------------------------------------------------------------------
# 6) Scatter plot Reale vs Predetto per tutti i modelli
# ---------------------------------------------------------------------------
print("\n================ Scatter Plots Reale vs Predetto =================")
def plot_real_vs_pred(y_true, preds, labels):
    plt.figure(figsize=(12, 8))
    for y_pred, lab in zip(preds, labels):
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, label=lab)
    lims = [y_true.min(), y_true.max()]
    plt.plot(lims, lims, 'k--', lw=2)
    plt.xlabel("Prezzo Reale")
    plt.ylabel("Prezzo Predetto")
    plt.legend()
    plt.grid(True)
    plt.title("Confronto modelli: Reale vs Predetto")
    plt.show()

plot_real_vs_pred(
    y_test,
    [y_pred_lin, y_pred_ridge, y_pred_lasso, y_pred_gb, y_pred_xgb],
    ["Linear", "Ridge", "Lasso", "GB", "XGBoost"]
)

print("\n======================= Fine ========================")


# -------------------------
# 6) Calcolo RMSE e NRMSE
# -------------------------
print("\n================ Calcolo RMSE e NRMSE =================")

models_preds = {
    "Linear Regression":        y_pred_lin,
    "Ridge Regression":         y_pred_ridge,
    "Lasso Regression":         y_pred_lasso,
    "Gradient Boosting Regr.":  y_pred_gb,
    "XGBoost Regressor":        y_pred_xgb
}

for name, y_pred in models_preds.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    y_range = y_test.max() - y_test.min()
    nrmse_range = rmse / y_range
    y_mean = y_test.mean()
    nrmse_mean = rmse / y_mean

    print(f"\n{name}")
    print(f"  RMSE:          {rmse:.2f}")
    print(f"  NRMSE range:   {nrmse_range:.4f}")
    print(f"  NRMSE mean:    {nrmse_mean:.4f}")