
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# Importiamo le metriche per la regressione
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Assumiamo che la funzione 'preprocess' gestisca il caricamento dati,
# pulizia iniziale e split iniziale in X e y (senza VIF o scaling).
# Assumiamo che la funzione 'feature' prenda X e y, esegua la selezione
# VIF iterativa, la standardizzazione e lo split train/test,
# restituendo i dati pronti per l'addestramento.
#
# Importantissimo: Assicurati che la tua funzione feature(X, y) restituisca
# X_train, X_test, y_train, y_test che sono stati:
# 1. Feature selezionate (es. con VIF iterativo)
# 2. Scalati (es. con StandardScaler)
# Altrimenti, i modelli Ridge e Lasso non funzioneranno correttamente.
from preprocess import preprocess # Assicurati che questo file esista e contenga preprocess
from features import feature     # Assicurati che questo file esista e contenga feature


# Assicurati che il file sia nella stessa directory dello script o specifica il percorso completo
file_path = r'29_aprile_2025/house/house_data.csv' # Ho accorciato il percorso per comodità, adattalo se necessario
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Errore: Il file non trovato a questo percorso: {file_path}")
    # Esci o gestisci l'errore in modo appropriato
    exit()

# Passaggio 1: Preprocessing iniziale (caricamento, pulizia, split X/y iniziale)
# La tua funzione preprocess(df) dovrebbe restituire X (dataframe) e y (series)
print("========================= Preprocessing Iniziale tramite funzione preprocess() =========================")
X_initial, y_initial = preprocess(df) # Chiamiamo preprocess per l'input iniziale
print(f"Shape X_initial ottenuta da preprocess(): {X_initial.shape}")
print(f"Shape y_initial ottenuta da preprocess(): {y_initial.shape}")
print("========================= Fine Preprocessing Iniziale =========================")


# Passaggio 2: Feature Engineering, Selezione VIF, Standardizzazione e Split Train/Test
# La tua funzione feature(X, y) dovrebbe prendere X e y, fare VIF, scaling e split
print("\n========================= Feature Engineering e Preparazione Dati tramite funzione feature() =========================")
# Passiamo X_initial e y_initial alla funzione feature
X_train, X_test, y_train, y_test = feature(X_initial, y_initial)
print(f"Shape X_train ottenuta dalla funzione feature(): {X_train.shape}")
print(f"Shape X_test ottenuta dalla funzione feature(): {X_test.shape}")
print(f"Shape y_train ottenuta dalla funzione feature(): {y_train.shape}")
print(f"Shape y_test ottenuta dalla funzione feature(): {y_test.shape}")
print("========================= Fine Feature Engineering e Preparazione Dati =========================")


# =============================================================================
# Addestramento e Valutazione dei Modelli di Regressione
# =============================================================================
print("\n========================= Addestramento e Valutazione Modelli di Regressione =========================")

# 1. Modello di Regressione Lineare Standard
print("\n--- Modello: Linear Regression ---")
linear_model = LinearRegression()
# Addestra il modello sui dati di training (presumibilmente scalati e filtrati dalla funzione feature)
linear_model.fit(X_train, y_train)
# Fa previsioni sul test set (presumibilmente scalato e filtrato dalla funzione feature)
y_pred_linear = linear_model.predict(X_test)

# Calcola le metriche di regressione
r2_linear = r2_score(y_test, y_pred_linear)


# Stampa le metriche
print(f"R2 Score (Linear Regression): {r2_linear:.4f}")

# --- Linear Regression ---
mse_linear = mean_squared_error(y_test, y_pred_linear)  
rmse_linear = np.sqrt(mse_linear)
print(f"RMSE (Linear Regression): {rmse_linear:.4f}")

# 1. Normalizzazione su range
y_range = y_test.max() - y_test.min()
nrmse_range = rmse_linear / y_range
print(f"NRMSE rispetto al range: {nrmse_range:.4f}")

# 2. Normalizzazione su media
y_mean = y_test.mean()
nrmse_mean = rmse_linear / y_mean
print(f"NRMSE rispetto alla media (CV RMSE): {nrmse_mean:.4f}")

# 2. Modello Ridge (Regolarizzazione L2)
print("\n--- Modello: Ridge ---")
# alpha è il parametro di regolarizzazione.
# Un valore comune di partenza è 1.0. Andrebbe ottimizzato.
ridge_model = Ridge(alpha=1.0)
# Addestra il modello sui dati di training scalati
ridge_model.fit(X_train, y_train)
# Fa previsioni sul test set scalato
y_pred_ridge = ridge_model.predict(X_test)

# Calcola le metriche di regressione
r2_ridge = r2_score(y_test, y_pred_ridge)


# Stampa le metriche
print(f"R2 Score (Ridge Regression, alpha=1.0): {r2_ridge:.4f}")







# --- Ridge Regression ---
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
print(f"RMSE (Ridge Regression, alpha=1.0): {rmse_ridge:.4f}")
# 1. Normalizzazione su range
y_range = y_test.max() - y_test.min()
nrmse_range = rmse_ridge / y_range
print(f"NRMSE rispetto al range: {nrmse_range:.4f}")

# 2. Normalizzazione su media
y_mean = y_test.mean()
nrmse_mean = rmse_ridge / y_mean
print(f"NRMSE rispetto alla media (CV RMSE): {nrmse_mean:.4f}")

# 3. Modello Lasso (Regolarizzazione L1)
print("\n--- Modello: Lasso ---")
# alpha è il parametro di regolarizzazione.
# Un valore comune di partenza è 1.0. Andrebbe ottimizzato.
# Lasso può portare i coefficienti esattamente a zero.
lasso_model = Lasso(alpha=1.0)
# Addestra il modello sui dati di training scalati
lasso_model.fit(X_train, y_train)
# Fa previsioni sul test set scalato
y_pred_lasso = lasso_model.predict(X_test)

# Calcola le metriche di regressione
r2_lasso = r2_score(y_test, y_pred_lasso)


# Stampa le metriche
print(f"R2 Score (Lasso Regression, alpha=1.0): {r2_lasso:.4f}")

# --- Lasso Regression ---
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
print(f"RMSE (Lasso Regression, alpha=1.0): {rmse_lasso:.4f}")

# 1. Normalizzazione su range
y_range = y_test.max() - y_test.min()
nrmse_range = rmse_lasso / y_range
print(f"NRMSE rispetto al range: {nrmse_range:.4f}")

# 2. Normalizzazione su media
y_mean = y_test.mean()
nrmse_mean = rmse_lasso / y_mean
print(f"NRMSE rispetto alla media (CV RMSE): {nrmse_mean:.4f}")






# ESEMPIO (Facoltativo) di come potresti visualizzare i risultati della Regressione con uno Scatter Plot
print("\n========================= Esempio Scatter Plot Reale vs Predetto (Lineare) =========================")
plt.figure(figsize=(10, 6))
# Scatter plot dei valori reali (y_test) vs i valori predetti (y_pred_ridge)
sns.scatterplot(x=y_test, y=y_pred_linear, alpha=0.5) # alpha per trasparenza se ci sono molti punti
# Aggiungi una linea diagonale perfetta (dove i valori reali = predetti)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto (Lineare)")
plt.title("Scatter Plot: Prezzo Reale vs Predetto (Modello Regressione Lineare)")
plt.grid(True)
plt.show()
print("========================= Fine Scatter Plot =========================")


print("\n========================= Fine Addestramento e Valutazione Modelli =========================")

# ESEMPIO (Facoltativo) di come potresti visualizzare i risultati della Regressione con uno Scatter Plot
print("\n========================= Esempio Scatter Plot Reale vs Predetto (Ridge) =========================")
plt.figure(figsize=(10, 6))
# Scatter plot dei valori reali (y_test) vs i valori predetti (y_pred_ridge)
sns.scatterplot(x=y_test, y=y_pred_ridge, alpha=0.5) # alpha per trasparenza se ci sono molti punti
# Aggiungi una linea diagonale perfetta (dove i valori reali = predetti)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto (Ridge)")
plt.title("Scatter Plot: Prezzo Reale vs Predetto (Modello Ridge)")
plt.grid(True)
plt.show()
print("========================= Fine Scatter Plot =========================")



# ESEMPIO (Facoltativo) di come potresti visualizzare i risultati della Regressione con uno Scatter Plot
print("\n========================= Esempio Scatter Plot Reale vs Predetto (Lasso) =========================")
plt.figure(figsize=(10, 6))
# Scatter plot dei valori reali (y_test) vs i valori predetti (y_pred_ridge)
sns.scatterplot(x=y_test, y=y_pred_lasso, alpha=0.5) # alpha per trasparenza se ci sono molti punti
# Aggiungi una linea diagonale perfetta (dove i valori reali = predetti)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto (Lasso)")
plt.title("Scatter Plot: Prezzo Reale vs Predetto (Modello Lasso)")
plt.grid(True)
plt.show()
print("========================= Fine Scatter Plot =========================")










# === 4. Modello: Gradient Boosting Regressor ===
print("\n--- Modello: Gradient Boosting Regressor ---")
# Imposta i parametri (puoi cambiare n_estimators, learning_rate, max_depth a piacere)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=73
)
# Addestra il modello sui dati di training scalati
gb_model.fit(X_train, y_train)
# Fa previsioni sul test set scalato
y_pred_gb = gb_model.predict(X_test)

# Calcola le metriche di regressione
r2_gb = r2_score(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)

# Normalizzazioni
y_range = y_test.max() - y_test.min()
nrmse_gb_range = rmse_gb / y_range
y_mean = y_test.mean()
nrmse_gb_mean = rmse_gb / y_mean

# Stampa le metriche
print(f"R2 Score (GB Regressor): {r2_gb:.4f}")
print(f"RMSE (GB Regressor): {rmse_gb:.4f}")
print(f"NRMSE rispetto al range: {nrmse_gb_range:.4f}")
print(f"NRMSE rispetto alla media (CV RMSE): {nrmse_gb_mean:.4f}")

# === Scatter Plot Reale vs Predetto (GB) ===
print("\n========================= Scatter Plot Reale vs Predetto (GB) =========================")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', lw=2)
plt.xlabel("Prezzo Reale")
plt.ylabel("Prezzo Predetto (GB Regressor)")
plt.title("Scatter Plot: Prezzo Reale vs Predetto (Gradient Boosting)")
plt.grid(True)
plt.show()
print("========================= Fine Scatter Plot GB =========================")