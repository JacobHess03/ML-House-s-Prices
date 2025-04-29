
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
# Importiamo le metriche per la regressione
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
file_path = r'house_data.csv' # Ho accorciato il percorso per comodità, adattalo se necessario
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