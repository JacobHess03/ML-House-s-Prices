import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor



def preprocess(df):
    print("========================= Dati Iniziali =========================")
    print(df.head())
    print("\n========================= Info Dati =========================")
    df.info()
    print("\n========================= Statistiche Descrittive =========================")
    print(df.describe())
    print("========================= Fine Dati Iniziali =========================")


    print("\n========================= Pulizia Dati: Rimozione Duplicati =========================")
    df_cleaned = df.copy()
    initial_rows = df_cleaned.shape[0]
    df_cleaned = df.drop_duplicates()
    rows_after_cleaning = df_cleaned.shape[0]
    print(f"Righe iniziali: {initial_rows}")
    print(f"Righe dopo rimozione duplicati: {rows_after_cleaning}")
    print(f"Numero di duplicati rimossi: {initial_rows - rows_after_cleaning}")
    print("========================= Fine Pulizia Dati =========================")


    # Prepariamo X e y dai dati puliti
    # Rimuoviamo 'price' (target), 'id' (identificativo non utile per il modello),
    # 'zipcode' e 'date' (richiederebbero un'encoding specifica, per ora li rimuoviamo)
    X = df_cleaned.drop(['price', 'id', 'zipcode','date'], axis=1)
    y = df_cleaned['price']

    print("\n========================= Matrice di Correlazione =========================")
    # Calcoliamo la matrice di correlazione solo sulle colonne numeriche dei dati puliti
    correlation_matrix = df_cleaned.corr(numeric_only=True)
    plt.figure(figsize=(10, 8)) # Aumento leggermente le dimensioni per migliore leggibilità
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Formatta i numeri per chiarezza
    plt.title('Matrice di correlazione delle colonne numeriche (dopo pulizia)')
    plt.show()
    print("========================= Fine Matrice di Correlazione =========================")


    # =============================================================================
    # Rimozione Iterativa della Multicollinearità con VIF
    # =============================================================================
    # Creiamo una copia di X su cui lavoreremo per rimuovere le feature
    X_vif_filtered = X.copy()

    # Definiamo la soglia VIF
    vif_threshold = 10 # Una soglia comune, puoi sperimentare con 5 o 10

    print(f"\n========================= Rimozione Iterativa Feature con VIF > {vif_threshold} =========================")

    # Inizializziamo variabili per il ciclo
    max_vif = float('inf') # Partiamo con un valore alto per entrare nel ciclo
    iteration = 0

    # Eseguiamo il ciclo finché il VIF massimo è sopra la soglia E ci sono più di 1 feature
    while max_vif > vif_threshold and X_vif_filtered.shape[1] > 1:
        iteration += 1
        print(f"\n--- Iterazione {iteration} ---")
        print(f"Features rimanenti: {X_vif_filtered.shape[1]}")

        # Calcola VIF per l'attuale set di feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif_filtered.columns

        # Converti il DataFrame corrente in un array numpy per variance_inflation_factor
        X_np = X_vif_filtered.values

        vif_list = []
        # Itera sugli indici delle colonne dell'array numpy
        for i in range(X_np.shape[1]):
            # Gestisce colonne con un solo valore unico (la varianza è zero, VIF infinito)
            if np.unique(X_np[:, i]).size <= 1:
                vif = float('inf') # Assegna infinito
            else:
                try:
                    # Calcola il VIF per la colonna i rispetto a tutte le altre colonne in X_np
                    vif = variance_inflation_factor(X_np, i)
                except Exception as e:
                    # Cattura altri possibili errori nel calcolo VIF
                    print(f"Errore nel calcolo VIF per colonna '{X_vif_filtered.columns[i]}' (indice {i}): {e}")
                    vif = np.nan # Assegna NaN in caso di errore

            vif_list.append(vif)

        vif_data["VIF"] = vif_list

        # Rimuovi righe con VIF NaN prima di trovare il massimo, per sicurezza
        vif_data = vif_data.dropna(subset=["VIF"])

        # Trova la feature con il VIF più alto nell'attuale set di feature
        if not vif_data.empty:
            # Ordina e prendi la prima riga (quella con il VIF massimo)
            max_vif_row = vif_data.sort_values(by="VIF", ascending=False).iloc[0]
            max_vif = max_vif_row["VIF"]
            feature_to_remove = max_vif_row["Feature"]

            # Se il VIF massimo è sopra la soglia, rimuovi quella feature
            if max_vif > vif_threshold:
                print(f"  - VIF Max: {max_vif:.2f} (Feature: '{feature_to_remove}'). Rimuovo...")
                # Rimuovi la colonna dal DataFrame X_vif_filtered
                X_vif_filtered = X_vif_filtered.drop(columns=[feature_to_remove])
            else:
                # Se il VIF massimo è sotto la soglia, esci dal ciclo
                print(f"  - VIF Max ({max_vif:.2f}) è sotto la soglia ({vif_threshold}). Processo terminato.")

        else: # Questo caso si verifica se rimangono 0 o 1 feature (gestito dalla condizione del while) o se tutti i VIF sono NaN
            print("  - Nessun dato VIF valido da analizzare o solo una feature rimasta. Processo terminato.")
            max_vif = 0 # Imposta max_vif a 0 per uscire dal ciclo

    # Una volta terminato il ciclo, X_vif_filtered contiene solo le feature con VIF <= threshold
    print("\n========================= Fine Rimozione Iterativa VIF =========================")
    print(f"Processo di rimozione VIF terminato.")
    print(f"Features finali selezionate ({X_vif_filtered.shape[1]}):")
    print(X_vif_filtered.columns.tolist())

    # Ora, aggiorniamo X con le features selezionate per le fasi successive
    X_selected = X_vif_filtered



    return X_selected, y