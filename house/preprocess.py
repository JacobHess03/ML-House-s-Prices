
import matplotlib.pyplot as plt

import seaborn as sns

from utils import elimina_variabili_vif_pvalue


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

    X_selected = elimina_variabili_vif_pvalue(X, y, vif_threshold=10.0, pvalue_threshold=0.05)
    # con VIF
    #X_selected = elimina_vif(X)
   



    return X_selected, y