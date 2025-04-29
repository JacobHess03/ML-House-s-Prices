# ✨ Previsione Prezzo Case 🏡

Questo progetto è una pipeline di Machine Learning end-to-end per prevedere il prezzo delle case 🏠 basandosi su un dataset pubblico. Il nostro viaggio include la preparazione accurata dei dati, con un focus speciale sulla **gestione della multicollinearità** 📈 tra le feature utilizzando un metodo iterativo basato sul Variance Inflation Factor (VIF) 🔍. Alla fine, confronteremo le performance di tre popolari modelli di regressione lineare: Regressione Lineare standard, Ridge e Lasso 📊.

## 📂 Dataset

Il cuore del nostro progetto è il dataset `house_data.csv` (o `kc_house_data.csv`). Contiene una ricchezza di informazioni sulle proprietà immobiliari per aiutarci a svelare i segreti dei prezzi delle case!

## ✅ Obiettivi del Progetto

Ecco cosa ci prefiggiamo di fare:

* 🔎 Caricare ed esplorare i dati per capire di cosa si tratta.
* ✨ Pulire i dati, dicendo addio ai duplicati 👋.
* 📊 Analizzare la correlazione tra le feature con una bella heatmap 🔥.
* 🔧 Affrontare la multicollinearità: identificare e mitigare le feature problematiche usando un processo di rimozione iterativa basato sul VIF 🔍.
    * 🛡️ **Protezione Speciale:** Alcune feature chiave (come `bathrooms`, `bedrooms`, `floors`) sono protette e non verranno rimosse dal processo VIF, anche se mostrano alta collinearità.
* 📏 Standardizzare le feature per prepararle al meglio per i modelli regolarizzati.
* ✂️ Dividere i dati in set di training e test per un addestramento e una valutazione onesti.
* 🧠 Addestrare i nostri tre modelli di regressione: Regressione Lineare, Ridge e Lasso.
* 📈 Valutare le performance di ogni modello con metriche di regressione appropriate (R2 Score, MSE, RMSE, MAE).

## 📁 Struttura del Progetto (Esempio)

Per mantenere tutto ordinato 🧹, il codice è suddiviso in file con responsabilità specifiche:

* 📄 `main_script.py` (o `run.py`): La mente 🧠 dietro l'operazione. Carica i dati, chiama le funzioni di pre-elaborazione e infine addestra/valuta i modelli.
* 📄 `preprocess.py`: Il pulitore ✨ e selezionatore VIF 🔍. Contiene la funzione `preprocess(df)` per la pulizia iniziale e la gestione iterativa del VIF.
* 📄 `features.py`: Lo standardizzatore 📏 e divisore ✂️. Contiene la funzione `feature(X_selected, y)` per lo scaling e lo split train/test sui dati già filtrati.

## 🐍 Requisiti

Per far girare questo progetto sul tuo PC 💻, assicurati di avere installato:

* 📦 `pandas`
* 📦 `numpy`
* 📦 `matplotlib`
* 📦 `seaborn`
* 📦 `scikit-learn`
* 📦 `statsmodels`

Il modo più semplice per installarli è via pip in un ambiente virtuale:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```
Se hai un file requirements.txt, puoi semplicemente usare:
Bash
```
pip install -r requirements.txt
```
▶️ Come Eseguire lo Script

Segui questi semplici passi:

  Clona (o scarica) questo repository sul tuo computer.

  ⚠️ Assicurati che il file del dataset (house_data.csv o kc_house_data.csv) si trovi nella stessa cartella degli script Python, oppure aggiorna il percorso nel codice.

  Apri il terminale o prompt dei comandi nella directory del progetto.

  Esegui lo script principale:
    Bash
```
    python main_script.py
```
Osserva il terminale per vedere i passaggi di pre-elaborazione e i risultati dei modelli! ✨
🔧 Metodologia di Pre-elaborazione Dati

Il processo è stato attentamente pensato 🤔:

  Caricamento & Pulizia: I dati vengono caricati e i duplicati rimossi ✅.
  Split Iniziale X/y: Separiamo le feature dal prezzo target 🎯, rimuovendo colonne non utili.
  Analisi Correlazione: Creiamo una matrice di correlazione visuale (heatmap 🔥) per capire le relazioni tra le feature numeriche.
  Rimozione Iterativa VIF: Usiamo un loop per identificare e rimuovere le feature con VIF > soglia (default 10) 🔍. Le feature protette ('bathrooms', 'bedrooms', 'floors') vengono saltate in questo processo 🛡️, permettendo al ciclo di continuare a valutare le altre feature.
  Standardizzazione: Le feature selezionate vengono standardizzate (ridimensionate con media 0 e deviazione standard 1) 📏, cruciale per Ridge e Lasso.
  Split Train/Test: I dati standardizzati vengono divisi in set di training e test (80/20) ✂️.

📊 Valutazione dei Modelli

Ogni modello viene addestrato sul set di training standardizzato e valutato sul set di test utilizzando le metriche fondamentali per la regressione ✅:

  R2 Score (quanto del prezzo è spiegato dal modello)
  Mean Squared Error (MSE)
  Root Mean Squared Error (RMSE - errore tipico nella stessa unità del prezzo)
  Mean Absolute Error (MAE)

Questi numeri ci dicono quanto bene ogni modello riesce a prevedere i prezzi sul set di dati non visto.
📈 Risultati e Visualizzazioni (Opzionale)

Lo script stamperà le metriche di valutazione per ogni modello. Per un'analisi visiva più approfondita 🖼️, potresti aggiungere plot come:

  Uno scatter plot dei prezzi reali vs. i prezzi predetti (ideale: punti su una linea diagonale perfetta).
  Un plot dei residui per controllare l'errore del modello.
