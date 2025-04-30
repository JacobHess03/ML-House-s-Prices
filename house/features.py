from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def feature(X_selected, y):
    

    # =============================================================================
    # Suddivisione in training e test set (con le features selezionate)
    # =============================================================================
    print("\n========================= Suddivisione Training/Test =========================")
    # Usiamo X_selected che contiene solo le feature dopo la rimozione VIF
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    print(f"Dimensioni X_train: {X_train.shape}")
    print(f"Dimensioni X_test: {X_test.shape}")
    print(f"Dimensioni y_train: {y_train.shape}")
    print(f"Dimensioni y_test: {y_test.shape}")
    print("========================= Fine Suddivisione Training/Test =========================")


    # =============================================================================
    # Standardizzazione delle feature (sui dati di training e test selezionati)
    # =============================================================================
    print("\n========================= Standardizzazione delle feature =========================")
    scaler = StandardScaler()  # Normalizza i dati con media=0 e deviazione standard=1

    # Applica fit_transform solo sul training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Applica solo transform sul test set (usa i parametri calcolati sul training)
    X_test_scaled = scaler.transform(X_test)

    print("Standardizzazione completata sulle features selezionate e suddivise.")
    print("========================= Fine Standardizzazione =========================")
    return X_train_scaled, X_test_scaled, y_train, y_test

