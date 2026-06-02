# ============================================================
#  data_loader.py  –  Veri yükleme ve ön işleme
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_SEED, FEATURE_NAMES


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    CSV'yi yükler, özellik matrisini ve etiket vektörünü döndürür.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,)
    feature_names : list[str]
    """
    df = pd.read_csv(DATA_PATH)

    # Eksik değer kontrolü
    if df.isnull().any().any():
        print("[UYARI] Eksik değerler bulundu. Satırlar düşürülüyor...")
        df = df.dropna()

    # Hedef değişkeni ikili sınıfa normalize et (bazı veri setlerinde 0-4 arası)
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df[TARGET_COL].values.astype(int)

    print(f"[VERİ] {X.shape[0]} örnek | {X.shape[1]} özellik")
    print(f"[VERİ] Sınıf dağılımı → 0: {(y==0).sum()}  1: {(y==1).sum()}")
    return X, y, FEATURE_NAMES


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Ölçekleme + eğitim/test bölme.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"[ÖN İŞLEME] Eğitim: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler
