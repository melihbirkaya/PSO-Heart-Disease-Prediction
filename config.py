# ============================================================
#  config.py  –  Merkezi parametre dosyası
#  PSO Tabanlı Özellik Seçimi | Kalp Hastalığı Tahmini
# ============================================================

import os

# ── Dizinler ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "heart.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ── Veri Seti ─────────────────────────────────────────────────
TARGET_COL  = "target"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── PSO Hiperparametreleri ────────────────────────────────────
PSO = {
    "n_particles"  : 30,    # Sürü büyüklüğü
    "n_iterations" : 50,    # Maksimum iterasyon
    "w"            : 0.6,   # Atalet katsayısı
    "c1"           : 2.0,   # Bilişsel bileşen
    "c2"           : 2.0,   # Sosyal bileşen
    "vel_clamp"    : 4.0,   # Hız kırpma sınırı  ±vel_clamp
    "threshold"    : 0.5,   # Özellik seçim eşiği
    "penalty"      : 0.01,  # Özellik sayısı ceza katsayısı
}

# ── Model ─────────────────────────────────────────────────────
MODEL = {
    "max_iter"     : 1000,
    "random_state" : RANDOM_SEED,
}

# ── CV ────────────────────────────────────────────────────────
CV_FOLDS = 5
