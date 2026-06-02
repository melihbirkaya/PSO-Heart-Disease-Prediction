# PSO Tabanlı Özellik Seçimi ile Kalp Hastalığı Tahmini

> Optimizasyon Dersi Bildirisi — Kırıkkale Üniversitesi, Elektrik-Elektronik Mühendisliği  
> **Yazar:** Melih Alperen Birkaya

---

## Proje Özeti

Bu projede UCI Heart Disease veri seti üzerinde **Parçacık Sürü Optimizasyonu (PSO)** tabanlı özellik seçimi uygulanmış, seçilen özellik alt kümesiyle Logistic Regression sınıflandırıcısının performansı baseline modelle karşılaştırılmıştır.

| Metrik | Baseline (13 özellik) | PSO (7 özellik) |
|---|---|---|
| Test Doğruluğu | %80.33 | **%88.52** |
| ROC-AUC | — | — |
| 5-Fold CV | %83.49 | %77.54 |

---

## Klasör Yapısı

```
pso_heart/
├── data/
│   └── heart.csv          # UCI Heart Disease veri seti
├── src/
│   ├── config.py          # Tüm hiperparametreler
│   ├── data_loader.py     # Veri yükleme ve ön işleme
│   ├── pso.py             # PSO algoritması
│   ├── evaluate.py        # Model değerlendirme
│   ├── visualize.py       # Grafikler
│   └── main.py            # Ana çalıştırma dosyası
├── outputs/               # Grafikler buraya kaydedilir
├── requirements.txt
└── README.md
```

---

## Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Çalıştır
cd src
python main.py
```

Çıktılar `outputs/` klasörüne kaydedilir.

---

## Yöntem

**PSO Özellik Seçimi:**
- Her parçacık 13 boyutlu bir pozisyon vektörü taşır (her boyut bir özelliğe karşılık gelir)
- Pozisyon değeri > 0.5 olan boyutlar "seçildi" anlamına gelir
- Fitness = Logistic Regression doğruluğu − α × (seçilen özellik oranı)

**PSO Parametreleri:**

| Parametre | Değer |
|---|---|
| Parçacık sayısı | 30 |
| İterasyon | 50 |
| Atalet (w) | 0.6 |
| c₁ (bilişsel) | 2.0 |
| c₂ (sosyal) | 2.0 |

---

## Bağımlılıklar

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Kaynak

- Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization. IEEE ICNN.
- UCI Machine Learning Repository — Heart Disease Dataset
