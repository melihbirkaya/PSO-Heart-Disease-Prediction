import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Veri setini yükle
data = pd.read_csv("heart.csv")

# Özellikler ve hedef değişken
X = data.drop("target", axis=1)
y = data["target"]

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model oluştur
model = LogisticRegression(max_iter=1000)

# Eğit
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Sonuç
accuracy = accuracy_score(y_test, y_pred)

print("Model Doğruluğu:", accuracy)
