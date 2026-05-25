import os
import time
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


CSV_PATH = "veri/archive/train.csv"
IMAGE_SIZE = (64, 64)
LIMIT_TOTAL = 3000
OUTPUT_DIR = "ciktilar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Veri seti büyük olduğu için deneylerde rastgele seçilen 3000 görsel kullanılmıştır.
df = df.sample(n=LIMIT_TOTAL, random_state=42)

X = []
y = []

print("Görseller yükleniyor...")

for index, row in df.iterrows():
    img_path = "veri/archive/" + row["file_name"]
    label = row["label"]

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        arr = np.array(img) / 255.0
        X.append(arr.flatten())
        y.append(label)

    except Exception as e:
        print("Okunamayan görsel:", img_path, "| Hata:", e)

X = np.array(X)
y = np.array(y)

print("Toplam veri:", len(X))
print("Özellik sayısı:", X.shape[1])
print("Sınıf dağılımı:")
print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42)
}

results = []
best_model_name = None
best_f1 = -1
best_cm = None

print("\nModel sonuçları:\n")

for name, model in models.items():
    start = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end = time.time()

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    duration = end - start

    print("Model:", name)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion Matrix:")
    print(cm)
    print("Süre:", round(duration, 2), "saniye")
    print("-" * 40)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(pre, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "Sure_saniye": round(duration, 2)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_cm = cm

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/model_sonuclari.csv", index=False)

np.save(f"{OUTPUT_DIR}/best_confusion_matrix.npy", best_cm)

print("\nSonuçlar ciktilar/model_sonuclari.csv dosyasına kaydedildi.")
print("En başarılı model:", best_model_name)