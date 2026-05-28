import os
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

CSV_PATH = "veri/archive/train.csv"
IMAGE_SIZE = (64, 64)
LIMIT_TOTAL = 3000
OUTPUT_DIR = "ciktilar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.sample(n=LIMIT_TOTAL, random_state=42)

X = []
y = []

print("Görseller yükleniyor...")

for _, row in df.iterrows():
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

print("\nToplam veri:", len(X))
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
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42)
}

param_grids = {
    "Logistic Regression": {
        "C": [0.1, 1, 10]
    },
    "Random Forest": {
        "n_estimators": [100, 150],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    },
    "Extra Trees": {
        "n_estimators": [100, 150],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []
cv_results = []
best_params_list = []

best_model_name = None
best_model = None
best_f1 = -1
best_cm = None
best_fpr = None
best_tpr = None
best_auc = None

print("\nModel eğitimi, 10-Fold CV ve GridSearchCV başlıyor...\n")

for name, model in models.items():
    print("=" * 50)
    print("Model:", name)

    start = time.time()

    print("10-Fold Cross Validation yapılıyor...")
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    print("CV F1 Scores:", np.round(cv_scores, 4))
    print("CV Mean F1:", round(cv_scores.mean(), 4))
    print("CV Std:", round(cv_scores.std(), 4))

    print("GridSearchCV yapılıyor...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    tuned_model = grid.best_estimator_
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    end = time.time()
    duration = end - start

    print("En iyi parametreler:", grid.best_params_)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall:", round(rec, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:")
    print(cm)
    print("Süre:", round(duration, 2), "saniye")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(pre, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Sure_saniye": round(duration, 2)
    })

    cv_results.append({
        "Model": name,
        "CV_F1_Mean": round(cv_scores.mean(), 4),
        "CV_F1_Std": round(cv_scores.std(), 4)
    })

    best_params_list.append({
        "Model": name,
        "Best_Params": str(grid.best_params_)
    })

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    }).to_csv(f"{OUTPUT_DIR}/roc_{name.replace(' ', '_').lower()}.csv", index=False)

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = tuned_model
        best_cm = cm
        best_fpr = fpr
        best_tpr = tpr
        best_auc = auc

results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)
best_params_df = pd.DataFrame(best_params_list)

results_df.to_csv(f"{OUTPUT_DIR}/model_sonuclari.csv", index=False)
cv_df.to_csv(f"{OUTPUT_DIR}/cv_sonuclari.csv", index=False)
best_params_df.to_csv(f"{OUTPUT_DIR}/best_parameters.csv", index=False)

np.save(f"{OUTPUT_DIR}/best_confusion_matrix.npy", best_cm)
np.save(f"{OUTPUT_DIR}/shap_sample_X.npy", X_test[:100])

pd.DataFrame({
    "fpr": best_fpr,
    "tpr": best_tpr
}).to_csv(f"{OUTPUT_DIR}/best_roc_curve.csv", index=False)

joblib.dump(best_model, f"{OUTPUT_DIR}/best_model.pkl")

with open(f"{OUTPUT_DIR}/best_model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"En başarılı model: {best_model_name}\n")
    f.write(f"En iyi F1-score: {round(best_f1, 4)}\n")
    f.write(f"ROC-AUC: {round(best_auc, 4)}\n")

print("\nTüm sonuçlar ciktilar klasörüne kaydedildi.")
print("En başarılı model:", best_model_name)
print("Best model kaydedildi:", f"{OUTPUT_DIR}/best_model.pkl")