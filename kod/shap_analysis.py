import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "ciktilar"
MODEL_PATH = f"{OUTPUT_DIR}/best_model.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("En iyi model yükleniyor...")
model = joblib.load(MODEL_PATH)

print("Veri dosyası yükleniyor...")
X_sample_path = f"{OUTPUT_DIR}/shap_sample_X.npy"

if not os.path.exists(X_sample_path):
    print("SHAP için örnek veri bulunamadı.")
    print("Bu nedenle train.py içine SHAP örnek verisi kaydetme adımı eklenmeli.")
    exit()

X_sample = np.load(X_sample_path)

print("SHAP analizi başlıyor...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Binary classification için class 1 değerlerini al
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

plt.figure()
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(
    shap_values_to_plot,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP grafikleri oluşturuldu:")
print("- shap_summary.png")
print("- shap_bar.png")