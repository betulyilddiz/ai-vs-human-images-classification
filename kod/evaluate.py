import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "ciktilar"

results_path = f"{OUTPUT_DIR}/model_sonuclari.csv"
cm_path = f"{OUTPUT_DIR}/best_confusion_matrix.npy"

results = pd.read_csv(results_path)

models = results["Model"]
accuracy = results["Accuracy"]
f1_score = results["F1-score"]

# Accuracy grafiği
plt.figure(figsize=(8, 5))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")
plt.close()

# F1-score grafiği
plt.figure(figsize=(8, 5))
plt.bar(models, f1_score)
plt.title("Model F1-score Comparison")
plt.xlabel("Models")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/f1_comparison.png")
plt.close()

# En iyi model confusion matrix
cm = np.load(cm_path)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Best Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Human/Real", "AI"])
plt.yticks([0, 1], ["Human/Real", "AI"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/best_model_confusion_matrix.png")
plt.close()

print("Grafikler ciktilar klasörüne kaydedildi.")