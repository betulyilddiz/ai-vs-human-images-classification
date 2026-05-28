import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "ciktilar"

results = pd.read_csv(f"{OUTPUT_DIR}/model_sonuclari.csv")
cv_results = pd.read_csv(f"{OUTPUT_DIR}/cv_sonuclari.csv")
cm = np.load(f"{OUTPUT_DIR}/best_confusion_matrix.npy")
roc_curve_data = pd.read_csv(f"{OUTPUT_DIR}/best_roc_curve.csv")

# Accuracy grafiği
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["Accuracy"])
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
plt.bar(results["Model"], results["F1-score"])
plt.title("Model F1-score Comparison")
plt.xlabel("Models")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/f1_comparison.png")
plt.close()

# ROC-AUC grafiği
plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["ROC-AUC"])
plt.title("Model ROC-AUC Comparison")
plt.xlabel("Models")
plt.ylabel("ROC-AUC")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_auc_comparison.png")
plt.close()

# 10-Fold CV F1 Mean grafiği
plt.figure(figsize=(8, 5))
plt.bar(cv_results["Model"], cv_results["CV_F1_Mean"])
plt.title("10-Fold Cross Validation F1 Mean")
plt.xlabel("Models")
plt.ylabel("CV F1 Mean")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cv_f1_comparison.png")
plt.close()

# ROC Curve grafiği
plt.figure(figsize=(6, 5))
plt.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], label="Best Model ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("ROC Curve - Best Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/best_model_roc_curve.png")
plt.close()

# Confusion Matrix grafiği
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

print("Tüm değerlendirme grafikleri ciktilar klasörüne kaydedildi.")
print("- accuracy_comparison.png")
print("- f1_comparison.png")
print("- roc_auc_comparison.png")
print("- cv_f1_comparison.png")
print("- best_model_roc_curve.png")
print("- best_model_confusion_matrix.png")