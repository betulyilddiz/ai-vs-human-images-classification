import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = "veri/archive/train.csv"
IMAGE_FOLDER = "veri/archive/"
OUTPUT_DIR = "ciktilar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 3000 örnek
df = df.sample(n=3000, random_state=42)

# =========================
# SINIF DAĞILIMI
# =========================

counts = df["label"].value_counts()

labels = ["Human/Real", "AI Generated"]
values = [counts[0], counts[1]]

plt.figure(figsize=(6, 5))
plt.bar(labels, values)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.close()

# =========================
# ÖRNEK HUMAN GÖRSELLER
# =========================

human_df = df[df["label"] == 0].sample(5, random_state=42)

plt.figure(figsize=(15, 3))

for i, (_, row) in enumerate(human_df.iterrows()):
    img_path = IMAGE_FOLDER + row["file_name"]

    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))

    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle("Sample Human/Real Images")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/sample_human_images.png")
plt.close()

# =========================
# ÖRNEK AI GÖRSELLER
# =========================

ai_df = df[df["label"] == 1].sample(5, random_state=42)

plt.figure(figsize=(15, 3))

for i, (_, row) in enumerate(ai_df.iterrows()):
    img_path = IMAGE_FOLDER + row["file_name"]

    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))

    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.suptitle("Sample AI Generated Images")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/sample_ai_images.png")
plt.close()

# =========================
# RGB HISTOGRAM
# =========================

sample_images = df.sample(50, random_state=42)

r_values = []
g_values = []
b_values = []

for _, row in sample_images.iterrows():

    try:
        img_path = IMAGE_FOLDER + row["file_name"]

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

        r_values.extend(arr[:, :, 0].flatten())
        g_values.extend(arr[:, :, 1].flatten())
        b_values.extend(arr[:, :, 2].flatten())

    except:
        pass

plt.figure(figsize=(8, 5))

plt.hist(r_values, bins=50, alpha=0.5, label="Red")
plt.hist(g_values, bins=50, alpha=0.5, label="Green")
plt.hist(b_values, bins=50, alpha=0.5, label="Blue")

plt.title("RGB Pixel Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rgb_histogram.png")
plt.close()

# =========================
# PIXEL INTENSITY HISTOGRAM
# =========================

gray_values = []

for _, row in sample_images.iterrows():

    try:
        img_path = IMAGE_FOLDER + row["file_name"]

        img = Image.open(img_path).convert("L")
        arr = np.array(img)

        gray_values.extend(arr.flatten())

    except:
        pass

plt.figure(figsize=(8, 5))

plt.hist(gray_values, bins=50)

plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pixel_intensity_histogram.png")
plt.close()

print("EDA grafikleri başarıyla oluşturuldu.")
print("- class_distribution.png")
print("- sample_human_images.png")
print("- sample_ai_images.png")
print("- rgb_histogram.png")
print("- pixel_intensity_histogram.png")