import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("veri/archive/train.csv")

df = df.sample(n=3000, random_state=42)

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
plt.savefig("ciktilar/class_distribution.png")
plt.close()

print("Class distribution grafiği oluşturuldu.")