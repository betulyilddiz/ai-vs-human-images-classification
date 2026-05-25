import pandas as pd

CSV_PATH = "veri/archive/train.csv"

df = pd.read_csv(CSV_PATH)

print("İlk 5 veri:")
print(df.head())

print("\nVeri boyutu:")
print(df.shape)

print("\nSınıf dağılımı:")
print(df["label"].value_counts())

print("\nEksik veri kontrolü:")
print(df.isnull().sum())

print("\nÖn işleme adımları:")
print("- Görseller 64x64 boyutuna getirildi")
print("- Tüm görseller RGB formatına dönüştürüldü")
print("- Piksel değerleri 0-255 aralığından 0-1 aralığına normalize edildi")
print("- Görseller flatten işlemi ile tek boyutlu özellik vektörlerine dönüştürüldü")

print("\nDeney Bilgisi:")
print("- Model eğitiminde veri setinden rastgele 3000 görsel kullanıldı")
print("- Eğitim/test ayrımı %70 eğitim, %30 test olarak yapıldı")
print("- Label 0: Human/Real")
print("- Label 1: AI Generated")