# BLM308 Veri Madenciliği Final Projesi

## Makine Öğrenmesi Algoritmaları ile AI Üretimi ve Gerçek Görsellerin Sınıflandırılması

Bu projede yapay zekâ tarafından üretilen görseller ile gerçek görsellerin makine öğrenmesi algoritmaları kullanılarak sınıflandırılması amaçlanmıştır. Proje, BLM308 Veri Madenciliği dersi final projesi kapsamında hazırlanmıştır.

---

## Proje Amacı

Üretken yapay zekâ teknolojilerinin gelişmesiyle birlikte dijital ortamlarda gerçekçi sahte görsellerin sayısı artmıştır. Bu projede gerçek görseller ile yapay zekâ üretimi görsellerin otomatik olarak ayırt edilmesi hedeflenmiştir.

---

## Kullanılan Veri Seti

Veri seti: Kaggle - AI vs Human Generated Images Dataset

Veri seti içerisinde:
- Human/Real görseller
- AI Generated görseller

bulunmaktadır.

Çalışmada işlem maliyetini azaltmak amacıyla veri setinden rastgele seçilen 3000 görsel kullanılmıştır.

---

## Kullanılan Teknolojiler

- Python 3.12
- pandas
- numpy
- scikit-learn
- matplotlib
- Pillow (PIL)
- Visual Studio Code

---

## Kullanılan Modeller

- Logistic Regression
- Random Forest
- Extra Trees

---

## Ön İşleme Adımları

- Görseller 64x64 piksel boyutuna dönüştürülmüştür.
- Görseller RGB formatına çevrilmiştir.
- Piksel değerleri 0-1 aralığına normalize edilmiştir.
- Görseller flatten işlemi ile sayısal özellik vektörlerine dönüştürülmüştür.
- Veri seti %70 eğitim ve %30 test olarak ayrılmıştır.

---

## Proje Klasör Yapısı

```text
VeriMadenciligi_Final_Projesi/
│
├── rapor.docx
├── rapor.pdf
├── README.md
├── BONUSLAR.md
│
├── kod/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── eda.py
│
├── ciktilar/
│   ├── class_distribution.png
│   ├── accuracy_comparison.png
│   ├── f1_comparison.png
│   ├── best_confusion_matrix.npy
│   ├── best_model_confusion_matrix.png
│   └── model_sonuclari.csv
│
└── veri/
    └── README.md
```

---

## Kurulum

Gerekli Python kütüphanelerini yüklemek için aşağıdaki komut kullanılmalıdır:

```bash
pip install pandas numpy scikit-learn matplotlib pillow
```

---

## Projeyi Çalıştırma

### Veri Ön İşleme

```bash
python kod/preprocess.py
```

### Model Eğitimi

```bash
python kod/train.py
```

### Model Değerlendirme

```bash
python kod/evaluate.py
```

### Keşifsel Veri Analizi (EDA)

```bash
python kod/eda.py
```

---

## Çıktılar

Oluşturulan grafikler ve sonuç dosyaları `ciktilar/` klasörüne kaydedilmektedir.

Başlıca çıktılar:

- class_distribution.png
- accuracy_comparison.png
- f1_comparison.png
- best_model_confusion_matrix.png
- model_sonuclari.csv

---

## Grafikler

### Sınıf Dağılımı

Veri setindeki Human/Real ve AI Generated görsellerin dağılımı grafik ile gösterilmiştir.

### Accuracy Karşılaştırması

Makine öğrenmesi modellerinin doğruluk oranları karşılaştırılmıştır.

### F1-score Karşılaştırması

Modellerin F1-score performansları karşılaştırılmıştır.

### Confusion Matrix

En başarılı model olan Random Forest algoritmasına ait confusion matrix çıktısı oluşturulmuştur.

---

## Performans Değerlendirme Metrikleri

Projede aşağıdaki performans metrikleri kullanılmıştır:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Model Performansları

| Model | Accuracy | Precision | Recall | F1-score | Süre (s) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6056 | 0.6426 | 0.4434 | 0.5248 | 7.84 |
| Random Forest | 0.8900 | 0.8595 | 0.9276 | 0.8923 | 14.38 |
| Extra Trees | 0.8789 | 0.8675 | 0.8891 | 0.8782 | 4.69 |

En başarılı model Random Forest olmuştur.

---

## CRISP-DM Süreci

Bu proje kapsamında veri madenciliği süreçleri CRISP-DM yaklaşımına uygun şekilde gerçekleştirilmiştir.

Uygulanan aşamalar:

1. Veri Anlama
2. Veri Ön İşleme
3. Modelleme
4. Değerlendirme
5. Sonuç Analizi ve Raporlama

---

## Bonus Çalışmalar

Bu proje kapsamında aşağıdaki ek çalışmalar gerçekleştirilmiştir:

- GitHub repo hazırlanması
- README.md dosyası hazırlanması
- Demo video hazırlanması
- Ensemble learning yöntemlerinin kullanılması
- Accuracy ve F1-score grafiklerinin oluşturulması
- Confusion Matrix analizi
- Keşifsel Veri Analizi (EDA)

Detaylı açıklamalar BONUSLAR.md dosyasında belirtilmiştir.

---

## GitHub Repo

Proje GitHub üzerinde paylaşılmıştır.

---

## Akademik Bilgilendirme

Bu proje İstanbul Gedik Üniversitesi BLM308 Veri Madenciliği dersi final projesi kapsamında hazırlanmıştır.

---

## Yapay Zekâ Kullanımı

Bu proje sürecinde ChatGPT; akademik yazım düzeni, rapor formatı, teknik açıklamaların geliştirilmesi ve bazı kod düzenleme işlemlerinde yardımcı araç olarak kullanılmıştır. Tüm kod çalıştırmaları, deney sonuçları ve proje çıktıları proje sahibi tarafından kontrol edilmiştir.

---

## Geliştirici

Betül Yıldız  
İstanbul Gedik Üniversitesi  
Bilgisayar Mühendisliği