# BLM308 Veri Madenciliği Final Projesi

# Makine Öğrenmesi Algoritmaları ile AI Üretimi ve Gerçek Görsellerin Sınıflandırılması

Bu projede yapay zekâ tarafından üretilen görseller ile gerçek görsellerin makine öğrenmesi algoritmaları kullanılarak sınıflandırılması amaçlanmıştır. Proje, İstanbul Gedik Üniversitesi BLM308 Veri Madenciliği dersi final projesi kapsamında hazırlanmıştır.

---

# Proje Amacı

Üretken yapay zekâ teknolojilerinin gelişmesiyle birlikte dijital ortamlarda gerçekçi sahte görsellerin sayısı önemli ölçüde artmıştır. Özellikle AI tabanlı görsel üretim sistemleri sayesinde insanlar tarafından oluşturulmamış içerikler oldukça gerçekçi hale gelmiştir.

Bu projede amaç:

* AI üretimi görseller ile gerçek görselleri otomatik olarak ayırt etmek,
* farklı makine öğrenmesi algoritmalarını karşılaştırmak,
* ensemble learning yöntemlerinin performansını değerlendirmek,
* görüntü sınıflandırma problemi üzerinde veri madenciliği sürecini uçtan uca uygulamaktır.

---

# Kullanılan Veri Seti

Veri seti:

Kaggle – AI vs Human Generated Images Dataset

Veri seti içerisinde:

* Human / Real görseller
* AI Generated görseller

bulunmaktadır.

Çalışmada işlem maliyetini azaltmak ve kontrollü deney ortamı oluşturmak amacıyla veri setinden rastgele seçilen 3000 görsel kullanılmıştır.

---

# Kullanılan Teknolojiler

* Python 3.12
* pandas
* NumPy
* scikit-learn
* matplotlib
* Pillow (PIL)
* SHAP
* Visual Studio Code

---

# Kullanılan Modeller

Bu çalışmada aşağıdaki makine öğrenmesi algoritmaları kullanılmıştır:

* Logistic Regression
* Random Forest
* Extra Trees

---

# Ön İşleme Adımları

Veri seti modelleme öncesinde çeşitli ön işleme süreçlerinden geçirilmiştir:

* Görseller 64x64 piksel boyutuna dönüştürülmüştür.
* Görseller RGB formatına çevrilmiştir.
* Piksel değerleri 0–1 aralığına normalize edilmiştir.
* Flatten işlemi uygulanarak görseller sayısal özellik vektörlerine dönüştürülmüştür.
* Veri seti %70 eğitim ve %30 test olacak şekilde ayrılmıştır.
* Stratified train-test split yöntemi kullanılmıştır.
* random_state=42 değeri ile deneylerin tekrar üretilebilir olması sağlanmıştır.

---

# Proje Klasör Yapısı

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
│   ├── roc_auc_comparison.png
│   ├── best_model_confusion_matrix.png
│   ├── best_model_roc_curve.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── model_sonuclari.csv
│   └── cv_sonuclari.csv
│
└── veri/
    └── README.md
```

---

# Kurulum

Gerekli Python kütüphanelerini yüklemek için:

```bash
pip install pandas numpy scikit-learn matplotlib pillow shap
```

---

# Projeyi Çalıştırma

## Veri Ön İşleme

```bash
python kod/preprocess.py
```

## Model Eğitimi

```bash
python kod/train.py
```

## Model Değerlendirme

```bash
python kod/evaluate.py
```

## Keşifsel Veri Analizi (EDA)

```bash
python kod/eda.py
```

---

# Çıktılar

Oluşturulan grafikler ve sonuç dosyaları `ciktilar/` klasörüne kaydedilmektedir.

Başlıca çıktılar:

* class_distribution.png
* accuracy_comparison.png
* f1_comparison.png
* roc_auc_comparison.png
* best_model_confusion_matrix.png
* best_model_roc_curve.png
* shap_summary.png
* shap_bar.png
* model_sonuclari.csv
* cv_sonuclari.csv

---

# Performans Değerlendirme Metrikleri

Projede aşağıdaki performans metrikleri kullanılmıştır:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Cross Validation

---

# Model Performansları

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC | Süre (s) |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -------: |
| Logistic Regression |   0.6000 |    0.6258 | 0.4615 |   0.5312 |  0.5987 |   219.54 |
| Random Forest       |   0.8989 |    0.8664 | 0.9389 |   0.9012 |  0.9552 |   355.73 |
| Extra Trees         |   0.8844 |    0.8658 | 0.9050 |   0.8850 |  0.9493 |   144.38 |

En başarılı model Random Forest olmuştur.

---

# Açıklanabilir Yapay Zekâ (XAI)

Bu projede model yorumlanabilirliğini artırmak amacıyla SHAP (SHapley Additive exPlanations) yöntemi kullanılmıştır.

SHAP analizleri sayesinde:

* model kararlarını etkileyen özellikler incelenmiş,
* önemli piksel bölgeleri analiz edilmiş,
* model davranışları yorumlanabilir hale getirilmiştir.

Kullanılan görseller:

* shap_summary.png
* shap_bar.png

---

# CRISP-DM Süreci

Bu proje kapsamında veri madenciliği süreçleri CRISP-DM yaklaşımına uygun şekilde gerçekleştirilmiştir.

Uygulanan aşamalar:

1. İş Probleminin Anlaşılması
2. Veri Anlama
3. Veri Ön İşleme
4. Modelleme
5. Değerlendirme
6. Sonuç Analizi ve Raporlama

---

# Bonus Çalışmalar

Bu proje kapsamında aşağıdaki ek çalışmalar gerçekleştirilmiştir:

* GitHub repo hazırlanması
* README.md dosyası hazırlanması
* Demo video hazırlanması
* Ensemble learning yöntemlerinin kullanılması
* ROC-AUC analizi
* SHAP tabanlı XAI analizi
* Confusion Matrix analizi
* Cross Validation uygulanması
* Keşifsel Veri Analizi (EDA)
* Grafiksel performans karşılaştırmaları

Detaylı açıklamalar BONUSLAR.md dosyasında belirtilmiştir.

---

# GitHub Repo

Proje GitHub üzerinde paylaşılmıştır.

---

# Akademik Bilgilendirme

Bu proje İstanbul Gedik Üniversitesi BLM308 Veri Madenciliği dersi final projesi kapsamında hazırlanmıştır.

---

# Yapay Zekâ Kullanımı

Bu proje sürecinde ChatGPT; akademik yazım düzeni, rapor formatı, teknik açıklamaların geliştirilmesi, kod düzenlemeleri ve dokümantasyon süreçlerinde yardımcı araç olarak kullanılmıştır. Tüm kod çalıştırmaları, deney sonuçları ve proje çıktıları proje sahibi tarafından kontrol edilmiştir.

---

# Geliştirici

Betül Yıldız
İstanbul Gedik Üniversitesi
Bilgisayar Mühendisliği
