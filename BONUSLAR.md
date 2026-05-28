# BLM308 Veri Madenciliği Final Projesi — Bonus Çalışmalar

Bu proje kapsamında temel proje gereksinimlerine ek olarak çeşitli bonus çalışmalar gerçekleştirilmiştir. Bonus çalışmalar sayesinde proje yalnızca temel sınıflandırma uygulaması olmaktan çıkarılarak daha kapsamlı bir veri madenciliği ve makine öğrenmesi çalışmasına dönüştürülmüştür.

---

# Gerçekleştirilen Bonuslar

## 1. GitHub Repo Kullanımı

Proje kaynak kodları, rapor dosyaları ve proje çıktıları GitHub reposu üzerinden düzenli şekilde paylaşılmıştır. Böylece proje sürüm kontrolü sağlanmış ve proje yapısının daha profesyonel şekilde yönetilmesi amaçlanmıştır.

---

## 2. README.md Dosyası Hazırlanması

Projeye ait:

* kullanım açıklamaları,
* kurulum adımları,
* proje klasör yapısı,
* model performansları,
* çalıştırma komutları

gibi bilgileri içeren detaylı README.md dosyası hazırlanmıştır.

---

## 3. Demo Video Hazırlanması

Projenin:

* çalışma mantığını,
* veri ön işleme sürecini,
* model eğitim aşamalarını,
* grafik çıktıları ve sonuç analizlerini

gösteren demo video hazırlanmıştır.

Dosya:

* demo_video.mp4

---

## 4. Ensemble Learning Kullanımı

Projede yalnızca temel sınıflandırma algoritmaları değil, ensemble learning tabanlı yöntemler de kullanılmıştır.

Kullanılan ensemble modeller:

* Random Forest
* Extra Trees

Bu modeller sayesinde daha güçlü genelleme performansı elde edilmiştir.

---

## 5. ROC-AUC Analizi

Projede yalnızca Accuracy metriği kullanılmamış, modellerin sınıfları ayırt edebilme performanslarını değerlendirmek amacıyla ROC-AUC analizleri de gerçekleştirilmiştir.

Gerçekleştirilen çalışmalar:

* ROC eğrisi oluşturulması
* ROC-AUC skorlarının hesaplanması
* Modeller arası ROC-AUC karşılaştırması

---

## 6. Cross Validation Kullanımı

Model performanslarının daha güvenilir şekilde değerlendirilmesi amacıyla 10-Fold Cross Validation uygulanmıştır.

Bu sayede:

* modellerin farklı veri bölmeleri üzerindeki performansları analiz edilmiş,
* performans sonuçlarının daha güvenilir olması sağlanmıştır.

---

## 7. GridSearchCV ile Hiperparametre Optimizasyonu

Makine öğrenmesi modellerinin performansını artırmak amacıyla GridSearchCV yöntemi kullanılmıştır.

Optimize edilen parametreler:

* n_estimators
* max_iter
* criterion
* train/test split ayarları

Bu süreç sayesinde modeller için daha uygun hiperparametre değerleri belirlenmiştir.

---

## 8. Açıklanabilir Yapay Zekâ (XAI) Analizi

Projede model yorumlanabilirliğini artırmak amacıyla SHAP (SHapley Additive exPlanations) yöntemi kullanılmıştır.

Gerçekleştirilen çalışmalar:

* SHAP summary plot oluşturulması
* SHAP bar plot oluşturulması
* Model kararlarını etkileyen özelliklerin analiz edilmesi

Bu analizler sayesinde modelin yalnızca yüksek performans göstermesi değil, aynı zamanda yorumlanabilir olması sağlanmıştır.

---

## 9. Gelişmiş Görselleştirme ve Analiz

Proje kapsamında çeşitli grafiksel analizler gerçekleştirilmiştir.

Oluşturulan görseller:

* Accuracy karşılaştırma grafikleri
* F1-score karşılaştırma grafikleri
* ROC-AUC karşılaştırma grafikleri
* Confusion Matrix analizleri
* ROC eğrileri
* SHAP grafikleri
* Keşifsel Veri Analizi (EDA) görselleri

---

## 10. Python Tabanlı Uçtan Uca Veri Madenciliği Süreci

Projede veri ön işleme, modelleme, hiperparametre optimizasyonu, performans değerlendirme ve görselleştirme işlemleri tamamen Python ortamında gerçekleştirilmiştir.

Kullanılan kütüphaneler:

* pandas
* NumPy
* scikit-learn
* matplotlib
* Pillow (PIL)
* SHAP

---

## 11. CRISP-DM Yaklaşımı

Projede veri madenciliği süreçleri CRISP-DM yaklaşımına uygun şekilde uygulanmıştır.

Uygulanan aşamalar:

1. İş Probleminin Anlaşılması
2. Veri Anlama
3. Veri Ön İşleme
4. Modelleme
5. Değerlendirme
6. Sonuç Analizi ve Raporlama

---

## 12. Confusion Matrix ve Performans Analizi

Makine öğrenmesi modelleri detaylı şekilde analiz edilmiş ve performans karşılaştırmaları gerçekleştirilmiştir.

Kullanılan performans metrikleri:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Ayrıca confusion matrix analizleri sayesinde yanlış sınıflandırılan örnekler detaylı şekilde incelenmiştir.

---

## 13. Akademik Raporlama ve Kaynakça Düzeni

Proje raporu akademik yazım kurallarına uygun şekilde hazırlanmış, kaynakça ve literatür bölümleri düzenlenmiştir.

Raporda:

* literatür taraması,
* veri analizi,
* modelleme,
* hiperparametre optimizasyonu,
* ROC analizi,
* SHAP tabanlı yorumlanabilirlik,
* sonuç ve tartışma

bölümleri detaylı şekilde açıklanmıştır.

---

## 14. Tekrar Üretilebilir (Reproducible) Deney Yapısı

Projede gerçekleştirilen deneylerin tekrar üretilebilir olması amacıyla:

* random_state parametreleri sabit tutulmuş,
* aynı veri bölmeleri kullanılmış,
* tüm çıktı dosyaları otomatik olarak kaydedilmiştir.

Bu yaklaşım sayesinde deney sonuçlarının tutarlılığı sağlanmıştır.
