# Veri Seti

Bu projede Kaggle platformunda yayınlanan **AI vs Human Generated Images Dataset** veri seti kullanılmıştır.

Veri seti boyutu oldukça büyük olduğu için GitHub depolama limitlerini aşmamak amacıyla veri dosyaları repoya dahil edilmemiştir. Projeyi çalıştırabilmek için veri setinin aşağıdaki bağlantı üzerinden manuel olarak indirilmesi gerekmektedir.

---

## Veri Seti Linki

Kaggle Dataset:

https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

---

## Veri Seti İçeriği

Veri seti içerisinde aşağıdaki sınıflar bulunmaktadır:

* Human / Real Images
* AI Generated Images

Çalışmada işlem maliyetini azaltmak ve deney süreçlerini daha kontrollü şekilde gerçekleştirebilmek amacıyla veri setinden rastgele seçilen yaklaşık 3000 görsel kullanılmıştır.

---

## Veri Setini İndirme ve Yerleştirme

1. Yukarıdaki Kaggle bağlantısına gidin.
2. Veri setini `.zip` formatında indirin.
3. ZIP dosyasını çıkartın.
4. Çıkartılan `archive` klasörünü aşağıdaki dizin yapısına göre proje içerisine yerleştirin:

```text
veri/
└── archive/
    ├── train_data/
    ├── test_data_v2/
    ├── train.csv
    └── test.csv
```

Veri seti doğru şekilde yerleştirildikten sonra proje scriptleri sorunsuz şekilde çalıştırılabilir.
