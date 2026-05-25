# Veri Seti

Bu projede Kaggle platformunda yayınlanan **AI vs Human Generated Images Dataset** veri seti kullanılmıştır.

GitHub dosya boyutu sınırları nedeniyle veri seti repoya yüklenmemiştir.

---

## Veri Seti Linki

Kaggle Dataset:

https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset

---

## Veri Seti İçeriği

Veri seti içerisinde:

- Human / Real Images
- AI Generated Images

bulunmaktadır.

Projede işlem maliyetini azaltmak amacıyla veri setinden rastgele seçilen yaklaşık 3000 görsel kullanılmıştır.

---

## Veri Setini İndirme

1. Yukarıdaki Kaggle linkine gidin.
2. Veri setini indir butonuna tıklayın.
3. ZIP dosyasını çıkartın.
4. `archive` klasörünü aşağıdaki yapıya göre yerleştirin:

```text
veri/
└── archive/
    ├── train_data/
    ├── test_data_v2/
    ├── train.csv
    └── test.csv