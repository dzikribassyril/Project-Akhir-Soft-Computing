# Project Soft Computing - Deteksi Sampah UAV

Project ini berisi pipeline deteksi tingkat kepadatan sampah dari citra UAV menggunakan ekstraksi fitur visual dan model ANFIS.

## Struktur Folder

```text
.
|-- assets/
|   `-- TPSA.jpg
|-- data/
|   |-- raw/
|   |   |-- annotations/
|   |   |   `-- dronewaste_v2.0.json
|   |   `-- images/
|   `-- processed/
|       `-- dataset_fitur_sampah.csv
|-- docs/
|   `-- flow_proyek_anfis.md
|-- models/
|   |-- model_manual_feature.pth
|   `-- scaler_params.pkl
`-- notebooks/
    `-- Project_Soft_Computing_v2.ipynb
```

## Konvensi Pengembangan

- `data/raw/` menyimpan dataset asli dan anotasi COCO final.
- `data/processed/` menyimpan dataset tabular hasil ekstraksi fitur.
- `models/` menyimpan model terlatih, scaler, PCA, atau checkpoint eksperimen.
- `notebooks/` menyimpan notebook eksplorasi dan eksperimen.
- `docs/` menyimpan dokumentasi alur kerja, catatan eksperimen, dan ringkasan hasil.
- `assets/` menyimpan gambar pendukung untuk laporan, presentasi, atau aplikasi.

Untuk eksperimen baru, gunakan nama yang eksplisit, misalnya:

```text
notebooks/experiment_manual_features.ipynb
notebooks/experiment_cnn_embedding_anfis.ipynb
models/anfis_manual_features.pth
models/anfis_resnet18_pca.pth
data/processed/features_manual_v2.csv
data/processed/features_resnet18_pca.csv
```
