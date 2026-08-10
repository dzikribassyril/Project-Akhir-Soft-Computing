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
    |-- experiment_cnn_embedding_anfis.ipynb
    `-- Project_Soft_Computing_v2.ipynb
```

## Konvensi Pengembangan

- `data/raw/` menyimpan dataset asli dan anotasi COCO final.
- `data/processed/` menyimpan dataset tabular hasil ekstraksi fitur.
- `models/` menyimpan model terlatih, scaler, PCA, atau checkpoint eksperimen.
- `notebooks/` menyimpan notebook eksplorasi dan eksperimen.
- `docs/` menyimpan dokumentasi alur kerja, catatan eksperimen, dan ringkasan hasil.
- `assets/` menyimpan gambar pendukung untuk laporan, presentasi, atau aplikasi.

## Catatan Dependensi

Notebook `experiment_cnn_embedding_anfis.ipynb` membutuhkan `torchvision` untuk memuat CNN pretrained. Default eksperimen optimized memakai EfficientNet-B0 embedding dan ANFIS sebagai classifier akhir.

Untuk eksperimen baru, gunakan nama yang eksplisit, misalnya:

```text
notebooks/experiment_manual_features.ipynb
notebooks/experiment_cnn_embedding_anfis.ipynb
models/anfis_manual_features.pth
models/model_efficientnet_b0_original-hflip-vflip_grid256_anfis.pth
data/processed/features_manual_v2.csv
data/processed/features_efficientnet_b0_original-hflip-vflip_grid256_best_pca_anfis.csv
```

## Alur Pipeline

Pipeline utama mengikuti alur: citra UAV + anotasi COCO → pemotongan grid `256 x 256` (sliding window non-overlapping) → ekstraksi fitur per grid → dataset tabular → normalisasi Min-Max & stratified split → inisialisasi ANFIS via K-Means → pelatihan → evaluasi & plot membership function → inferensi peta panas.

### Fitur Input (X₁–X₆)

Setiap grid diekstrak fiturnya secara independen tanpa menggunakan masker ground truth:

| Fitur | Deskripsi |
|---|---|
| X₁–X₅ | Tekstur GLCM (entropy, contrast, homogeneity, energy, correlation) pada jarak d=1, dirata-ratakan untuk 4 arah (0°, 45°, 90°, 135°) |
| X₆ | Indeks vegetasi Excess Green (ExG = 2G − R − B), rata-rata seluruh piksel grid |

### Pelabelan Target

Persentase luas sampah dihitung dari piksel putih masker biner per grid:

| Kelas | Kepadatan Sampah |
|---|---|
| 0 — Aman/Bersih | < 5% |
| 1 — Tersebar Ringan | 5% – 40% |
| 2 — Klaster Kritis/Padat | > 40% |

Dataset akhir berisi **4.684 grid** (Kelas 0: 51.22%, Kelas 1: 23.91%, Kelas 2: 24.87%) dengan split latih 68% / validasi 12% / uji 20%.

### Arsitektur Model

Dua pendekatan yang dibandingkan:

| Jalur | Deskripsi |
|---|---|
| Manual Feature Engineering + ANFIS | GLCM + ExG sebagai input ANFIS yang diinisialisasi via K-Means |
| CNN + ANFIS | Embedding EfficientNet-B0 (fine-tuned, grid 256) → PCA + scaler → ANFIS |

## Aplikasi Streamlit

Aplikasi `app.py` menampilkan perbandingan prediksi kedua model pada citra UAV yang di-upload, lengkap dengan peta panas (heatmap) tingkat kepadatan sampah per grid.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runtime: Python 3.10 (sesuai `runtime.txt`). Dependensi utama: `streamlit`, `scikit-image` (GLCM), `scikit-learn`, `torch`/`torchvision` (EfficientNet-B0), `opencv-python-headless`, `pycocotools`, `seaborn`.
