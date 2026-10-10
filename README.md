# Tugas Pertemuan 6 MLOps — Model Experimentation & Tracking
**Nama:** Devi Rahayu  
**NIM:** 123450010  
**Kelas:** RA

Implementasi tugas berdasarkan instruksi slide Minggu 6: MLflow tracking server lokal (SQLite + artifact folder), subset Fashion-MNIST, baseline CNN, minimal 20 trial Optuna menggunakan nested runs, evaluasi konfigurasi terbaik dengan tiga seed, ablation minimal dua komponen, Model Registry dengan alias `@champion`, serta versioning dataset dengan DVC.

## 1. Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install dvc
```

## 2. Jalankan MLflow server (Terminal 1)
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
Biarkan terminal tetap berjalan. UI: http://127.0.0.1:5000

## 3. Siapkan dataset (Terminal 2)
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m src.main prepare
```
Dataset subset disimpan lokal di `data/fashion_mnist_subset.npz`. Unduhan Fashion-MNIST pertama kali memerlukan internet.

## 4. Versioning data dengan DVC
```powershell
dvc init
dvc add data/fashion_mnist_subset.npz
dvc remote add -d localstore ../dvc-store
dvc push
```
Jika repository Git belum dibuat:
```powershell
git init
git add .
git commit -m "Initial MLOps experiment tracking project"
```
Untuk praktik kelas, remote DVC lokal cukup sebagai demo versioning; bukan penyimpanan cloud.

## 5. Jalankan eksperimen secara berurutan
Gunakan Terminal 2 dengan environment dan server MLflow tetap aktif:
```powershell
python -m src.main baseline
python -m src.main tune --trials 24
python -m src.main seeds
python -m src.main ablation
python -m src.main register
python -m src.main predict
```
`predict` memuat model dengan URI `models:/fashion-mnist-cnn-classifier@champion` dan menghitung prediksi pada 16 sampel uji. Proses training cukup berat; gunakan GPU jika tersedia.

## 6. Bukti yang harus disimpan sendiri
Ambil screenshot dari MLflow UI setelah eksperimen benar-benar selesai:
1. Experiment overview dan run baseline.
2. Compare runs yang memperlihatkan 24 nested trial.
3. Ringkasan tiga seed (mean dan std).
4. Hasil ablation.
5. Model Registry yang menunjukkan alias `champion`.
6. Output prediksi 16 sampel dan terminal `dvc status`.

## Catatan kejujuran hasil
Angka metrik tidak ditulis tetap dalam README/laporan sebelum eksperimen dijalankan. Salin nilai yang benar-benar tercatat pada MLflow ke laporan final. Hash dataset yang benar adalah output aktual `data_md5`/file `.dvc`, bukan nilai contoh.
