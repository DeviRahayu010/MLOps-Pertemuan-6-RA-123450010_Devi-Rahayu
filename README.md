# Tugas Pertemuan 6 MLOps — Model Experimentation & Tracking

**Nama:** Devi Rahayu  
**NIM:** 123450010  
**Kelas:** RA

## Deskripsi

Proyek ini merupakan implementasi tugas MLOps Pertemuan 6 mengenai *Model Experimentation & Tracking* menggunakan dataset Fashion-MNIST. Eksperimen mencakup pencatatan eksperimen menggunakan MLflow, pelatihan model CNN, pencarian hyperparameter menggunakan Optuna, evaluasi model dengan beberapa random seed, ablation study, Model Registry, serta versioning dataset menggunakan DVC.

Komponen yang diterapkan meliputi:

- MLflow untuk pencatatan parameter, metrik, dan hasil eksperimen.
- CNN sebagai model baseline untuk klasifikasi Fashion-MNIST.
- Optuna untuk hyperparameter tuning dengan 24 trial menggunakan nested runs.
- Evaluasi konfigurasi terbaik menggunakan tiga random seed.
- Ablation study untuk membandingkan beberapa konfigurasi model.
- MLflow Model Registry dengan alias `champion`.
- DVC untuk versioning dataset.

## 1. Persiapan Lingkungan

Jalankan perintah berikut melalui Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install dvc
```

Pastikan seluruh dependensi berhasil terpasang sebelum menjalankan eksperimen.

## 2. Menjalankan MLflow Tracking Server

Jalankan perintah berikut pada Terminal 1:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Biarkan terminal tetap berjalan selama eksperimen berlangsung.

MLflow UI dapat diakses melalui:

`http://127.0.0.1:5000`

MLflow digunakan untuk menyimpan informasi eksperimen, termasuk parameter, metrik evaluasi, status run, dan artefak model.

## 3. Persiapan Dataset

Buka Terminal 2, aktifkan environment yang telah dibuat, lalu atur tracking URI:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

Jika proyek menyediakan perintah persiapan dataset melalui `src.main`, jalankan:

```powershell
python -m src.main prepare
```

Dataset yang digunakan adalah subset Fashion-MNIST untuk proses pelatihan dan evaluasi model. Proses pengunduhan pertama kali memerlukan koneksi internet.

Pastikan lokasi dan nama file dataset sesuai dengan implementasi pada kode proyek.

## 4. Versioning Dataset Menggunakan DVC

Inisialisasi DVC apabila belum dilakukan:

```powershell
dvc init
```

Tambahkan dataset ke dalam pelacakan DVC sesuai lokasi file yang digunakan oleh proyek. Contoh berikut berlaku apabila dataset disimpan sebagai `data/fashion_mnist_subset.npz`:

```powershell
dvc add data/fashion_mnist_subset.npz
```

Untuk menggunakan penyimpanan lokal sebagai remote DVC:

```powershell
dvc remote add -d localstore ../dvc-store
dvc push
```

Remote lokal digunakan sebagai contoh penyimpanan artefak DVC dan bukan sebagai penyimpanan cloud.

Periksa status versioning data dengan:

```powershell
dvc status
```

## 5. Pelaksanaan Eksperimen

Apabila kode proyek menyediakan antarmuka perintah `src.main`, eksperimen dapat dijalankan secara berurutan melalui Terminal 2:

### 5.1 Baseline CNN

```powershell
python -m src.main baseline
```

Tahap ini digunakan untuk melatih dan mengevaluasi model CNN awal sebagai pembanding bagi eksperimen selanjutnya.

### 5.2 Hyperparameter Tuning dengan Optuna

```powershell
python -m src.main tune --trials 24
```

Optuna digunakan untuk mencari kombinasi hyperparameter yang memberikan performa validasi terbaik. Setiap trial dicatat menggunakan MLflow sesuai implementasi nested runs pada proyek.

### 5.3 Evaluasi dengan Tiga Random Seed

```powershell
python -m src.main seeds
```

Evaluasi ini digunakan untuk mengamati kestabilan performa model ketika random seed diubah. Hasil evaluasi dirangkum menggunakan rata-rata (*mean*) dan standar deviasi (*standard deviation*).

### 5.4 Ablation Study

```powershell
python -m src.main ablation
```

Ablation study dilakukan dengan membandingkan model lengkap dan konfigurasi yang menghilangkan komponen tertentu untuk mengamati pengaruhnya terhadap performa model.

### 5.5 Model Registry

```powershell
python -m src.main register
```

Tahap ini digunakan untuk mendaftarkan model ke MLflow Model Registry dan menetapkan alias `champion` sesuai implementasi proyek.

Nama model yang digunakan pada URI prediksi harus sama dengan nama model yang terdaftar. Pada eksperimen ini, nama model yang tercatat adalah `Devi_FashionMNIST_CNN`.

### 5.6 Prediksi

```powershell
python -m src.main predict
```

Tahap prediksi memuat model terdaftar menggunakan URI:

```text
models:/Devi_FashionMNIST_CNN@champion
```

Prediksi dijalankan pada sampel data uji sesuai implementasi kode proyek.

**Catatan:** perintah `src.main` di atas hanya dapat digunakan apabila subperintah tersebut memang tersedia pada kode proyek. Jika eksperimen dijalankan melalui notebook Google Colab, gunakan notebook sebagai prosedur utama dan sesuaikan petunjuk ini dengan implementasi yang benar-benar tersedia.

## 6. Bukti Eksperimen

Bukti eksperimen disimpan dalam repositori sebagai dokumentasi pelaksanaan tugas. Bukti yang dikumpulkan meliputi:

1. Screenshot MLflow UI yang menampilkan eksperimen dan run baseline.
2. Screenshot MLflow Compare View untuk membandingkan run eksperimen.
3. Screenshot hasil hyperparameter tuning menggunakan Optuna.
4. Ringkasan evaluasi tiga random seed yang menampilkan mean dan standar deviasi.
5. Hasil ablation study.
6. Screenshot MLflow Model Registry yang menunjukkan alias `champion`.
7. Output prediksi model.
8. Output pemeriksaan status DVC.

Hanya bukti yang benar-benar telah dihasilkan dan disimpan yang dicantumkan sebagai hasil pelaksanaan tugas.

## 7. Hasil dan Evaluasi

Hasil eksperimen dianalisis berdasarkan metrik evaluasi yang dicatat pada setiap run MLflow. Perbandingan dilakukan untuk mengamati performa baseline CNN, hasil tuning hyperparameter, kestabilan model pada tiga random seed, serta pengaruh komponen model melalui ablation study.

Hasil aktual, nilai metrik, dan kesimpulan eksperimen didasarkan pada output notebook dan pencatatan MLflow.

## 8. Kesimpulan

Implementasi tugas ini menggabungkan pelatihan model, eksperimen hyperparameter, evaluasi konfigurasi, pencatatan eksperimen, pengelolaan model, dan versioning dataset dalam satu alur kerja MLOps. MLflow digunakan untuk melacak hasil eksperimen dan mengelola model, sedangkan DVC digunakan untuk mendukung pengelolaan versi dataset.

Evaluasi akhir dan pemilihan model terbaik dilakukan berdasarkan hasil eksperimen yang benar-benar tercatat, bukan berdasarkan asumsi atau nilai yang belum diuji.
