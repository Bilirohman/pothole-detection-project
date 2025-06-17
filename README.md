# Proyek AI: Deteksi & Klasifikasi Kerusakan Jalan untuk Smart City

Model AI ini mampu mendeteksi kerusakan jalan (jalan berlubang) secara otomatis melalui gambar atau video dan mengklasifikasikan tingkat keparahannya. Dibuat menggunakan **Python** dengan model AI **YOLOv8** dan antarmuka pengguna (UI) yang interaktif menggunakan **Streamlit**.

## 1. Problem yang Diselesaikan

Pelaporan dan perbaikan jalan rusak seringkali lambat karena proses identifikasi yang manual dan subjektif. Model AI ini dirancang untuk mengatasi masalah tersebut dengan kemampuan sebagai berikut:

* **🔎 Identifikasi Otomatis:** Model dapat secara otomatis menemukan lokasi lubang pada gambar atau frame video yang diunggah oleh pengguna atau diambil dari data survei jalan.
* **📊 Klasifikasi Tingkat Kerusakan:** Tidak hanya mendeteksi, model ini juga mampu melabeli setiap kerusakan ke dalam tiga kategori berdasarkan ukurannya secara visual:
    * **`Kerusakan Ringan`**: Lubang kecil yang belum terlalu dalam.
    * **`Kerusakan Sedang`**: Lubang yang lebih lebar dan dalam.
    * **`Kerusakan Berat`**: Kerusakan signifikan yang dapat membahayakan pengguna jalan.
* **⚡ Efisiensi Pelaporan:** Mempercepat proses pemetaan area jalan yang rusak, memungkinkan dinas terkait untuk memprioritaskan perbaikan secara lebih efektif dan berbasis data.
* **🖥️ Visualisasi Interaktif:** Antarmuka Streamlit menampilkan gambar/video asli beserta hasil deteksi (kotak pembatas/bounding box) yang diberi label dan warna sesuai tingkat kerusakan, sehingga mudah dipahami oleh siapa saja.

## 2. Model AI yang Digunakan

* **Model:** **YOLOv8 (You Only Look Once version 8)**.
* **Arsitektur:** YOLO adalah model *object detection* canggih yang terkenal karena kecepatan dan akurasinya yang tinggi. Model ini mampu memproses gambar dalam satu kali proses (hence, *You Only Look Once*) untuk mendeteksi berbagai objek beserta lokasinya.
* **Proses Pelatihan (Training):**
    1.  Model tersebut kami latih menggunakan dataset spesifik dari roboflox berisikan gambar-gambar jalan berlubang.
    2.  Selama pelatihan, model belajar mengenali pola visual dari tiga kelas yang kita definisikan: `ringan`, `sedang`, dan `berat`.
    3.  Hasil akhirnya adalah sebuah file bobot (`.pt`) yang berisi "pengetahuan" model untuk melakukan deteksi spesifik ini.

## 3. Dataset

* **Nama Dataset:** **Pothole Detection Dataset**
* **Sumber & Link:** [Roboflow: Pothole Detection Dataset](https://universe.roboflow.com/jerry-cooper-tlzkx/pothole_detection-hfnqo)
* **Deskripsi:** Dataset ini berisi ribuan gambar jalan dari berbagai kondisi dan sudut pandang, di mana setiap lubang telah diberi anotasi (diberi kotak pembatas).
* **Klasifikasi:**
    Dataset aslinya kemungkinan hanya memiliki satu kelas, yaitu `pothole`. Untuk memenuhi kebutuhan klasifikasi tingkat kerusakan (`ringan`, `sedang`, `berat`)
    1.  **Pra-pemrosesan & Pelabelan Ulang:** Sebelum pelatihan, data anotasi dari dataset ini perlu dimodifikasi. Kita dapat membuat skrip kecil untuk mengklasifikan setiap *bounding box* `pothole` menjadi tiga kelas baru berdasarkan luas relatifnya terhadap ukuran gambar.
    2.  **Contoh Logika Sederhana:**
        * Jika luas *bounding box* < 2% dari luas gambar -> `Kerusakan Ringan`.
        * Jika luas *bounding box* antara 2% - 5% dari luas gambar -> `Kerusakan Sedang`.
        * Jika luas *bounding box* > 5% dari luas gambar -> `Kerusakan Berat`.
        Ambang batas (threshold) ini dapat disesuaikan untuk mendapatkan hasil terbaik.

---
## Langkah-Langkah Implementasi Proyek

#### 1. Persiapan Lingkungan
* Instal semua library yang dibutuhkan: `streamlit`, `ultralytics`, `opencv-python-headless`, `Pillow`, `pandas`.
```python
pip install streamlit ultralytics opencv-python-headless pandas Pillow
```

#### 2. Mempersiapkan Dataset 
* Unduh dan siapkan dataset yang telah dilabeli ulang ke dalam 3 kelas.
```python
python prepare_dataset.py
```

#### 3. Melatih Model YOLOv8
```python
python train.py
```
    
#### 4. Pembuatan Aplikasi Streamlit (`app.py`)
* **Fitur UI:**
    * Judul dan deskripsi aplikasi.
    * Opsi untuk mengunggah file gambar (`jpg`, `png`) atau video (`mp4`).
    * Tombol "Proses" untuk menjalankan deteksi.
    * Area untuk menampilkan output (gambar/video dengan *bounding box*).
```python
streamlit run app.py
```

## 📁 Struktur Folder Proyek (Saran)
- pothole-detection-project/
- │
- ├── 📜 app.py                # File utama aplikasi Streamlit
- ├── 🐍 train.py              # Skrip untuk melatih model YOLOv8
- ├── ⚙️ prepare_dataset.py    # Skrip untuk memproses dataset (opsional tapi penting)
- ├── 📦 model/
- │   └── (kosongkan dulu, isi dengan best.pt setelah training)
- │
- ├── 🖼️ dataset/
- │   └── config.yaml         # File konfigurasi dataset untuk YOLO
- │
- ├── 📄 requirements.txt      # Daftar library Python yang dibutuhkan
- └── 📝 README.md             # File deskripsi proyek (dari jawaban sebelumnya)
