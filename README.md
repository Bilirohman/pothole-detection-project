# Proyek AI: Deteksi & Klasifikasi Kerusakan Jalan untuk Smart City

Proyek ini adalah sebuah purwarupa (prototype) aplikasi berbasis AI untuk mendukung inisiatif **Smart City** di Indonesia. Aplikasi ini mampu mendeteksi kerusakan jalan (jalan berlubang) secara otomatis melalui gambar atau video dan mengklasifikasikan tingkat keparahannya.

Dibuat menggunakan **Python** dengan model AI **YOLOv8** dan antarmuka pengguna (UI) yang interaktif menggunakan **Streamlit**.

## 1. Problem yang Diselesaikan (Kemampuan Model)

Pelaporan dan perbaikan jalan rusak seringkali lambat karena proses identifikasi yang manual dan subjektif. Model AI ini dirancang untuk mengatasi masalah tersebut dengan kemampuan sebagai berikut:

* **🔎 Identifikasi Otomatis:** Model dapat secara otomatis menemukan lokasi lubang pada gambar atau frame video yang diunggah oleh pengguna atau diambil dari data survei jalan.
* **📊 Klasifikasi Tingkat Kerusakan:** Tidak hanya mendeteksi, model ini juga mampu melabeli setiap kerusakan ke dalam tiga kategori berdasarkan ukurannya secara visual:
    * **`Kerusakan Ringan`**: Lubang kecil yang belum terlalu dalam.
    * **`Kerusakan Sedang`**: Lubang yang lebih lebar dan dalam.
    * **`Kerusakan Berat`**: Kerusakan signifikan yang dapat membahayakan pengguna jalan.
* **⚡ Efisiensi Pelaporan:** Mempercepat proses pemetaan area jalan yang rusak, memungkinkan dinas terkait untuk memprioritaskan perbaikan secara lebih efektif dan berbasis data.
* **🖥️ Visualisasi Interaktif:** Antarmuka Streamlit menampilkan gambar/video asli beserta hasil deteksi (kotak pembatas/bounding box) yang diberi label dan warna sesuai tingkat kerusakan, sehingga mudah dipahami oleh siapa saja.

## 2. Model AI yang Digunakan

Untuk tugas ini, kita tidak menggunakan API dari AI yang sudah jadi, melainkan melatih model kita sendiri.

* **Model:** **YOLOv8 (You Only Look Once version 8)**.
* **Arsitektur:** YOLO adalah model *object detection* canggih yang terkenal karena kecepatan dan akurasinya yang tinggi. Model ini mampu memproses gambar dalam satu kali proses (hence, *You Only Look Once*) untuk mendeteksi berbagai objek beserta lokasinya.
* **Proses Pelatihan (Training):**
    1.  Kami menggunakan *transfer learning* dari model YOLOv8 yang sudah dilatih pada dataset besar (seperti COCO).
    2.  Model tersebut kemudian dilatih ulang (*fine-tuning*) menggunakan dataset spesifik berisi gambar-gambar jalan berlubang.
    3.  Selama pelatihan, model belajar mengenali pola visual dari tiga kelas yang kita definisikan: `ringan`, `sedang`, dan `berat`.
    4.  Hasil akhirnya adalah sebuah file bobot (`.pt`) yang berisi "pengetahuan" model untuk melakukan deteksi spesifik ini.

## 3. Dataset

Dataset adalah komponen paling krusial dalam melatih model AI yang akurat. Untuk proyek ini, kita menggunakan dataset publik yang telah dianotasi.

* **Nama Dataset:** **Pothole Detection Dataset**
* **Sumber & Link:** [Kaggle: Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumarojha/pothole-detection-dataset)
* **Deskripsi:** Dataset ini berisi ratusan gambar jalan dari berbagai kondisi dan sudut pandang, di mana setiap lubang telah diberi anotasi (diberi kotak pembatas).
* **⭐ Catatan Penting Mengenai Klasifikasi:**
    Dataset aslinya kemungkinan hanya memiliki satu kelas, yaitu `pothole`. Untuk memenuhi kebutuhan klasifikasi tingkat kerusakan (`ringan`, `sedang`, `berat`), langkah tambahan perlu dilakukan:
    1.  **Pra-pemrosesan & Pelabelan Ulang:** Sebelum pelatihan, data anotasi dari dataset ini perlu dimodifikasi. Kita dapat membuat skrip kecil untuk mengklasifikan setiap *bounding box* `pothole` menjadi tiga kelas baru berdasarkan luas relatifnya terhadap ukuran gambar.
    2.  **Contoh Logika Sederhana:**
        * Jika luas *bounding box* < 2% dari luas gambar -> `Kerusakan Ringan`.
        * Jika luas *bounding box* antara 2% - 5% dari luas gambar -> `Kerusakan Sedang`.
        * Jika luas *bounding box* > 5% dari luas gambar -> `Kerusakan Berat`.
        Ambang batas (threshold) ini dapat disesuaikan untuk mendapatkan hasil terbaik.

---
## Langkah-Langkah Implementasi Proyek

Berikut adalah gambaran umum bagaimana proyek ini dapat dibangun dari awal hingga akhir.

#### 1. Persiapan Lingkungan
* Buat lingkungan virtual (misalnya dengan `venv` atau `conda`).
* Instal semua library yang dibutuhkan: `torch`, `ultralytics`, `streamlit`, `opencv-python`, `pandas`.

#### 2. Pelatihan Model YOLOv8
* Unduh dan siapkan dataset yang telah dilabeli ulang ke dalam 3 kelas.
* Buat file konfigurasi dataset (`data.yaml`) untuk YOLO.
* Jalankan skrip pelatihan Python menggunakan library `ultralytics`.
    ```python
    from ultralytics import YOLO

    # Load pre-trained model
    model = YOLO('yolov8n.pt') 

    # Train the model on custom dataset
    results = model.train(data='config.yaml', epochs=50, imgsz=640)

    # Hasil terbaik akan tersimpan di folder runs/detect/train/weights/best.pt
    ```

#### 3. Pembuatan Aplikasi Streamlit (`app.py`)
* Buat skrip Python untuk UI.
* **Fitur UI:**
    * Judul dan deskripsi aplikasi.
    * Opsi untuk mengunggah file gambar (`jpg`, `png`) atau video (`mp4`).
    * Tombol "Proses" untuk menjalankan deteksi.
    * Area untuk menampilkan output (gambar/video dengan *bounding box*).

#### 4. Integrasi Model dengan Streamlit
* Muat model YOLOv8 yang sudah dilatih (`best.pt`) di dalam aplikasi Streamlit.
* Buat fungsi untuk menerima file gambar/video, menjalankannya melalui model, dan menggambar hasil deteksi (kotak dan label) pada gambar/video tersebut menggunakan OpenCV.

## 📁 Struktur Folder Proyek (Saran)
pothole-detection-project/
│
├── 📜 app.py                # File utama aplikasi Streamlit
├── 🐍 train.py              # Skrip untuk melatih model YOLOv8
├── ⚙️ prepare_dataset.py    # Skrip untuk memproses dataset (opsional tapi penting)
├── 📦 model/
│   └── (kosongkan dulu, isi dengan best.pt setelah training)
│
├── 🖼️ dataset/
│   └── config.yaml         # File konfigurasi dataset untuk YOLO
│
├── 📄 requirements.txt      # Daftar library Python yang dibutuhkan
└── 📝 README.md             # File deskripsi proyek (dari jawaban sebelumnya)
