# File: prepare_dataset.py (VERSI PERBAIKAN)

import os
from glob import glob
from tqdm import tqdm

# --- KONFIGURASI ---
LABEL_DIRS = [
    "dataset/train/labels/",
    "dataset/valid/labels/",
    "dataset/test/labels/", 
]

# Definisikan ambang batas (threshold) untuk klasifikasi berdasarkan luas area
THRESHOLD_RINGAN_KE_SEDANG = 0.02  # 2% dari total area gambar
THRESHOLD_SEDANG_KE_BERAT = 0.05  # 5% dari total area gambar


def relabel_dataset():
    """
    Membaca semua file anotasi .txt dari YOLO, menghitung luas bounding box,
    dan mengganti class_id (awalnya 0) menjadi 0, 1, atau 2 berdasarkan ukurannya.
    """
    total_files_processed = 0
    stats = {"ringan": 0, "sedang": 0, "berat": 0}

    for label_dir in LABEL_DIRS:
        if not os.path.exists(label_dir):
            print(f"Peringatan: Direktori '{label_dir}' tidak ditemukan. Melewati...")
            continue

        # Dapatkan semua file .txt di direktori
        label_files = glob(os.path.join(label_dir, "*.txt"))

        if not label_files:
            print(
                f"Peringatan: Tidak ada file label .txt yang ditemukan di '{label_dir}'"
            )
            continue

        print(f"\nMemproses direktori: {label_dir}")

        for file_path in tqdm(
            label_files,
            desc=f"Processing {os.path.basename(os.path.dirname(label_dir))}",
        ):
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # class_id, x_center, y_center, width, height
                _, x_c, y_c, w, h = map(float, parts)

                area = w * h

                new_class_id = 0
                if area < THRESHOLD_RINGAN_KE_SEDANG:
                    new_class_id = 0  # Kerusakan Ringan
                    stats["ringan"] += 1
                elif THRESHOLD_RINGAN_KE_SEDANG <= area < THRESHOLD_SEDANG_KE_BERAT:
                    new_class_id = 1  # Kerusakan Sedang
                    stats["sedang"] += 1
                else:
                    new_class_id = 2  # Kerusakan Berat
                    stats["berat"] += 1

                new_lines.append(f"{new_class_id} {x_c} {y_c} {w} {h}\n")

            with open(file_path, "w") as f:
                f.writelines(new_lines)

            total_files_processed += 1

    print("\n=========================================")
    print("Proses pelabelan ulang selesai.")
    print(f"Total file yang diproses: {total_files_processed}")
    print("Statistik Kelas Baru:")
    print(f"  - Kerusakan Ringan: {stats['ringan']}")
    print(f"  - Kerusakan Sedang: {stats['sedang']}")
    print(f"  - Kerusakan Berat: {stats['berat']}")
    print("=========================================")


if __name__ == "__main__":
    relabel_dataset()
