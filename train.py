# File: train.py (Fungsi main() yang dioptimalkan untuk RTX 4060)

from ultralytics import YOLO
import torch


def main():
    """
    Fungsi utama untuk melatih model YOLOv8 pada dataset kerusakan jalan.
    """
    # Verifikasi cepat ketersediaan GPU saat skrip dijalankan
    if not torch.cuda.is_available():
        print(
            "Peringatan: CUDA tidak tersedia. Training akan berjalan di CPU dan akan sangat lambat."
        )
        device = "cpu"
    else:
        device = "cuda"
    print(f"Training akan dilakukan menggunakan: {device.upper()}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    print("\nMemulai proses training model dengan parameter teroptimasi untuk GPU...")
    results = model.train(
        data="dataset/config.yaml",
        epochs=50,
        imgsz=640,  # Resolusi gambar
        batch=6,  # VRAM
        name="yolov8n_pothole_RTX4060",
        cache=True,  # memuat data ke RAM
        workers=4,  # core CPU untuk data loading
    )

    print("\n=========================================")
    print("Training Selesai!")
    print(f"Hasil training disimpan di folder 'runs/detect/{results.save_dir.name}'")
    print("Salin file 'best.pt' dari folder tersebut ke dalam folder 'model/'")
    print("=========================================")


if __name__ == "__main__":
    main()
