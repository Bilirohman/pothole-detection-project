# File: app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# --- KONFIGURASI ---
MODEL_PATH = "model/best.pt"  # Path ke model hasil training Anda

# Warna untuk setiap kelas (dalam BGR format untuk OpenCV)
CLASS_COLORS = {
    "Kerusakan Ringan": (0, 255, 255),  # Kuning
    "Kerusakan Sedang": (0, 165, 255),  # Oranye
    "Kerusakan Berat": (0, 0, 255),  # Merah
}
CONFIDENCE_THRESHOLD = 0.3  # Ambang batas kepercayaan

# Fungsi untuk memuat model (dengan cache agar tidak loading berulang kali)
@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def process_frame(frame, model):
    """Memproses satu frame (gambar) dengan model YOLO dan menggambar hasilnya."""
    # Lakukan deteksi pada frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    # Dapatkan anotasi dari hasil deteksi pertama
    annotated_frame = results[0].plot()

    # Hitung jumlah deteksi per kelas
    detections = {"Kerusakan Ringan": 0, "Kerusakan Sedang": 0, "Kerusakan Berat": 0}
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name in detections:
            detections[class_name] += 1

    return annotated_frame, detections


# --- UI STREAMLIT ---
st.set_page_config(page_title="Deteksi Kerusakan Jalan", page_icon="🤖", layout="wide")

st.title("AI Deteksi & Klasifikasi Kerusakan Jalan")
st.markdown(
    "Aplikasi untuk mendeteksi jalan berlubang dari gambar atau video menggunakan model **YOLOv8**."
)
st.markdown("---")

# Muat model
model = load_model(MODEL_PATH)
if model is None:
    st.stop()

# Pilihan input di sidebar
st.sidebar.header("Pilih Input")
input_type = st.sidebar.radio("Pilih jenis input:", ("Gambar", "Video"))

uploaded_file = st.sidebar.file_uploader(
    "Unggah file...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Asli")
        if input_type == "Gambar":
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        else:
            st.video(uploaded_file)

    with col2:
        st.subheader("Hasil Deteksi")
        if st.sidebar.button("Proses Sekarang!"):
            with st.spinner("Sedang memproses..."):
                if input_type == "Gambar":
                    # Proses Gambar
                    image = Image.open(uploaded_file).convert("RGB")
                    frame = np.array(image)
                    processed_frame, detections = process_frame(frame, model)

                    st.image(
                        processed_frame, caption="Hasil Deteksi", use_column_width=True
                    )

                    st.success("Proses selesai!")
                    st.subheader("Ringkasan Deteksi:")
                    for cls, count in detections.items():
                        if count > 0:
                            st.write(f"- **{cls}:** {count} ditemukan")

                else:
                    # Proses Video
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name

                    cap = cv2.VideoCapture(video_path)

                    # Placeholder untuk menampilkan video hasil proses
                    output_video_container = st.empty()

                    total_detections = {
                        "Kerusakan Ringan": 0,
                        "Kerusakan Sedang": 0,
                        "Kerusakan Berat": 0,
                    }

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        processed_frame, frame_detections = process_frame(frame, model)

                        for cls, count in frame_detections.items():
                            total_detections[cls] += count

                        output_video_container.image(
                            processed_frame, channels="BGR", use_column_width=True
                        )

                    cap.release()
                    tfile.close()

                    st.success("Proses video selesai!")
                    st.subheader("Ringkasan Total Deteksi dari Video:")
                    for cls, count in total_detections.items():
                        if count > 0:
                            st.write(f"- **{cls}:** {count} total deteksi")
else:
    st.info("Silakan unggah file gambar atau video melalui sidebar untuk memulai.")

st.sidebar.markdown("---")
