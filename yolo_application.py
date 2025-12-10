import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import pytesseract
import os

# Create temp folder
os.makedirs("temp", exist_ok=True)

# App title
st.title("YOLOv5 ANPR - Number Plate Detection & Recognition (Tesseract OCR)")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# ---------------- LOAD YOLOv5 MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="best_license_plate_model.pt",
            force_reload=True
        )
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# ---------------- IMAGE PROCESSING ----------------
def predict_and_save_image(input_path, output_path):
    try:
        image = cv2.imread(input_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model(rgb)

        detected_text = None

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = rgb[y1:y2, x1:x2]

            if crop.size != 0:
                text = pytesseract.image_to_string(crop, config='--psm 7')
                text = "".join(filter(str.isalnum, text))

                if text:
                    detected_text = text
                    cv2.putText(
                        image, detected_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2
                    )

        cv2.imwrite(output_path, image)

        if detected_text:
            st.success(f"Detected Number Plate: **{detected_text}**")

        return output_path

    except Exception as e:
        st.error(f"Image Error: {e}")
        return None


# ---------------- VIDEO PROCESSING ----------------
def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video!")
            return None

        w = int(cap.get(3))
        h = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        progress = st.progress(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb)

            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = map(int, det[:6])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                crop = rgb[y1:y2, x1:x2]
                if crop.size != 0:
                    text = pytesseract.image_to_string(crop, config='--psm 7')
                    text = "".join(filter(str.isalnum, text))

                    if text:
                        cv2.putText(
                            frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2
                        )

            out.write(frame)
            count += 1
            progress.progress(count / total)

        cap.release()
        out.release()
        return output_path

    except Exception as e:
        st.error(f"Video Error: {e}")
        return None


# ---------------- ROUTER ----------------
def process_media(input_path, output_path):
    ext = input_path.split(".")[-1].lower()
    if ext in ["mp4", "mov", "avi", "mkv"]:
        return predict_and_plot_video(input_path, output_path)
    else:
        return predict_and_save_image(input_path, output_path)


# ---------------- MAIN ----------------
if uploaded_file:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", "output_" + uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing…")
    result = process_media(input_path, output_path)

    if result:
        if result.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            st.image(result)
        else:
            with open(result, "rb") as v:
                st.download_button(
                    "⬇ Download Processed Video",
                    v.read(),
                    file_name="processed_output.mp4",
                    mime="video/mp4"
                )
            st.info("Streamlit Cloud cannot preview video — download to view.")
