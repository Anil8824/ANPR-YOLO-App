import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import easyocr

# Create temp folder
os.makedirs("temp", exist_ok=True)

# App title
st.title("YOLO ANPR - Number Plate Detection & Recognition")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Load YOLO Model
try:
    model = YOLO("best_license_plate_model.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# Load EasyOCR
reader = easyocr.Reader(['en'])


# ---------------- IMAGE PROCESSING ----------------
def predict_and_save_image(input_path, output_path):
    try:
        image = cv2.imread(input_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.predict(rgb, device="cpu", verbose=False)

        detected_text = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop number plate
                crop = rgb[y1:y2, x1:x2]

                if crop.size != 0:
                    text_list = reader.readtext(crop, detail=0)
                    if len(text_list) > 0:
                        detected_text = text_list[0]

                        # Put extracted text above the box
                        cv2.putText(image, detected_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 0), 2)

        cv2.imwrite(output_path, image)

        if detected_text:
            st.success(f"Detected Number Plate: **{detected_text}**")

        return output_path

    except Exception as e:
        st.error(f"Image processing error: {e}")
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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        progress = st.progress(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, device="cpu", verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Crop & OCR
                    crop = rgb[y1:y2, x1:x2]
                    if crop.size != 0:
                        text_list = reader.readtext(crop, detail=0)
                        if len(text_list) > 0:
                            detected_text = text_list[0]

                            # Write OCR text
                            cv2.putText(frame, detected_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 0), 2)

            out.write(frame)
            count += 1
            progress.progress(count / total)

        cap.release()
        out.release()
        return output_path

    except Exception as e:
        st.error(f"Video processing error: {e}")
        return None


# ---------------- MEDIA ROUTER ----------------
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

            st.info("Streamlit Cloud does not support video preview—please download to view.")
