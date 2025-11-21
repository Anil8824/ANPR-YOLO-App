import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# Ensure temp folder exists
os.makedirs("temp", exist_ok=True)

# Set the title of the Streamlit app
st.title("YOLO Image and Video Processing")

# Allow users to upload images or videos
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# Load YOLO model
try:
    model = YOLO("best_license_plate_model.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")


# ---------------------- IMAGE PROCESSING ----------------------
def predict_and_save_image(path_test_car, output_image_path):
    try:
        results = model.predict(path_test_car, device='cpu', verbose=False)
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{conf*100:.2f}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ---------------------- VIDEO PROCESSING ----------------------
def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0 or fps is None:
            fps = 24  # fallback

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FIXED: correct mp4 writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        progress = st.progress(0)
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, device='cpu', verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf*100:.1f}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            out.write(frame)

            current_frame += 1
            progress.progress(min(current_frame / total_frames, 1.0))

        cap.release()
        out.release()
        progress.progress(1.0)

        return output_path

    except Exception as e:
        st.error(f"Video processing error: {e}")
        return None


# ---------------------- MEDIA HANDLER ----------------------
def process_media(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)

    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)

    st.error("Unsupported file format.")
    return None


# ---------------------- MAIN EXECUTION ----------------------
if uploaded_file:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", "output_" + uploaded_file.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Processing...")
    result = process_media(input_path, output_path)

    if result:
        ext = result.lower()

        # ---------------------- FIXED VIDEO DISPLAY ----------------------
        if ext.endswith((".mp4", ".avi", ".mov", ".mkv")):
            st.success("Video processed successfully!")

            with open(result, "rb") as v:
                st.video(v.read())

        else:
            st.success("Image processed successfully!")
            st.image(result)
