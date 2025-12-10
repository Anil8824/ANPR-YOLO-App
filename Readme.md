🚗 ANPR – Automatic Number Plate Recognition using YOLOv8, EasyOCR & Streamlit

This project is a complete ANPR (Automatic Number Plate Recognition) system that detects number plates from images and videos using YOLOv8 and extracts the text using EasyOCR. The system runs end-to-end on Streamlit Cloud using CPU only and supports both image and video uploads.

Features

Number Plate Detection using YOLOv8

Text Extraction using EasyOCR

Supports Images and Videos

Bounding box + extracted text drawn on output

Streamlit-based clean web interface

Works fully on CPU

Downloadable processed video output

Streamlit Cloud compatible

Live Demo

Streamlit App:
https://anpr-yolo-app-6776.streamlit.app/

Project Structure

ANPR-YOLO-App/
│
├── best_license_plate_model.pt (Trained YOLOv8 model)
├── yolo_application.py (Main application file)
├── requirements.txt (Dependencies)
├── demo.mp4 (Sample video)
├── images.jpeg (Sample image)
└── README.md (Documentation)

How It Works

User uploads an image or video.

YOLOv8 detects the number plate and returns bounding boxes.

The detected region is cropped and passed to EasyOCR for text extraction.

Extracted text is displayed on the image or embedded on each video frame.

Final output image is displayed; processed videos are provided as a downloadable file.

Tech Stack

Python
YOLOv8 (Ultralytics)
EasyOCR
OpenCV
Streamlit
PyTorch

Run Locally

Clone the repository:
git clone https://github.com/Anil8824/ANPR-YOLO-App.git

cd ANPR-YOLO-App

Create virtual environment:
python -m venv venv
venv\Scripts\activate (Windows)

Install dependencies:
pip install -r requirements.txt

Run Streamlit app:
streamlit run yolo_application.py

Model Used

A custom-trained YOLOv8 model for ANPR detection:
best_license_plate_model.pt

Notes for Streamlit Cloud

opencv-python-headless is used to avoid libGL errors

EasyOCR works fine without GPU

Streamlit Cloud cannot preview video; therefore download button is provided

Developer

Anil Agarwal
Python Developer | ML/AI Enthusiast | Computer Vision Learner

GitHub: https://github.com/Anil8824

LinkedIn: https://www.linkedin.com/in/anil-agarwal-a5a1a2217/

Support This Project

If you like this project, please star the repository on GitHub.
Your support motivates more AI/ML projects! ⭐
