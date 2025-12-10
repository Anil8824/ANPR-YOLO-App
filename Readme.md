# 🚗 ANPR – Automatic Number Plate Recognition using YOLOv8, EasyOCR & Streamlit

**This project is a complete ANPR (Automatic Number Plate Recognition) system that detects number plates from images and videos using YOLOv8 and extracts the text using EasyOCR.**

**The system runs fully on CPU and is deployed on Streamlit Cloud.**

## ✨ Features

**🔍 Number Plate Detection using YOLOv8**

**🧾 Text Extraction using EasyOCR**

**🖼 Supports Images & Videos**

**🟩 Bounding boxes + extracted text overlay**

**🌐 Streamlit-based clean web interface**

**⚡ CPU-only — no GPU required**

**📥 Downloadable processed video**

**☁ Works smoothly on Streamlit Cloud**


## 🌍 Live Demo

### 👉 Try the deployed app:

**https://anpr-yolo-app-6776.streamlit.app/**



## 📁 Project Structure

ANPR-YOLO-App/
│
├── best_license_plate_model.pt      # YOLOv8 trained ANPR model
├── yolo_application.py              # Main Streamlit application
├── requirements.txt                 # Project dependencies
├── demo.mp4                         # Sample video
├── images.jpeg                      # Sample test image
└── README.md                        # Documentation




## ⚙ How It Works

**User uploads an image or video.**

**YOLOv8 detects number plate regions.**

**Detected plate is cropped and passed to EasyOCR.**

**OCR text is drawn on top of the frame.**

**For videos, each frame is processed and exported as a downloadable MP4.**

**Final image or video is returned to the user.**



## 🛠 Tech Stack

| Technology               | Purpose                |
| ------------------------ | ---------------------- |
| **YOLOv8 (Ultralytics)** | Number Plate Detection |
| **EasyOCR**              | Text Extraction        |
| **OpenCV**               | Frame/Image Processing |
| **Streamlit**            | Web Interface          |
| **Python**               | Core Logic             |
| **PyTorch**              | Backend for YOLO       |



## 🚀 Run Locally

### 1️⃣ Clone the repository

**git clone https://github.com/Anil8824/ANPR-YOLO-App.git**

**cd ANPR-YOLO-App**


### 2️⃣ Create virtual environment

**python -m venv venv**
**venv\Scripts\activate    # Windows**


### 3️⃣ Install dependencies

**pip install -r requirements.txt**


### 4️⃣ Run the Streamlit app

**streamlit run yolo_application.py**


## 🧠 Model Used

**This project uses a custom-trained YOLOv8 ANPR model:**

**best_license_plate_model.pt**


## ☁ Notes for Streamlit Cloud
**Uses opencv-python-headless → avoids libGL errors**

**EasyOCR works fine without GPU**

**Streamlit Cloud cannot preview videos, so videos are provided as downloadable file**


## 👨‍💻 Developer

**Anil Agarwal**

**Python Developer | ML/AI Enthusiast | Computer Vision Learner**

**🔗 GitHub: https://github.com/Anil8824**

**🔗 LinkedIn: https://www.linkedin.com/in/anil-agarwal-a5a1a2217/**


## ⭐ Support This Project
**If this helped you, please ⭐ star the repo on GitHub.**

**Your support motivates more AI/ML projects!**
