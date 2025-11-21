# 🚗 ANPR – Automatic Number Plate Recognition using YOLOv8 & Streamlit

This project is a **Real-Time Automatic Number Plate Recognition (ANPR)** system built using **YOLOv8 (Ultralytics)** for license plate detection and **Streamlit** for a clean, interactive web interface.  
The application allows users to upload **images or videos**, automatically detects number plates, and displays the processed output instantly.

---

## 🔥 Features

- ✔ Real-time **License Plate Detection**  
- ✔ Supports **Images & Video Processing**  
- ✔ Built using **YOLOv8 (Ultralytics)**  
- ✔ Clean & interactive **Streamlit Web UI**  
- ✔ Automatic bounding boxes & confidence scores  
- ✔ Fully **Deployed on Streamlit Cloud**  
- ✔ Works on CPU (no GPU required)  
- ✔ Simple, fast, and accurate ANPR solution  

---

## 🌐 Live Demo (Deployed App)

👉 **LIVE APP:**  
https://anpr-yolo-app-6776.streamlit.app/

---

## 📂 Project Structure
```
ANPR-YOLO-App/
│
├── best_license_plate_model.pt      # YOLOv8 trained ANPR model
├── yolo_application.py              # Main Streamlit application
├── requirements.txt                 # Dependencies for the project
├── demo.mp4                         # Sample video
├── images.jpeg                      # Sample test image
└── README.md                        # Project documentation
```



---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core Programming |
| **YOLOv8 (Ultralytics)** | Number Plate Detection |
| **OpenCV** | Image & Video Processing |
| **Streamlit** | Web App Framework |
| **PyTorch** | Deep Learning Backend |

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anil8824/ANPR-YOLO-App.git
cd ANPR-YOLO-App
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run App
```bash
streamlit run yolo_application.py
```



## 📦 Model Used
This project uses a custom-trained YOLOv8 ANPR model:
```
best_license_plate_model.pt
```



## 👨‍💻 Developer Info

**Anil Agarwal**  
Python Developer | ML/AI Enthusiast | YOLO Specialist  

🔗 GitHub: https://github.com/Anil8824  
🔗 LinkedIn: https://www.linkedin.com/in/anil-agarwal-a5a1a2217/



⭐ Support This Project
If you found this project helpful, please consider giving it a ⭐ star on GitHub.
Your support motivates more AI/ML projects like this!

