⚖️ IPC Section Suggestor – Crime Description to IPC Section Classification (NLP Model)

This project is an NLP-based system that predicts the most relevant IPC (Indian Penal Code) sections based on a given crime description.
It uses a trained text-classification model and a Streamlit web app to provide fast & accurate IPC suggestions.

✨ Features

🧠 Crime-to-IPC Section Classification (AI Model)

⌨️ Simple text input → Instant IPC result

⚡ Works 100% on CPU

🌐 Streamlit Web Interface

📚 Supports multiple IPC sections

🚨 Helps in early crime categorization for law-related applications

🌍 Live Demo (If Deployed)

👉 (Add your Streamlit link here later)
https://your-ipc-app.streamlit.app/

📁 Project Structure
IPC-Section-Suggestor/
│
├── ipc_model.pkl                   # Trained NLP model (vectorizer + classifier)
├── ipc_sections.json               # IPC label mapping file
├── ipc_app.py                      # Main Streamlit application
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation

⚙️ How It Works

User enters any crime description text
Example: "A man forcibly entered a house and stole jewellery."

NLP pipeline processes the text

Cleaning

Tokenization

TF-IDF vectorization

Classification model prediction

The app returns the most relevant IPC section, e.g.

IPC 457 – Lurking house-trespass or house-breaking at night

IPC 380 – Theft in dwelling house

Streamlit UI displays:
✔ Predicted section
✔ Full section meaning
✔ Confidence score (optional)

🛠 Tech Stack
Technology	Purpose
Python	Programming
scikit-learn	NLP Model Training
TF-IDF Vectorizer	Text Feature Extraction
Streamlit	Web UI
Pickle / JSON	Model & Mapping Storage
🚀 Run Locally
1️⃣ Clone the repository
git clone https://github.com/your-username/IPC-Section-Suggestor.git
cd IPC-Section-Suggestor

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run ipc_app.py

📦 Model Used

This project uses an NLP text classification model trained on crime descriptions mapped to IPC sections.

Stored in:

ipc_model.pkl
ipc_sections.json

📘 Example Input & Output
Input

"A person intentionally injured another person with a knife."

Output

Predicted IPC: IPC 324

Description: Voluntarily causing hurt by dangerous weapons.

Category: Offence against the human body

☁ Notes for Streamlit Cloud

No GPU needed

Model loads instantly (pickle file)

Lightweight dependencies → fast deployment

Perfect for law, police automation, documentation classification apps

👨‍💻 Developer

Anil Agarwal
Python Developer | ML/AI Enthusiast | NLP & Computer Vision

🔗 GitHub: https://github.com/Anil8824

🔗 LinkedIn: https://www.linkedin.com/in/anil-agarwal-a5a1a2217/

⭐ Support This Project

If this project helped you, please ⭐ star the repo.
Your support motivates more AI/NLP projects!
