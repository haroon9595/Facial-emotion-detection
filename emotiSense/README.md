# 🧠 EmotiSense AI — Facial Emotion Detection System

A professional AI-powered facial emotion detection app built with **Streamlit**, **TensorFlow/Keras**, and **OpenCV**.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🖼️ Image Detection | Upload any photo, detect all faces + emotions |
| 🎥 Live Detection | Real-time webcam emotion recognition |
| 📊 Analytics Dashboard | Charts, history log, statistics |
| 🌙 Dark / Light Mode | Toggle theme anytime |
| 📸 Screenshot Download | Save annotated images |
| ⬇️ CSV Export | Export history and results |
| 👥 Multi-face Detection | Handles multiple faces simultaneously |
| 📈 Confidence Bars | Visual probability display per emotion |

---

## 🎭 Detectable Emotions

😡 Angry | 🤢 Disgust | 😨 Fear | 😄 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise

---

## 🛠️ Installation

### 1. Clone / extract the project
```bash
cd face/
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## 📁 Project Structure

```
face/
├── app.py                  ← Main upgraded Streamlit app
├── main.py                 ← Original app (backup)
├── emotion_model.keras     ← Trained CNN model
├── requirements.txt        ← Dependencies
└── README.md               ← This file
```

---

## 🧠 Model Info

- Architecture: CNN (Convolutional Neural Network)
- Input: 48×48 grayscale face image
- Output: 7-class emotion probabilities
- Face Detection: OpenCV Haar Cascade

---

## 📊 Navigation

| Page | Purpose |
|---|---|
| 🏠 Home | Overview and tech info |
| 🖼️ Image Detection | Upload & detect emotions |
| 🎥 Live Detection | Webcam real-time mode |
| 📊 Analytics | History, charts, export |

---

## ⚙️ Requirements

- Python 3.8+
- Webcam (for live mode)
- ~500MB RAM for model loading

---

## 🔧 Troubleshooting

**Live mode not working?**
```bash
pip install streamlit-webrtc av
```

**Model not found?**
Make sure `emotion_model.keras` is in the same folder as `app.py`.

**OpenCV error?**
```bash
pip install opencv-python-headless
```
