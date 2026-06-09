import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import altair as alt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 🔹 Page Config
st.set_page_config(page_title="Emotion Detection", layout="wide")

# 🔹 Sidebar
st.sidebar.title("⚙️ Options")
option = st.sidebar.selectbox("🎯 Choose Mode", ["Image Detection", "Live Detection"])

# 🔹 Load Model
@st.cache_resource
def load_my_model():
    return load_model("emotion_model.keras")

model = load_my_model()

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emoji_map = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# 🔹 Prediction Function
def predict_emotion(img):
    img_gray = img.convert('L')
    img_gray = img_gray.resize((48,48))
    img_array = image.img_to_array(img_gray)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return pred

# =========================================================
# IMAGE MODE
# =========================================================
if option == "Image Detection":

    st.title("😊 Emotion Detection App")
    st.write("Upload an image to detect human emotions")

    col1, col2, col3 = st.columns([1.2, 1.5, 1.2])

    # 🔹 LEFT PANEL
    with col1:
        st.markdown("### 📂 Detection Source")
        uploaded_file = st.file_uploader("Drop or Upload Image", type=["jpg","png","jpeg"])

        st.markdown("#### ⚙️ Actions")
        if st.button("🔄 Reset"):
            st.rerun()

    # 🔹 CENTER PANEL
    with col2:
        st.markdown("### 🧠 Detection Results")

        if uploaded_file:
            img = Image.open(uploaded_file)
            img_cv = np.array(img)

            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                st.warning("⚠️ Face not detected!")
            else:
                for (x,y,w,h) in faces:
                    cv2.rectangle(img_cv,(x,y),(x+w,y+h),(255,0,0),2)

                    # 🔥 Crop face for prediction
                    face = gray[y:y+h, x:x+w]
                    face_img = Image.fromarray(face)

                    pred = predict_emotion(face_img)
                    pred_index = np.argmax(pred)
                    confidence = float(pred[0][pred_index] * 100)

                    emotion = class_names[pred_index]
                    emoji = emoji_map[emotion]

                st.image(img_cv, width=300, caption="Detected Face")

                # 🔹 Confidence Threshold
                threshold = st.slider("Confidence Threshold", 0, 100, 50)

                if confidence >= threshold:
                    st.success(f"{emoji} {emotion} ({confidence:.1f}%)")
                else:
                    st.warning("Confidence too low ❗")

    # 🔹 RIGHT PANEL
    with col3:
        st.markdown("### 📊 Emotion Analytics")

        if uploaded_file and 'pred' in locals():
            data = pd.DataFrame({
                "Emotion": class_names,
                "Probability": pred[0]
            })

            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X("Emotion", sort=None),
                y="Probability",
                color="Emotion"
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

            # 🔹 Download CSV
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv, "results.csv")


# =========================================================
# LIVE MODE (STABLE + EMOJI 🔥)
# =========================================================
elif option == "Live Detection":

    st.title("🎥 Live Emotion Detection")

    class EmotionDetector(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_emotion = ""
            self.last_confidence = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            self.frame_count += 1
            

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_img = Image.fromarray(face)

                # 🔥 Predict only every 5 frames
                if self.frame_count % 5 == 0:
                    pred = predict_emotion(face_img)
                    pred_index = np.argmax(pred)

                    self.last_emotion = class_names[pred_index]
                    self.last_confidence = float(pred[0][pred_index] * 100)

                emoji = emoji_map.get(self.last_emotion, "")

                # 🔥 Always draw rectangle (no flicker)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                # 🔥 Label + emoji + confidence
                label = f"{emoji} {self.last_emotion} ({self.last_confidence:.1f}%)"

                cv2.putText(img, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            return img

    webrtc_streamer(
        key="emotion",
        video_transformer_factory=EmotionDetector
    )