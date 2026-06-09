import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import altair as alt
from datetime import datetime
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EmotiSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME & CUSTOM CSS
# ============================================================
def apply_theme(dark_mode):
    if dark_mode:
        bg = "#0f1117"
        card_bg = "#1a1d27"
        sidebar_bg = "#13151f"
        text = "#e8eaf6"
        subtext = "#9fa8da"
        border = "#2d3561"
        accent = "#7c6af7"
        input_bg = "#1e2130"
    else:
        bg = "#f0f2f8"
        card_bg = "#ffffff"
        sidebar_bg = "#e8eaf6"
        text = "#1a1a2e"
        subtext = "#5c6bc0"
        border = "#c5cae9"
        accent = "#5c6bc0"
        input_bg = "#f5f5ff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {bg};
        color: {text};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border};
    }}

    /* Cards */
    .emotion-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }}

    .metric-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }}

    .metric-card h2 {{
        font-size: 2rem;
        font-weight: 700;
        color: {accent};
        margin: 0;
    }}

    .metric-card p {{
        font-size: 0.85rem;
        color: {subtext};
        margin: 4px 0 0 0;
    }}

    /* Navbar Tabs */
    .nav-tabs {{
        display: flex;
        gap: 8px;
        margin-bottom: 24px;
        border-bottom: 2px solid {border};
        padding-bottom: 8px;
    }}

    /* Emotion Badges */
    .badge-happy   {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }}
    .badge-sad     {{ background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }}
    .badge-angry   {{ background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }}
    .badge-fear    {{ background: #f3e5f5; color: #6a1b9a; border: 1px solid #ce93d8; }}
    .badge-disgust {{ background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }}
    .badge-surprise{{ background: #fffde7; color: #f57f17; border: 1px solid #fff176; }}
    .badge-neutral {{ background: #eceff1; color: #37474f; border: 1px solid #b0bec5; }}

    .emotion-badge {{
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 8px 0;
    }}

    /* Progress bars */
    .conf-bar-wrap {{
        background: {border};
        border-radius: 8px;
        height: 10px;
        margin: 4px 0 12px 0;
        overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 10px;
        border-radius: 8px;
        background: linear-gradient(90deg, {accent}, #a78bfa);
        transition: width 0.4s ease;
    }}

    /* Titles */
    .page-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 4px;
    }}
    .page-subtitle {{
        font-size: 1rem;
        color: {subtext};
        margin-bottom: 24px;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {subtext};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid {border};
    }}

    /* Toast */
    .toast {{
        background: {card_bg};
        border-left: 4px solid {accent};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: {text};
        font-size: 0.9rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}

    /* History table */
    .history-row {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
        color: {text};
    }}

    /* Sidebar label */
    .sidebar-label {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {subtext};
        margin-bottom: 6px;
    }}

    /* FIXED: Removed header from here so the sidebar toggle button shows on mobile/small screens */
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid {border} !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
    }}

    /* File uploader */
    .stFileUploader {{
        background: {input_bg};
        border-radius: 12px;
        padding: 4px;
    }}

    /* Selectbox, Slider */
    .stSelectbox > div, .stSlider > div {{
        background: {input_bg} !important;
        border-radius: 8px;
    }}

    div[data-testid="stImage"] img {{
        border-radius: 12px;
        border: 2px solid {border};
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# ============================================================
# THEME APPLY
# ============================================================
apply_theme(st.session_state.dark_mode)

# ============================================================
# CONSTANTS
# ============================================================
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOJI_MAP = {
    "Angry": "😡", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😄", "Neutral": "😐", "Sad": "😢", "Surprise": "😲"
}

COLOR_MAP = {
    "Happy":    "#2e7d32", "Sad":     "#1565c0", "Angry":    "#c62828",
    "Fear":     "#6a1b9a", "Disgust": "#e65100", "Surprise": "#f57f17",
    "Neutral":  "#37474f"
}

BADGE_CLASS = {
    "Happy": "badge-happy", "Sad": "badge-sad", "Angry": "badge-angry",
    "Fear": "badge-fear", "Disgust": "badge-disgust",
    "Surprise": "badge-surprise", "Neutral": "badge-neutral"
}

CHART_COLORS = ["#ef5350","#ab47bc","#7e57c2","#26a69a","#78909c","#42a5f5","#ffca28"]

# ============================================================
# MODEL — Load once, cache forever
# ============================================================
@st.cache_resource
def load_my_model():
    try:
        return load_model("emotion_model.keras")
    except Exception as e:
        return None # Added fallback if model is not present

model = load_my_model()

if model is None:
    st.error("⚠️ Warning: 'emotion_model.keras' not found. Please ensure the model file is in the exact same folder.")

# ============================================================
# IMPROVED PREPROCESSING PIPELINE
# ============================================================
def preprocess_face(gray_face_np: np.ndarray) -> np.ndarray:
    # Resize
    face = cv2.resize(gray_face_np, (48, 48))
    # CLAHE — adaptive contrast enhancement (big accuracy boost)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face = clahe.apply(face)
    # Slight gaussian blur to reduce sensor noise
    face = cv2.GaussianBlur(face, (3, 3), 0)
    # Normalize to [0, 1]
    face = face.astype("float32") / 255.0
    # Reshape → (1, 48, 48, 1)
    face = np.expand_dims(face, axis=-1)   # channel dim
    face = np.expand_dims(face, axis=0)    # batch dim
    return face


# ============================================================
# IMPROVED PREDICTION WITH CONFIDENCE CALIBRATION
# ============================================================
TEMPERATURE = 1.5

def calibrated_predict(gray_face_np: np.ndarray) -> np.ndarray:
    if model is None:
        return np.array([0.1]*7) # Dummy fallback
    
    h, w = gray_face_np.shape[:2]
    variants = []

    # 1. Original
    variants.append(preprocess_face(gray_face_np))
    # 2. Horizontal flip
    variants.append(preprocess_face(cv2.flip(gray_face_np, 1)))
    # 3. Brightness +20
    variants.append(preprocess_face(np.clip(gray_face_np.astype(np.int16) + 20, 0, 255).astype(np.uint8)))
    # 4. Brightness -20
    variants.append(preprocess_face(np.clip(gray_face_np.astype(np.int16) - 20, 0, 255).astype(np.uint8)))
    # 5. Center crop 90%
    cy, cx = h // 2, w // 2
    ch, cw = max(10, int(h * 0.9)), max(10, int(w * 0.9))
    cropped = gray_face_np[max(0,cy-ch//2):max(0,cy-ch//2)+ch, max(0,cx-cw//2):max(0,cx-cw//2)+cw]
    if cropped.size > 0:
        variants.append(preprocess_face(cropped))

    # Batch predict all variants
    batch     = np.vstack(variants)
    preds     = model.predict(batch, verbose=0)
    avg_probs = preds.mean(axis=0)

    # Temperature scaling
    log_probs  = np.log(np.clip(avg_probs, 1e-9, 1.0)) / TEMPERATURE
    exp_probs  = np.exp(log_probs - np.max(log_probs))
    calibrated = exp_probs / exp_probs.sum()
    return calibrated



# ============================================================
# DNN-BASED FACE DETECTOR
# ============================================================
DNN_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

@st.cache_resource
def load_face_detector():
    import urllib.request, os, tempfile
    proto_path = os.path.join(tempfile.gettempdir(), "deploy.prototxt")
    model_path = os.path.join(tempfile.gettempdir(), "face_detector.caffemodel")
    try:
        if not os.path.exists(proto_path):
            urllib.request.urlretrieve(DNN_PROTO, proto_path)
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(DNN_MODEL, model_path)
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return ("dnn", net)
    except Exception:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return ("haar", cascade)

# Load detector at startup
_detector_type, _detector = load_face_detector()


def detect_faces_dnn(img_bgr: np.ndarray, conf_thresh: float = 0.5):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    _detector.setInput(blob)
    detections = _detector.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def detect_faces_haar(gray: np.ndarray):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def get_faces(img_bgr: np.ndarray):
    if _detector_type == "dnn":
        return detect_faces_dnn(img_bgr), "DNN"
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return detect_faces_haar(gray), "Haar"


def hex_to_bgr(hex_color: str):
    return (
        int(hex_color[5:7], 16),
        int(hex_color[3:5], 16),
        int(hex_color[1:3], 16),
    )


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================
def detect_faces_and_predict(pil_img):
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces, detector_used = get_faces(img_bgr)
    results = []
    
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        calibrated_probs = calibrated_predict(face_gray)
        idx      = np.argmax(calibrated_probs)
        emotion  = CLASS_NAMES[idx]
        conf     = float(calibrated_probs[idx] * 100)
        results.append((emotion, conf, calibrated_probs))

        bgr_color = hex_to_bgr(COLOR_MAP[emotion])
        thickness = max(2, int(min(w, h) / 80)) 
        label = f"{emotion}  {conf:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(img_bgr, (x, y - lh - 14), (x + lw + 6, y), bgr_color, -1)
        cv2.putText(img_bgr, label, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), bgr_color, thickness)

    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated, results, _detector_type

def add_to_history(emotion, confidence, source="Image"):
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "emotion": emotion,
        "confidence": round(confidence, 1),
        "source": source
    })

def img_to_bytes(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return buf.getvalue()

def render_confidence_bars(pred_array):
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, pred_array)):
        pct = float(prob) * 100
        color = CHART_COLORS[i]
        st.markdown(f"""
        <div style="margin-bottom:6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:2px;">
                <span>{EMOJI_MAP[name]} {name}</span>
                <span style="color:{color}; font-weight:600;">{pct:.1f}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.3rem; font-weight:700; letter-spacing:0.04em;">EmotiSense AI</div>
        <div style="font-size:0.78rem; opacity:0.6; margin-top:4px;">Facial Emotion Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    pages = ["🏠 Home", "🖼️ Image Detection", "🎥 Live Detection", "📊 Analytics Dashboard"]
    for p in pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p

    st.markdown("---")

    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    conf_threshold = st.slider("Confidence Threshold %", 0, 100, 40, help="Minimum confidence to show result")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Session Stats</div>', unsafe_allow_html=True)
    total = len(st.session_state.history)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:8px;">
        <h2>{total}</h2>
        <p>Detections Made</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        top = pd.DataFrame(st.session_state.history)["emotion"].value_counts().idxmax()
        st.markdown(f"""
        <div class="metric-card">
            <h2>{EMOJI_MAP.get(top,'')}</h2>
            <p>Top Emotion: {top}</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
        
    # NEW FEATURE: About Section in Sidebar
    st.markdown("---")
    with st.expander("ℹ️ About App"):
        st.markdown("""
        **Developer:** Haroon Rashid  
        **Tech:** Python, Streamlit, Keras  
        This project acts as an advanced computer vision module suitable for modern ML portfolios.
        """)

page = st.session_state.page

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.markdown('<div class="page-title">Welcome to EmotiSense AI 🧠</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">A professional AI-powered facial emotion detection system</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🖼️", "Image Detection", "Upload any photo and detect emotions from faces with confidence scores."),
        (c2, "🎥", "Live Detection", "Real-time emotion detection using your webcam feed."),
        (c3, "📊", "Analytics", "Visualize emotion history with charts, stats, and CSV export."),
    ]:
        with col:
            st.markdown(f"""
            <div class="emotion-card" style="text-align:center; cursor:pointer;">
                <div style="font-size:2.5rem;">{icon}</div>
                <div style="font-size:1.1rem; font-weight:600; margin:8px 0 6px;">{title}</div>
                <div style="font-size:0.88rem; opacity:0.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎭 Detectable Emotions")
    cols = st.columns(7)
    for col, name in zip(cols, CLASS_NAMES):
        with col:
            badge = BADGE_CLASS[name]
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:2rem;">{EMOJI_MAP[name]}</div>
                <div class="emotion-badge {badge}" style="font-size:0.78rem; padding:4px 10px;">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Model Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    pipeline_info = [
        ("🎯", "Face Detection", "OpenCV DNN\nSSD ResNet-10\n(Haar fallback)"),
        ("🖼️", "Preprocessing", "CLAHE Contrast\n+ Gaussian Denoise\n+ Normalize"),
        ("🧠", "CNN Model", "Keras trained\n48×48 grayscale\n7 emotions"),
        ("📐", "Calibration", "Temperature\nScaling (T=1.5)\nHonest confidence"),
    ]
    for col, (icon, title, desc) in zip([p1,p2,p3,p4], pipeline_info):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem; margin-bottom:4px;">{icon}</div>
                <h2 style="font-size:0.95rem;">{title}</h2>
                <p style="white-space:pre-line; font-size:0.78rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    for col, t, v in [(c1,"Model","CNN (Keras)"), (c2,"CV","OpenCV"), (c3,"UI","Streamlit"), (c4,"Data","Pandas + Altair")]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="font-size:1rem;">{t}</h2>
                <p>{v}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# IMAGE DETECTION
# ============================================================
elif page == "🖼️ Image Detection":
    st.markdown('<div class="page-title">🖼️ Image Emotion Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a photo — AI detects faces and emotions instantly</div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1.1, 1.6, 1.3])

    with left:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop or select image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("")

        if uploaded:
            st.markdown('<div class="section-header">Controls</div>', unsafe_allow_html=True)
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
            if st.button("⬇️ Download Image", use_container_width=True, key="dl_img"):
                st.session_state._trigger_download = True

    with mid:
        st.markdown('<div class="section-header">Detection Result</div>', unsafe_allow_html=True)

        if uploaded:
            pil_img = Image.open(uploaded)
            annotated, results, det_type = detect_faces_and_predict(pil_img)

            det_badge = "🟢 DNN (SSD ResNet)" if det_type == "dnn" else "🟡 Haar Cascade"
            st.markdown(f'<div class="toast" style="font-size:0.79rem;">⚙️ Detector: <strong>{det_badge}</strong> &nbsp;|&nbsp; Pre-process: <strong>CLAHE + Denoise</strong> &nbsp;|&nbsp; Calibration: <strong>Temperature Scaling</strong></div>', unsafe_allow_html=True)

            if not results:
                st.markdown("""
                <div class="toast">⚠️ No faces detected. Try a clearer, front-facing photo.</div>
                """, unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
            else:
                st.image(annotated, use_container_width=True, caption=f"✅ {len(results)} face(s) detected")

                for i, (emotion, conf, pred_arr) in enumerate(results):
                    badge = BADGE_CLASS[emotion]
                    if conf >= conf_threshold:
                        st.markdown(f"""
                        <div class="toast">
                            ✅ Face {i+1}: 
                            <span class="emotion-badge {badge}" style="padding:3px 12px; margin:0 6px;">
                                {EMOJI_MAP[emotion]} {emotion}
                            </span>
                            — Confidence: <strong>{conf:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        add_to_history(emotion, conf, "Image")
                    else:
                        st.markdown(f"""
                        <div class="toast" style="border-left-color:#f44336;">
                            ⚠️ Face {i+1}: Low confidence ({conf:.1f}%) — below threshold
                        </div>
                        """, unsafe_allow_html=True)

                # Screenshot download
                img_bytes = img_to_bytes(annotated)
                st.download_button("📸 Download Annotated Image", img_bytes,
                                   file_name="emotisense_result.png", mime="image/png",
                                   use_container_width=True)
        else:
            st.markdown("""
            <div class="emotion-card" style="text-align:center; padding:60px 24px;">
                <div style="font-size:3rem; margin-bottom:12px;">📂</div>
                <div style="font-size:1rem; opacity:0.6;">Upload an image to begin detection</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Emotion Probabilities</div>', unsafe_allow_html=True)

        if uploaded and results:
            # Show bars for first face
            _, _, pred_arr = results[0]
            render_confidence_bars(pred_arr)
            
            # NEW FEATURE: Detailed Face Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Detailed Raw Metrics"):
                prob_df = pd.DataFrame([pred_arr], columns=CLASS_NAMES)
                st.dataframe(prob_df.style.format("{:.2%}"))

            st.markdown("")
            st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

            all_data = []
            for i, (em, cf, pa) in enumerate(results):
                for name, prob in zip(CLASS_NAMES, pa):
                    all_data.append({
                        "Face": i+1, "Emotion": name,
                        "Probability": round(float(prob)*100, 2),
                        "Detected": em, "Confidence": round(cf, 2)
                    })
            df = pd.DataFrame(all_data)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export CSV", csv, "emotisense_results.csv",
                               "text/csv", use_container_width=True)
        else:
            st.markdown("""
            <div class="emotion-card" style="text-align:center; padding:40px 16px;">
                <div style="font-size:2rem; margin-bottom:8px;">📊</div>
                <div style="font-size:0.85rem; opacity:0.6;">Probabilities appear here after detection</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# LIVE DETECTION
# ============================================================
elif page == "🎥 Live Detection":
    st.markdown('<div class="page-title">🎥 Live Emotion Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time webcam-based facial emotion recognition</div>', unsafe_allow_html=True)

    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        import av

        class EmotionDetector(VideoTransformerBase):
            def __init__(self):
                self.frame_count = 0
                self.face_cache = {}

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces, _ = get_faces(img)
                new_cache = {}
                for (x, y, w, h) in faces:
                    cache_key = (round(x/20)*20, round(y/20)*20)
                    if self.frame_count % 4 == 0 or cache_key not in self.face_cache:
                        face_gray = gray[y:y+h, x:x+w]
                        probs     = calibrated_predict(face_gray)
                        idx       = np.argmax(probs)
                        emotion   = CLASS_NAMES[idx]
                        conf      = float(probs[idx] * 100)
                        new_cache[cache_key] = (emotion, conf, probs)
                    else:
                        new_cache[cache_key] = self.face_cache[cache_key]

                    em, cf, _ = new_cache[cache_key]
                    bgr       = hex_to_bgr(COLOR_MAP.get(em, "#ffffff"))
                    label = f"{em} {cf:.0f}%"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x, y - lh - 12), (x + lw + 6, y), bgr, -1)
                    cv2.putText(img, label, (x + 3, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.rectangle(img, (x, y), (x+w, y+h), bgr, 2)

                self.face_cache = new_cache
                return img

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="section-header">Camera Feed</div>', unsafe_allow_html=True)
            webrtc_streamer(
                key="emotisense_live",
                video_transformer_factory=EmotionDetector,
                media_stream_constraints={"video": True, "audio": False},
            )

        with c2:
            st.markdown('<div class="section-header">Live Tips</div>', unsafe_allow_html=True)
            tips = [
                ("💡", "Face the camera directly"),
                ("💡", "Good lighting = better results"),
                ("💡", "Stay ~50cm from camera"),
                ("💡", "Predictions update every 5 frames"),
                ("💡", "Multiple faces detected simultaneously"),
            ]
            for icon, tip in tips:
                st.markdown(f"""
                <div class="toast">{icon} {tip}</div>
                """, unsafe_allow_html=True)

    except ImportError:
        st.markdown("""
        <div class="emotion-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem; margin-bottom:16px;">📷</div>
            <div style="font-size:1.1rem; font-weight:600; margin-bottom:8px;">Live Mode Requires Extra Package</div>
            <div style="font-size:0.9rem; opacity:0.7; margin-bottom:20px;">Install streamlit-webrtc to enable full live video webcam support</div>
            <code style="background:#1e2130; padding:10px 20px; border-radius:8px; display:block; margin:0 auto; max-width:360px;">
                pip install streamlit-webrtc
            </code>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # NEW FEATURE: Native Snapshot Mode Alternative
    cam_col, img_col = st.columns(2)
    
    with cam_col:
        st.markdown("### 📸 Native Snapshot Mode")
        st.markdown("<p style='font-size: 0.9rem; opacity: 0.8;'>Capture directly from browser without extra packages.</p>", unsafe_allow_html=True)
        cam_image = st.camera_input("Take a picture")
        if cam_image:
            pil = Image.open(cam_image)
            annotated, results, _ = detect_faces_and_predict(pil)
            st.image(annotated, use_container_width=True)
            for em, cf, _ in results:
                badge = BADGE_CLASS[em]
                st.markdown(f"""
                <div class="toast">
                    {EMOJI_MAP[em]}
                    <span class="emotion-badge {badge}" style="padding:3px 12px;">{em}</span>
                    — {cf:.1f}%
                </div>
                """, unsafe_allow_html=True)
                add_to_history(em, cf, "Live-Snapshot")

    with img_col:
        st.markdown("### 🖼️ Test with Image Instead")
        st.markdown("<p style='font-size: 0.9rem; opacity: 0.8;'>Upload a picture from your device manually.</p>", unsafe_allow_html=True)
        test_img = st.file_uploader("Upload test image", type=["jpg","jpeg","png"])
        if test_img:
            pil = Image.open(test_img)
            annotated, results, _ = detect_faces_and_predict(pil)
            st.image(annotated, use_container_width=True)
            for em, cf, _ in results:
                badge = BADGE_CLASS[em]
                st.markdown(f"""
                <div class="toast">
                    {EMOJI_MAP[em]}
                    <span class="emotion-badge {badge}" style="padding:3px 12px;">{em}</span>
                    — {cf:.1f}%
                </div>
                """, unsafe_allow_html=True)
                add_to_history(em, cf, "Live-Test")


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================
elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="page-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Emotion detection history, trends, and statistics</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="emotion-card" style="text-align:center; padding:60px 24px;">
            <div style="font-size:3rem; margin-bottom:12px;">📭</div>
            <div style="font-size:1.1rem; font-weight:600; margin-bottom:6px;">No Data Yet</div>
            <div style="font-size:0.9rem; opacity:0.6;">
                Run some detections in Image or Live mode — results will appear here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(st.session_state.history)

        # ── KPI Row
        total = len(df)
        top_em = df["emotion"].value_counts().idxmax()
        avg_conf = df["confidence"].mean()
        sources = df["source"].nunique()

        k1, k2, k3, k4 = st.columns(4)
        for col, val, label in [
            (k1, total, "Total Detections"),
            (k2, f"{EMOJI_MAP[top_em]} {top_em}", "Top Emotion"),
            (k3, f"{avg_conf:.1f}%", "Avg Confidence"),
            (k4, sources, "Detection Sources"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{val}</h2>
                    <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Charts Row
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="section-header">Emotion Distribution</div>', unsafe_allow_html=True)
            counts = df["emotion"].value_counts().reset_index()
            counts.columns = ["Emotion", "Count"]
            counts["Color"] = counts["Emotion"].map(COLOR_MAP)
            bar = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Emotion:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q"),
                    color=alt.Color("Emotion:N",
                        scale=alt.Scale(
                            domain=list(COLOR_MAP.keys()),
                            range=list(COLOR_MAP.values())
                        ), legend=None),
                    tooltip=["Emotion", "Count"]
                )
                .properties(height=280)
            )
            st.altair_chart(bar, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-header">Confidence Over Time</div>', unsafe_allow_html=True)
            df_idx = df.reset_index().rename(columns={"index": "Detection #"})
            line = (
                alt.Chart(df_idx)
                .mark_line(point=True, strokeWidth=2)
                .encode(
                    x=alt.X("Detection #:Q"),
                    y=alt.Y("confidence:Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("emotion:N",
                        scale=alt.Scale(
                            domain=list(COLOR_MAP.keys()),
                            range=list(COLOR_MAP.values())
                        )),
                    tooltip=["time", "emotion", "confidence", "source"]
                )
                .properties(height=280)
            )
            st.altair_chart(line, use_container_width=True)

        # ── Pie / Donut
        st.markdown('<div class="section-header">Emotion Share</div>', unsafe_allow_html=True)
        pie = (
            alt.Chart(counts)
            .mark_arc(innerRadius=55, outerRadius=110)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Emotion:N",
                    scale=alt.Scale(
                        domain=list(COLOR_MAP.keys()),
                        range=list(COLOR_MAP.values())
                    )),
                tooltip=["Emotion", "Count"]
            )
            .properties(height=260)
        )
        c_pie, c_hist = st.columns(2)
        with c_pie:
            st.altair_chart(pie, use_container_width=True)
        with c_hist:
            st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
            hist = (
                alt.Chart(df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("confidence:Q", bin=alt.Bin(maxbins=10), title="Confidence %"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.value("#7c6af7"),
                    tooltip=["count()"]
                )
                .properties(height=240)
            )
            st.altair_chart(hist, use_container_width=True)

        # ── History Log
        st.markdown('<div class="section-header">Detection History Log</div>', unsafe_allow_html=True)
        cols_h = st.columns([1, 2, 2, 2, 2])
        for col, header in zip(cols_h, ["#", "Time", "Emotion", "Confidence", "Source"]):
            col.markdown(f"**{header}**")

        for i, row in df.iloc[::-1].iterrows():
            badge = BADGE_CLASS[row["emotion"]]
            c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])
            c1.markdown(f"`{i+1}`")
            c2.markdown(row["time"])
            c3.markdown(f"""<span class="emotion-badge {badge}" style="padding:2px 10px; font-size:0.8rem;">{EMOJI_MAP[row['emotion']]} {row['emotion']}</span>""", unsafe_allow_html=True)
            c4.markdown(f"**{row['confidence']}%**")
            c5.markdown(row["source"])

        # ── Export
        st.markdown("")
        csv_full = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Full History (CSV)",
            csv_full,
            "emotisense_history.csv",
            "text/csv",
            use_container_width=True
        )