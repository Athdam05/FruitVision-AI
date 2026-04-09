import streamlit as st
import os

try:
    import cv2
except ImportError:
    # Streamlit Cloud OpenCV GUI Dependency Fixer
    st.warning("First-time setup: Optimizing OpenCV for Streamlit Cloud... This will take a few seconds, please wait.")
    os.system("pip uninstall -y opencv-python opencv-python-headless")
    os.system("pip install opencv-python-headless")
    import cv2

import numpy as np
from detection import FruitDetector
from utils import FPSCounter, generate_csv_report
import tempfile
import time
import os
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="FruitVision AI",
    layout="wide",
    page_icon="🍎",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# PREMIUM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF4B4B, #FF8F70, #FFD166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8892A4;
    margin-top: 0.4rem;
}

/* Metric Cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
.metric-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: #FF4B4B;
    display: block;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8892A4;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Section Headers */
.section-header {
    font-size: 0.8rem;
    font-weight: 700;
    color: #FF4B4B;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,75,75,0.3);
}

/* Status Badge */
.badge-ok   { background:#1a3a2a; border:1px solid #2ecc71; color:#2ecc71; border-radius:8px; padding:0.4rem 0.8rem; font-size:0.85rem; }
.badge-warn { background:#3a2a1a; border:1px solid #f39c12; color:#f39c12; border-radius:8px; padding:0.4rem 0.8rem; font-size:0.85rem; }
.badge-wait { background:#1a2a3a; border:1px solid #3498db; color:#3498db; border-radius:8px; padding:0.4rem 0.8rem; font-size:0.85rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,15,26,0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label { color: #CDD5E0 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

<div class="hero-header">
  <p class="hero-title">🍎 FruitVision AI</p>
  <p class="hero-subtitle">High-Accuracy Fruit Detection & Counting Engine &nbsp;|&nbsp; YOLOv8</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    model_choice = st.selectbox(
        "🧠 Model Weights",
        ["best.pt  (Custom Trained ✅)", "yolov8n.pt  (COCO - Fast)", "yolov8s.pt  (COCO - Balanced)"]
    )
    model_file = "best.pt" if "best.pt" in model_choice else model_choice.split(" ")[0]

    conf_threshold = st.slider("🎚️ Confidence Threshold", 0.10, 1.00, 0.25, 0.05,
                                help="Lower = more detections. Higher = fewer but more certain.")
    iou_threshold = st.slider("🔗 IoU Threshold (NMS)", 0.10, 0.90, 0.45, 0.05,
                               help="Higher = allows MORE overlapping boxes (good for dense fruits). Lower = aggressive removal of duplicates.")
    
    st.markdown("---")
    enable_sahi = st.toggle("🧩 Enable Deep Slicing (SAHI)", value=True,
                                 help="Slices image into overlapping patches to perfectly detect densely packed fruits.")
    if enable_sahi:
        sahi_slice = st.slider("✂️ Slice Size (SAHI)", 256, 1280, 416, 32,
                                help="Smaller slices zoom in on dense areas. Use 320-416 for incredibly dense/tiny fruits.")
        img_size = 1280 # Unused in SAHI, but kept for fallback
    else:
        img_size = st.slider("🔍 Detection Resolution", 640, 1920, 1280, 320,
                               help="Higher resolution helps detect tiny/packed fruits more accurately.")
        sahi_slice = 640

    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(255,75,75,0.08);border:1px solid rgba(255,75,75,0.3);border-radius:10px;padding:0.8rem;'>
      <p style='color:#FF4B4B;font-weight:700;margin:0 0 4px 0;'>💡 Pro Tip</p>
      <p style='color:#8892A4;font-size:0.82rem;margin:0;'>Custom weights (best.pt) trained on your fruit dataset give the highest accuracy. Use COCO models only for quick demos.</p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
def load_detector(model_name):
    return FruitDetector(model_name)

detector = load_detector(model_file)
fps_counter = FPSCounter()

if detector.model is None:
    st.error(f"⚠️ Model `{model_file}` could not be loaded. Check file exists in the project folder.")
    st.stop()

# ──────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────
left_col, right_col = st.columns([2.5, 1.5], gap="large")

with left_col:
    st.markdown('<div class="section-header">🎥 Detection Feed</div>', unsafe_allow_html=True)
    feed_placeholder = st.empty()
    fps_placeholder  = st.empty()

with right_col:
    st.markdown('<div class="section-header">📊 Live Analytics</div>', unsafe_allow_html=True)
    total_metric  = st.empty()
    status_badge  = st.empty()
    st.markdown('<div class="section-header" style="margin-top:1rem;">🍇 Per-Category Counts</div>', unsafe_allow_html=True)
    table_placeholder = st.empty()
    st.markdown('<div class="section-header" style="margin-top:1rem;">📈 Distribution Chart</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    download_placeholder = st.empty()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def process_image(frame):
    """
    Standard full-image detection or SAHI overlapping patch detection.
    """
    if enable_sahi:
        boxes, confs, clss = detector.detect_sahi(frame, slice_size=sahi_slice, overlap=0.25, 
                                                  conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        annotated = detector.draw_manual_boxes(frame, boxes, confs, clss)
        
        counts = {}
        for cls_id in clss:
            cls_name = detector.labels.get(cls_id, "Unknown")
            counts[cls_name] = counts.get(cls_name, 0) + 1
        total = sum(counts.values())
    else:
        result = detector.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold, imgsz=img_size)
        annotated = detector.draw_boxes(frame, result)
        counts = {}
        for box in result.boxes:
            cls_name = detector.labels.get(int(box.cls[0]), "Unknown")
            counts[cls_name] = counts.get(cls_name, 0) + 1
        total = sum(counts.values())
        
    fps = fps_counter.update()
    return annotated, counts, total, fps


def render_ui(frame_rgb, counts, total, fps):
    # Feed
    feed_placeholder.image(frame_rgb)
    fps_placeholder.caption(f"⏱️ Speed: **{fps:.1f} FPS**  ·  Objects detected: **{total}**")

    # Total Metric Card
    total_metric.markdown(f"""
    <div class="metric-card">
        <span class="metric-number">{total}</span>
        <span class="metric-label">Total Fruits Detected</span>
    </div>
    """, unsafe_allow_html=True)

    # Status Badge
    if total == 0:
        status_badge.markdown('<span class="badge-wait">⏳ Waiting for objects…</span>', unsafe_allow_html=True)
    elif total > 15:
        status_badge.markdown(f'<span class="badge-warn">🚨 High Density — {total} objects</span>', unsafe_allow_html=True)
    else:
        status_badge.markdown(f'<span class="badge-ok">✅ {total} fruits tracked successfully</span>', unsafe_allow_html=True)

    # Table
    if counts:
        df = pd.DataFrame(counts.items(), columns=["Fruit", "Count"]).sort_values("Count", ascending=False)
        table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

        # Chart
        chart = alt.Chart(df).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("Count:Q", axis=alt.Axis(grid=False)),
            y=alt.Y("Fruit:N", sort="-x", axis=alt.Axis(labelColor="#CDD5E0")),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Fruit", "Count"]
        ).properties(height=200, background="transparent").configure_axis(
            domainColor="#333", tickColor="#333"
        ).configure_view(strokeWidth=0)
        chart_placeholder.altair_chart(chart, use_container_width=True)
    else:
        table_placeholder.info("No detections yet.")

# ──────────────────────────────────────────────
# INPUT HANDLER
# ──────────────────────────────────────────────
img_file = st.sidebar.file_uploader("Upload a fruit image", type=["png", "jpg", "jpeg"])
if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    annotated, counts, total, fps = process_image(frame)
    render_ui(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), counts, total, fps)

    # CSV export
    csv_path = generate_csv_report(counts, total)
    with open(csv_path, "r") as f:
        download_placeholder.download_button("📥 Export CSV Report", f, "fruit_report.csv", "text/csv")
else:
    feed_placeholder.info("👈 Upload an image from the sidebar to start detection.")

