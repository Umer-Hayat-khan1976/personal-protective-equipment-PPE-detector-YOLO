"""
Site Safety Inspector
Object detection web app for construction-site PPE, powered by a fine-tuned YOLOv8 model.
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
SAMPLES_DIR = BASE_DIR / "samples"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="Site Safety Inspector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stSlider {
        font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
    }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

    .app-header {
        background: #1B2430;
        border-left: 6px solid #F2B01E;
        border-radius: 4px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        color: #FFFFFF; font-size: 1.75rem; font-weight: 600;
        letter-spacing: -0.01em; margin: 0 0 0.35rem 0; padding: 0;
    }
    .app-header p { color: #B9C2CE; font-size: 0.97rem; margin: 0; max-width: 60ch; line-height: 1.5; }

    .panel-title {
        font-size: 0.95rem; font-weight: 600; color: #1B2430;
        padding-bottom: 0.4rem; margin-bottom: 0.75rem;
        border-bottom: 2px solid #1B2430;
    }

    .kpi {
        background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 4px;
        padding: 0.9rem 1.05rem; height: 100%;
    }
    .kpi .label { font-size: 0.8rem; color: #5B6675; margin-bottom: 0.2rem; }
    .kpi .value { font-size: 1.7rem; font-weight: 600; color: #1B2430; font-variant-numeric: tabular-nums; line-height: 1.2; }
    .kpi .unit  { font-size: 0.85rem; font-weight: 400; color: #5B6675; margin-left: 0.2rem; }

    .status {
        border-radius: 4px; padding: 0.8rem 1.05rem; margin: 1rem 0 1.25rem 0;
        font-size: 0.95rem; line-height: 1.45; border: 1px solid;
    }
    .status.alert { background: #FDEDED; border-color: #E5A3A3; color: #7A1F1F; }
    .status.ok    { background: #EAF5EE; border-color: #A5CFB3; color: #1D5132; }
    .status.info  { background: #EEF1F5; border-color: #C9D0DA; color: #1B2430; }
    .status b { font-weight: 600; }

    .empty {
        background: #FFFFFF; border: 1px dashed #AEB7C4; border-radius: 4px;
        padding: 2.5rem 1.5rem; text-align: center; color: #5B6675;
    }
    .empty .t { font-size: 1.05rem; font-weight: 600; color: #1B2430; margin-bottom: 0.35rem; }

    section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9DEE5; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem; font-weight: 600; color: #1B2430;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 4px; font-weight: 500; border: 1px solid #1B2430;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def is_violation(name: str) -> bool:
    """Classes prefixed with 'NO-' (e.g. NO-Hardhat) mark missing protective equipment."""
    return name.lower().replace("_", "-").replace(" ", "-").startswith("no-")


def run_inference(model: YOLO, image: Image.Image, conf: float, iou: float,
                  imgsz: int, class_ids):
    bgr = np.array(image)[..., ::-1]  # PIL RGB -> BGR expected by Ultralytics
    start = time.perf_counter()
    result = model.predict(
        source=bgr, conf=conf, iou=iou, imgsz=imgsz,
        classes=class_ids or None, verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = Image.fromarray(result.plot(line_width=3)[..., ::-1])

    rows = []
    for i, box in enumerate(result.boxes, start=1):
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append({
            "ID": i,
            "Class": name,
            "Confidence": float(box.conf[0]),
            "Category": "Violation" if is_violation(name) else "Detected",
            "Box (x1, y1, x2, y2)": f"{x1}, {y1}, {x2}, {y2}",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)
        df["ID"] = range(1, len(df) + 1)
    return annotated, df, elapsed_ms


def kpi(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}{unit_html}</div></div>')


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Site Safety Inspector</h1>
        <p>Upload a photo from a construction site. The model locates workers and checks
        for hard hats, masks and safety vests, flagging any missing equipment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.markdown(
        '<div class="status alert"><b>Model file not found.</b> Place the trained weights at '
        '<code>models/best.pt</code> and restart the app. See the README for training steps.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

model = load_model(str(MODEL_PATH))
class_names = [model.names[i] for i in sorted(model.names)]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Detection settings")
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.40, 0.05,
        help="Detections scoring below this value are hidden.",
    )
    iou = st.slider(
        "Overlap threshold (NMS IoU)", 0.10, 0.90, 0.45, 0.05,
        help="Lower values remove more overlapping boxes.",
    )
    imgsz = st.select_slider("Inference image size", options=[320, 416, 512, 640, 800], value=640)

    selected = st.multiselect(
        "Classes to detect", options=class_names, default=class_names,
        help="Remove a class to exclude it from the results.",
    )
    class_ids = [i for i, n in model.names.items() if n in selected]

    st.markdown("---")
    st.markdown("### Model")
    st.caption("Architecture: YOLOv8 (fine-tuned)")
    st.caption(f"Classes: {len(class_names)}")
    st.caption("Task: object detection")

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
st.markdown('<div class="panel-title">Image input</div>', unsafe_allow_html=True)

sample_files = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower().lstrip(".") in ALLOWED_TYPES)

source_col, sample_col = st.columns([3, 2], gap="large")
with source_col:
    uploaded = st.file_uploader(
        "Upload an image", type=ALLOWED_TYPES, label_visibility="collapsed",
        help="JPG, PNG or WEBP, up to 20 MB.",
    )
with sample_col:
    sample_choice = None
    if sample_files:
        sample_choice = st.selectbox(
            "Or use a sample image", ["None"] + [p.name for p in sample_files],
        )
        sample_choice = None if sample_choice == "None" else sample_choice

image = None
image_label = ""
if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    image_label = uploaded.name
elif sample_choice:
    image = ImageOps.exif_transpose(Image.open(SAMPLES_DIR / sample_choice)).convert("RGB")
    image_label = sample_choice

if image is None:
    st.markdown(
        '<div class="empty"><div class="t">No image selected</div>'
        'Upload a JPG, PNG or WEBP file to start the inspection.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not selected:
    st.markdown('<div class="status info">Select at least one class in the sidebar.</div>',
                unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
with st.spinner("Running detection"):
    annotated, df, elapsed_ms = run_inference(model, image, conf, iou, imgsz, class_ids)

total = len(df)
violations = int((df["Category"] == "Violation").sum()) if total else 0
mean_conf = float(df["Confidence"].mean()) if total else 0.0
classes_found = df["Class"].nunique() if total else 0

# Status banner
if total == 0:
    st.markdown(
        '<div class="status info"><b>No objects detected.</b> Lower the confidence threshold '
        'or try a different image.</div>', unsafe_allow_html=True)
elif violations:
    kinds = ", ".join(sorted(df.loc[df["Category"] == "Violation", "Class"].unique()))
    st.markdown(
        f'<div class="status alert"><b>{violations} safety violation'
        f'{"s" if violations != 1 else ""} detected.</b> Missing equipment flagged: {kinds}.</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="status ok"><b>No missing-equipment violations detected</b> at the current '
        'confidence threshold.</div>', unsafe_allow_html=True)

# KPIs
k1, k2, k3, k4 = st.columns(4, gap="small")
k1.markdown(kpi("Objects detected", str(total)), unsafe_allow_html=True)
k2.markdown(kpi("Violations", str(violations)), unsafe_allow_html=True)
k3.markdown(kpi("Average confidence", f"{mean_conf * 100:.1f}", "%"), unsafe_allow_html=True)
k4.markdown(kpi("Inference time", f"{elapsed_ms:.0f}", "ms"), unsafe_allow_html=True)

st.write("")

# Images
left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="panel-title">Original image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with right:
    st.markdown('<div class="panel-title">Detection result</div>', unsafe_allow_html=True)
    st.image(annotated, use_container_width=True)
    st.download_button(
        "Download annotated image", data=to_png_bytes(annotated),
        file_name=f"detected_{Path(image_label).stem}.png", mime="image/png",
    )

st.write("")

# Results
tab_list, tab_summary = st.tabs(["Detected objects", "Class summary"])

with tab_list:
    if total:
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"detections_{Path(image_label).stem}.csv", mime="text/csv",
        )
    else:
        st.caption("No detections to display.")

with tab_summary:
    if total:
        summary = (
            df.groupby("Class")
            .agg(Count=("Class", "size"),
                 **{"Mean confidence": ("Confidence", "mean"),
                    "Highest confidence": ("Confidence", "max")})
            .sort_values("Count", ascending=False)
            .reset_index()
        )
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.dataframe(
                summary, use_container_width=True, hide_index=True,
                column_config={
                    "Mean confidence": st.column_config.NumberColumn(format="%.2f"),
                    "Highest confidence": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        with c2:
            st.bar_chart(summary.set_index("Class")["Count"], color="#1B2430")
    else:
        st.caption("No detections to summarise.")
