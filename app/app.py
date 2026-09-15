import os
import sys
import io
import hashlib
from datetime import datetime

import cv2
import pandas as pd
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# IMPORT EXISTING MODEL
# ============================================================

try:
    from src.models.model import HemorrhageClassifier
    MODEL_IMPORT_ERROR = None

except Exception as e:
    HemorrhageClassifier = None
    MODEL_IMPORT_ERROR = str(e)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based Intracranial Hemorrhage Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "No Hemorrhage",
    "Epidural",
    "Intraparenchymal",
    "Subdural",
    "Intraventricular",
    "Subarachnoid"
]

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "outputs",
    "models",
    "efficientnetv2_best.pth"
)

GRADCAM_PATH = os.path.join(
    ROOT_DIR,
    "outputs",
    "plots",
    "gradcam_result.png"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "analysis_done": False,
    "predicted_class": None,
    "confidence": 0.0,
    "probabilities": None,
    "scan_count": 0,
    "abnormal_count": 0,
    "normal_count": 0,
    "current_file_hash": None,
    "gradcam_image": None,
    "raw_cam_mask": None,
    "raw_rgb_img": None,
    "cam_opacity": 0.45,
    "cam_colormap": "JET",
    "cam_threshold": 0.0,
    "ct_window_mode": "Standard"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ========================================================
GLOBAL
======================================================== */

#MainMenu {
visibility: hidden;
}

footer {
visibility: hidden;
}

header {
visibility: hidden;
}

.stApp {

background:
radial-gradient(
circle at 15% 10%,
rgba(35, 115, 190, 0.14),
transparent 28%
),

radial-gradient(
circle at 90% 80%,
rgba(35, 160, 130, 0.08),
transparent 25%
),

#07111f;
}

.block-container {

max-width: 1450px;

padding-top: 25px;
padding-bottom: 45px;
}


/* ========================================================
HEADER
======================================================== */

.app-title {

font-size: 40px;
font-weight: 800;

color: #f1f7ff;

letter-spacing: -1px;

margin-bottom: 4px;
}

.app-subtitle {

color: #94a8bd;

font-size: 14px;

margin-bottom: 24px;
}


/* ========================================================
SECTION HEADINGS
======================================================== */

.section-title {

font-size: 18px;

font-weight: 800;

color: #e8f1fb;

margin-top: 18px;

margin-bottom: 10px;
}


/* ========================================================
METRIC CARDS
======================================================== */

.metric-card {

background:
linear-gradient(
145deg,
rgba(20, 42, 68, 0.98),
rgba(13, 29, 48, 0.98)
);

border: 1px solid rgba(110, 160, 210, 0.25);

border-radius: 15px;

padding: 18px;

text-align: center;

min-height: 88px;

box-shadow:
0 8px 25px rgba(0,0,0,0.15);
}

.metric-number {

font-size: 28px;

font-weight: 800;

color: #ffffff;
}

.metric-name {

font-size: 11px;

color: #8ea5bc;

margin-top: 5px;

letter-spacing: 0.4px;
}

[data-testid="stMetricValue"] {
font-size: 1.15rem !important;
white-space: normal !important;
word-break: break-word !important;
overflow-wrap: break-word !important;
line-height: 1.3 !important;
}

[data-testid="stMetricLabel"] {
font-size: 0.85rem !important;
color: #8ea5bc !important;
}


/* ========================================================
UPLOAD
======================================================== */

.upload-header {

font-size: 15px;

font-weight: 750;

color: #eaf3fc;

margin-bottom: 7px;
}

[data-testid="stFileUploader"] {

background: rgba(20, 37, 58, 0.72);

border: 1px solid rgba(110, 160, 205, 0.18);

border-radius: 14px;

padding: 8px;
}


/* ========================================================
STATUS
======================================================== */

.status-success {

background: rgba(20, 90, 76, 0.28);

border: 1px solid rgba(60, 210, 165, 0.35);

border-left: 4px solid #31c48d;

border-radius: 12px;

padding: 12px 15px;

color: #d9f9ee;

margin-top: 12px;

margin-bottom: 12px;
}


/* ========================================================
PANEL HEADINGS
======================================================== */

.panel-heading {

color: #eaf3fc;

font-size: 17px;

font-weight: 800;

margin-top: 3px;

margin-bottom: 10px;
}


/* ========================================================
RESULT CARD
======================================================== */

.result-normal {

background:
linear-gradient(
145deg,
rgba(16, 110, 72, 0.35),
rgba(11, 65, 52, 0.25)
);

border: 1px solid rgba(45, 210, 135, 0.45);

border-radius: 16px;

padding: 20px;

text-align: center;

margin-top: 10px;

margin-bottom: 12px;
}

.result-warning {

background:
linear-gradient(
145deg,
rgba(130, 40, 48, 0.38),
rgba(80, 25, 35, 0.25)
);

border: 1px solid rgba(255, 105, 105, 0.50);

border-radius: 16px;

padding: 20px;

text-align: center;

margin-top: 10px;

margin-bottom: 12px;
}

.result-small {

color: #9db1c5;

font-size: 11px;

text-transform: uppercase;

letter-spacing: 0.8px;
}

.result-name {

color: #f3f8ff;

font-size: 24px;

font-weight: 800;

margin: 8px 0;
}

.result-confidence {

color: #ffffff;

font-size: 34px;

font-weight: 850;

margin-top: 4px;
}


/* ========================================================
INFO BOX
======================================================== */

.info-box {

background: rgba(25, 43, 65, 0.75);

border-radius: 12px;

border: 1px solid rgba(120, 160, 200, 0.15);

padding: 14px;

color: #d9e8f5;

font-size: 13px;

line-height: 1.55;

margin-top: 10px;
}


/* ========================================================
IMAGE CAPTION
======================================================== */

.image-caption {

color: #8297aa;

font-size: 11px;

margin-top: 3px;
}


/* ========================================================
PROBABILITY BOX
======================================================== */

.probability-title {

color: #eaf3fc;

font-size: 18px;

font-weight: 800;

margin-top: 20px;

margin-bottom: 12px;
}

.probability-name {

color: #dbe8f5;

font-size: 12px;

font-weight: 650;
}

.probability-value {

color: #ffffff;

font-size: 12px;

font-weight: 750;

float: right;
}


/* ========================================================
SUMMARY
======================================================== */

.summary-box {

background: rgba(22, 54, 83, 0.75);

border-left: 4px solid #3ba7ff;

border-radius: 10px;

padding: 14px 16px;

color: #d9ebfa;

font-size: 13px;

line-height: 1.55;
}


/* ========================================================
REPORT AREA
======================================================== */

.report-heading {

color: #eaf3fc;

font-size: 18px;

font-weight: 800;

margin-top: 22px;

margin-bottom: 10px;
}


/* ========================================================
BUTTONS
======================================================== */

.stButton > button {

border-radius: 10px;

min-height: 42px;

font-weight: 700;
}

.stDownloadButton > button {

border-radius: 10px;

min-height: 42px;

font-weight: 700;
}


/* ========================================================
DISCLAIMER
======================================================== */

.disclaimer {

margin-top: 30px;

padding: 15px 18px;

border-radius: 13px;

background: rgba(130, 100, 30, 0.13);

border: 1px solid rgba(220, 175, 70, 0.25);

color: #aebdcb;

font-size: 12px;

line-height: 1.6;
}


</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="app-title">
🧠 AI-Based Intracranial Hemorrhage Detection
</div>

<div class="app-subtitle">
Real-time CT scan screening and explainable analysis
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# TOP STATISTICS
# ============================================================

m1, m2, m3 = st.columns(3, gap="medium")


with m1:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-number">
{st.session_state.scan_count}
</div>

<div class="metric-name">
SCANS ANALYZED
</div>

</div>
""",
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-number">
{st.session_state.abnormal_count}
</div>

<div class="metric-name">
ABNORMAL SCREENINGS
</div>

</div>
""",
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-number">
{st.session_state.normal_count}
</div>

<div class="metric-name">
NON-HEMORRHAGE SCREENINGS
</div>

</div>
""",
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    """
<div class="upload-header">
Upload Brain CT Image
</div>
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    help="Upload a CT brain image in JPG, JPEG or PNG format."
)

sample_options = {
    "None (Upload your own CT scan)": None,
    "Sample 1: Normal CT Scan (No Hemorrhage)": "normal_sample.jpg",
    "Sample 2: Intraparenchymal Hemorrhage": "intraparenchymal_sample.jpg",
    "Sample 3: Epidural Hemorrhage": "epidural_sample.jpg",
    "Sample 4: Subdural Hemorrhage": "subdural_sample.jpg",
}

selected_sample_label = st.selectbox(
    "Or try a pre-loaded sample CT scan:",
    options=list(sample_options.keys()),
    index=0
)
selected_sample_file = sample_options[selected_sample_label]

file_bytes = None
file_name = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif selected_sample_file is not None:
    sample_path = os.path.join(ROOT_DIR, "sample_images", selected_sample_file)
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
        file_name = selected_sample_file


# ============================================================
# NO IMAGE
# ============================================================

if file_bytes is None:

    st.markdown(
        """
<div class="info-box">

<b>Ready for analysis</b><br>

Upload a brain CT image above or select a sample scan to begin the
screening process.

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="disclaimer">

<b>Important:</b>

This application is an AI-assisted screening
prototype intended for academic and research
demonstration.

Results should not be considered a definitive
medical diagnosis or a substitute for professional
clinical review.

</div>
""",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# PROCESS IMAGE BYTES
# ============================================================

current_hash = hashlib.md5(file_bytes).hexdigest() if file_bytes is not None else None


# ============================================================
# DETECT NEW IMAGE
# ============================================================

if current_hash is not None and st.session_state.current_file_hash != current_hash:

    st.session_state.current_file_hash = current_hash

    st.session_state.analysis_done = False

    st.session_state.predicted_class = None

    st.session_state.confidence = 0.0

    st.session_state.probabilities = None

    st.session_state.gradcam_image = None


# ============================================================
# LOAD IMAGE
# ============================================================

if file_bytes is not None:
    try:
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")
    except Exception:
        st.error(
            "The uploaded file could not be read as an image."
        )
        st.stop()
else:
    image = None


# ============================================================
# UPLOAD STATUS
# ============================================================

st.markdown(
    """
<div class="status-success">

✓ CT scan uploaded successfully

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if HemorrhageClassifier is None:

        raise RuntimeError(
            "Unable to load the analysis component."
        )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Required analysis file was not found."
        )

    try:

        model = HemorrhageClassifier(
            num_classes=len(CLASS_NAMES)
        )

    except TypeError:

        model = HemorrhageClassifier()


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # DETERMINE STATE DICT
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # REMOVE DATA PARALLEL PREFIX
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):

            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value


    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    except RuntimeError:

        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )


    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, pil_image):

    image_tensor = transform(
        pil_image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)


    with torch.no_grad():

        output = model(
            image_tensor
        )

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        probabilities = F.softmax(
            output,
            dim=1
        )


    probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )


    predicted_index = int(
        np.argmax(probabilities)
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = float(
        probabilities[predicted_index]
    )


    return (
        predicted_class,
        confidence,
        probabilities,
        predicted_index
    )


# ============================================================
# GRAD-CAM GENERATION
# ============================================================

def apply_ct_window(pil_img, window_mode="Standard"):
    """
    Simulates clinical Hounsfield Unit (HU) windowing levels used by neuroradiologists.
    """
    if pil_img is None:
        return None
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if window_mode == "Brain Window (Soft Tissue)":
        w = np.clip((arr - 0.15) / (0.75 - 0.15), 0.0, 1.0)
    elif window_mode == "Subdural/Blood Window":
        w = np.clip((arr - 0.25) / (0.85 - 0.25), 0.0, 1.0) ** 1.3
    elif window_mode == "Bone Window":
        w = np.clip((arr - 0.60) / (1.0 - 0.60), 0.0, 1.0) ** 0.8
    else:
        w = arr
    return Image.fromarray(np.uint8(np.clip(w * 255.0, 0, 255)))


def blend_cam(rgb_img, cam_mask, opacity=0.45, colormap_name="JET", threshold=0.0):
    """
    Blends raw activation mask with input RGB image using specified colormap, opacity, and threshold.
    """
    cmaps = {
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "HOT": cv2.COLORMAP_HOT
    }
    m = np.copy(cam_mask)
    m = np.where(m >= threshold, m, 0.0)
    if m.max() > 0:
        m = m / m.max()
    cm_code = cmaps.get(colormap_name, cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * m), cm_code)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1.0 - opacity) * rgb_img + opacity * heatmap
    return Image.fromarray(np.uint8(np.clip(blended * 255.0, 0, 255)))


def generate_gradcam(model, pil_image, target_class_index):
    """
    Runs Grad-CAM on the given model for the given image and
    returns (grayscale_cam, rgb_img) for zero-latency dynamic re-blending.
    """
    resized_image = pil_image.resize((224, 224))
    rgb_img = np.array(resized_image).astype(np.float32) / 255.0

    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    target_layers = [model.model.features[-1]]
    targets = [ClassifierOutputTarget(target_class_index)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets
        )
        grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam, rgb_img


# ============================================================
# MAIN ANALYSIS LAYOUT
# ============================================================

col_image, col_analysis, col_xai = st.columns(
    [1.05, 1.0, 1.05],
    gap="large"
)


# ============================================================
# COLUMN 1 — INPUT CT
# ============================================================

with col_image:

    st.markdown(
        """
<div class="panel-heading">
🩻 Input CT Scan
</div>
""",
        unsafe_allow_html=True
    )

    if image is not None:
        window_mode = st.selectbox(
            "CT Windowing Preset",
            options=["Standard", "Brain Window (Soft Tissue)", "Subdural/Blood Window", "Bone Window"],
            index=0,
            help="Simulates clinical CT Hounsfield Unit (HU) windowing levels used by neuroradiologists."
        )

        display_img = apply_ct_window(image, window_mode)

        if display_img is not None:
            st.image(
                display_img,
                use_container_width=True
            )

        st.markdown(
            f"""
<div class="image-caption">
File: {file_name} | Window: {window_mode}
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# COLUMN 2 — ANALYSIS
# ============================================================

with col_analysis:

    st.markdown(
        """
<div class="panel-heading">
⚡ AI Analysis
</div>
""",
        unsafe_allow_html=True
    )


    analyze_button = st.button(
        "🔍 Analyze CT Scan",
        use_container_width=True,
        type="primary"
    )


    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    if analyze_button:

        try:

            with st.spinner(
                "Analyzing CT scan..."
            ):

                model = load_model()

                (
                    predicted_class,
                    confidence,
                    probabilities,
                    predicted_index
                ) = predict_image(
                    model,
                    image
                )

                raw_cam, raw_rgb = generate_gradcam(
                    model,
                    image,
                    predicted_index
                )

                st.session_state.raw_cam_mask = raw_cam
                st.session_state.raw_rgb_img = raw_rgb
                gradcam_image = blend_cam(
                    raw_rgb,
                    raw_cam,
                    st.session_state.cam_opacity,
                    st.session_state.cam_colormap,
                    st.session_state.cam_threshold
                )

            st.session_state.analysis_done = True
            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence
            st.session_state.probabilities = probabilities
            st.session_state.gradcam_image = gradcam_image


            st.session_state.scan_count += 1


            if predicted_class == "No Hemorrhage":

                st.session_state.normal_count += 1

            else:

                st.session_state.abnormal_count += 1


            st.session_state.analysis_done = True

            st.rerun()


        except Exception as e:

            st.error(
                "Analysis could not be completed."
            )

            st.code(
                str(e)
            )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.analysis_done:

        predicted_class = (
            st.session_state.predicted_class
        )

        confidence = (
            st.session_state.confidence
        )


        if predicted_class == "No Hemorrhage":

            result_class = "result-normal"

            result_icon = "✓"

        else:

            result_class = "result-warning"

            result_icon = "⚠️"


        # IMPORTANT:
        # This HTML block is completely self-contained.
        # No Streamlit widgets are placed inside it.

        st.markdown(
            f"""
<div class="{result_class}">

<div class="result-small">
SCREENING RESULT
</div>

<div class="result-name">
{result_icon} {predicted_class}
</div>

<div class="result-confidence">
{confidence * 100:.2f}%
</div>

<div class="result-small">
CONFIDENCE
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # Quick information

        q1, q2 = st.columns(2)


        with q1:

            st.metric(
                "Classification",
                predicted_class
            )


        with q2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


# ============================================================
# COLUMN 3 — EXPLAINABLE AI
# ============================================================

with col_xai:

    st.markdown(
        """
<div class="panel-heading">
🔬 Explainable AI
</div>
""",
        unsafe_allow_html=True
    )


    if st.session_state.analysis_done:

        if st.session_state.raw_cam_mask is not None and st.session_state.raw_rgb_img is not None:

            c1, c2 = st.columns(2)

            with c1:
                opacity = st.slider(
                    "Heatmap Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.cam_opacity),
                    step=0.05,
                    key="cam_op_slider"
                )
                st.session_state.cam_opacity = opacity

            with c2:
                colormap_options = ["JET", "VIRIDIS", "INFERNO", "MAGMA", "HOT"]
                cur_cm_idx = colormap_options.index(st.session_state.cam_colormap) if st.session_state.cam_colormap in colormap_options else 0
                colormap = st.selectbox(
                    "Colormap",
                    colormap_options,
                    index=cur_cm_idx,
                    key="cam_cm_select"
                )
                st.session_state.cam_colormap = colormap

            threshold = st.slider(
                "Focus Threshold",
                min_value=0.0,
                max_value=0.8,
                value=float(st.session_state.cam_threshold),
                step=0.05,
                key="cam_th_slider",
                help="Filters out diffuse background activations to highlight the focal bleed."
            )
            st.session_state.cam_threshold = threshold

            blended_cam = blend_cam(
                st.session_state.raw_rgb_img,
                st.session_state.raw_cam_mask,
                opacity,
                colormap,
                threshold
            )
            st.session_state.gradcam_image = blended_cam

            st.image(
                blended_cam,
                use_container_width=True
            )

            st.caption(
                f"Grad-CAM Attention Map ({colormap} | {int(opacity * 100)}% Opacity | Thr: {threshold:.2f})"
            )

        elif st.session_state.gradcam_image is not None:

            st.image(
                st.session_state.gradcam_image,
                use_container_width=True
            )

            st.caption(
                "Grad-CAM visualization of attention regions"
            )

        else:

            st.info(
                "Explainability visualization is not available."
            )


        st.markdown(
            """
<div class="info-box">

<b>How to interpret the visualization:</b>
<br><br>

The heatmap highlights regions of the CT
scan that received stronger attention during
the classification process.

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.info(
            "Run CT analysis to view the explainability output."
        )


# ============================================================
# CLASSIFICATION PROBABILITIES
# ============================================================

if st.session_state.analysis_done:

    st.markdown(
        """
<div class="probability-title">
📊 Classification Probabilities
</div>
""",
        unsafe_allow_html=True
    )


    probabilities = (
        st.session_state.probabilities
    )


    probability_columns = st.columns(2)


    for i, (
        class_name,
        probability
    ) in enumerate(
        zip(
            CLASS_NAMES,
            probabilities
        )
    ):

        with probability_columns[i % 2]:

            percentage = (
                float(probability) * 100
            )


            st.markdown(
                f"""
<div>

<span class="probability-name">
{class_name}
</span>

<span class="probability-value">
{percentage:.2f}%
</span>

</div>
""",
                unsafe_allow_html=True
            )


            st.progress(
                float(probability)
            )


# ============================================================
# ANALYSIS SUMMARY
# ============================================================

if st.session_state.analysis_done:

    predicted_class = (
        st.session_state.predicted_class
    )

    confidence = (
        st.session_state.confidence
    )


    if predicted_class == "No Hemorrhage":

        summary_text = (
            f"The uploaded CT scan was classified as "
            f"<b>No Hemorrhage</b> with an estimated "
            f"confidence of "
            f"<b>{confidence * 100:.2f}%</b>."
        )

    else:

        summary_text = (
            f"The uploaded CT scan was classified as "
            f"<b>{predicted_class}</b> with an estimated "
            f"confidence of "
            f"<b>{confidence * 100:.2f}%</b>."
        )


    st.markdown(
        """
<div class="section-title">
📝 Analysis Summary
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
<div class="summary-box">

{summary_text}

<br><br>

Classification probabilities are provided
for further review.

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def generate_pdf_report(
    pil_image,
    predicted_class,
    confidence,
    probabilities
):

    from reportlab.lib.pagesizes import A4

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as ReportImage
    )

    from reportlab.lib.styles import (
        getSampleStyleSheet
    )

    from reportlab.lib import colors

    from reportlab.lib.units import inch


    buffer = io.BytesIO()


    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    heading_style = styles["Heading2"]

    body_style = styles["BodyText"]


    story = []


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI-ASSISTED CT SCAN ANALYSIS REPORT",
            title_style
        )
    )

    story.append(
        Spacer(1, 8)
    )


    story.append(
        Paragraph(
            "Intracranial Hemorrhage Screening",
            body_style
        )
    )


    story.append(
        Spacer(1, 10)
    )


    story.append(
        Paragraph(
            "Analysis Date: "
            + datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            body_style
        )
    )


    story.append(
        Spacer(1, 18)
    )


    # --------------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Input CT Scan",
            heading_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    image_buffer = io.BytesIO()

    pil_image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)


    report_image = ReportImage(
        image_buffer,
        width=3.5 * inch,
        height=3.5 * inch
    )


    story.append(
        report_image
    )


    story.append(
        Spacer(1, 18)
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Screening Result",
            heading_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    story.append(
        Paragraph(
            f"<b>Predicted Classification:</b> "
            f"{predicted_class}",
            body_style
        )
    )


    story.append(
        Paragraph(
            f"<b>Confidence:</b> "
            f"{confidence * 100:.2f}%",
            body_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Classification Probabilities",
            heading_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    table_data = [
        [
            "Classification",
            "Probability"
        ]
    ]


    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        table_data.append(
            [
                class_name,
                f"{float(probability) * 100:.2f}%"
            ]
        )


    probability_table = Table(
        table_data,
        colWidths=[
            3.8 * inch,
            1.4 * inch
        ]
    )


    probability_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#234A70")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "RIGHT"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )


    story.append(
        probability_table
    )


    story.append(
        Spacer(1, 18)
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Analysis Summary",
            heading_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    if predicted_class == "No Hemorrhage":

        report_summary = (
            "The uploaded CT scan was classified as "
            "<b>No Hemorrhage</b> with an estimated "
            f"confidence of "
            f"<b>{confidence * 100:.2f}%</b>."
        )

    else:

        report_summary = (
            "The uploaded CT scan was classified as "
            f"<b>{predicted_class}</b> with an estimated "
            f"confidence of "
            f"<b>{confidence * 100:.2f}%</b>."
        )


    story.append(
        Paragraph(
            report_summary,
            body_style
        )
    )


    story.append(
        Spacer(1, 20)
    )


    # --------------------------------------------------------
    # NOTICE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Important Notice",
            heading_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    story.append(
        Paragraph(
            "This report is generated by an AI-assisted "
            "screening prototype. The result is not a "
            "definitive medical diagnosis and should be "
            "reviewed by an appropriately qualified "
            "medical professional.",
            body_style
        )
    )


    document.build(
        story
    )


    buffer.seek(0)

    return buffer


# ============================================================
# REPORT SECTION
# ============================================================

if st.session_state.analysis_done:

    st.markdown(
        """
<div class="report-heading">
📄 Analysis Report
</div>
""",
        unsafe_allow_html=True
    )


    report_col1, report_col2 = st.columns(
        [1, 1]
    )


    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------

    with report_col1:

        try:

            pdf_data = generate_pdf_report(
                image,
                st.session_state.predicted_class,
                st.session_state.confidence,
                st.session_state.probabilities
            )


            st.download_button(
                label="📄 Download PDF Report",

                data=pdf_data,

                file_name="ICH_Analysis_Report.pdf",

                mime="application/pdf",

                use_container_width=True
            )


        except Exception as e:

            st.error(
                "Could not generate PDF report."
            )

            st.caption(
                str(e)
            )


    # --------------------------------------------------------
    # ANALYZE ANOTHER SCAN
    # --------------------------------------------------------

    with report_col2:

        if st.button(
            "🔄 Analyze Another CT Scan",
            use_container_width=True
        ):

            st.session_state.analysis_done = False

            st.session_state.predicted_class = None

            st.session_state.confidence = 0.0

            st.session_state.probabilities = None

            st.session_state.current_file_hash = None

            st.session_state.gradcam_image = None

            st.rerun()


# ============================================================
# CLINICAL RADIOLOGY ATLAS & MODEL PERFORMANCE TRANSPARENCY
# ============================================================

st.markdown("<br><hr style='border: 1px solid rgba(110, 160, 210, 0.2);'><br>", unsafe_allow_html=True)

tab_atlas, tab_metrics = st.tabs([
    "📖 Clinical Neuroradiology Reference Atlas",
    "📊 Model Architecture & Performance Transparency"
])

with tab_atlas:

    st.markdown(
        """
<div class="panel-heading">📖 Clinical Neuroradiology Reference Atlas</div>
<div class="app-subtitle">Evidence-based radiological patterns, anatomical boundaries, and triage red flags for non-contrast head CT intracranial hemorrhage interpretation.</div>
""",
        unsafe_allow_html=True
    )

    with st.expander("🚨 Emergency Surgical Red Flags (Mass Effect & Herniation)", expanded=True):
        st.markdown(
            """
- **Midline Shift (> 5 mm)**: Lateral displacement of the septum pellucidum across the cerebral falx indicates significant intracranial hypertension requiring urgent neurosurgical decompression.
- **Uncal / Transtentorial Herniation**: Medial temporal lobe (uncus) herniates over the tentorium cerebelli, causing ipsilateral pupillary dilation (CN III compression) and contralateral hemiparesis.
- **Subfalcine Herniation**: Cingulate gyrus herniates under the falx cerebri, risking anterior cerebral artery (ACA) compression and subsequent frontal lobe ischemia.
- **Basal Cistern Effacement**: Obliteration of the ambient, quadrigeminal, or suprasellar cisterns is a key marker of impending fatal brainstem compression.
- **Acute Hydrocephalus**: Ventricular enlargement proximal to an obstruction (frequently caused by IVH clot formation).
"""
        )

    atlas_col1, atlas_col2 = st.columns(2)

    with atlas_col1:

        with st.container():
            st.markdown(
                """
#### 1. Epidural Hemorrhage (EDH)
* **Typical Vessel**: Middle Meningeal Artery (MMA, 85%) following temporal squama fracture; rarely dural venous sinus.
* **CT Appearance**: Classic **biconvex (lenticular)** high-density extra-axial collection adjacent to inner table of skull.
* **Anatomical Boundary**: **Does NOT cross cranial sutures** (periosteum firmly bound at suture lines); *can* cross dural folds (falx/tentorium).
* **Clinical Course**: Classic "lucid interval" followed by rapid deterioration into coma and uncal herniation.
* **Surgical Triage**: 🔴 **Hyperacute Emergency** — emergent craniotomy indicated for volume >30 cm³ or thickness >15 mm.
"""
            )

        st.divider()

        with st.container():
            st.markdown(
                """
#### 2. Subdural Hemorrhage (SDH)
* **Typical Vessel**: Bridging cortical veins traversing the dural border cell layer to superior sagittal sinus.
* **CT Appearance**: **Crescentic (concave)** extra-axial hyperdense collection tracking along cerebral hemisphere convexities.
* **Anatomical Boundary**: **CROSSES cranial suture lines** freely; bounded by the falx cerebri and tentorium cerebelli.
* **Chronicity on CT**:
  * *Acute (< 3 days)*: Uniformly hyperdense (50-90 HU).
  * *Subacute (3-21 days)*: Isodense with brain parenchyma (difficult to discern without IV contrast).
  * *Chronic (> 21 days)*: Hypodense (approaching CSF density, 0-20 HU).
* **Surgical Triage**: 🔴 **Acute Emergency** — evacuation recommended for thickness >10 mm or midline shift >5 mm.
"""
            )

        st.divider()

        with st.container():
            st.markdown(
                """
#### 3. Intraparenchymal Hemorrhage (IPH / ICH)
* **Etiology**: Chronic uncontrolled hypertension (Charcot-Bouchard microaneurysms), amyloid angiopathy (elderly lobar), trauma, AVMs.
* **Classic Locations**: Putamen / Basal Ganglia (50%), Thalamus (15%), Lobar white matter (15%), Pons (10%), Cerebellum (10%).
* **CT Appearance**: Hyperdense intra-axial focus surrounded by a rim of hypodense vasogenic edema and local mass effect.
* **Clinical Monitoring**: High risk of hematoma expansion in first 24 hours; target systolic blood pressure reduction (<140 mmHg).
"""
            )

    with atlas_col2:

        with st.container():
            st.markdown(
                """
#### 4. Intraventricular Hemorrhage (IVH)
* **Etiology**: Secondary extension from deep hypertensive IPH (caudate/thalamus) or aneurysmal SAH; rarely primary IVH.
* **CT Appearance**: Fluid-blood layering in dependent occipital horns of lateral ventricles, or high-density casts filling 3rd/4th ventricles.
* **Diagnostic Complication**: Acute obstructive hydrocephalus due to blood clotting at the Aqueduct of Sylvius or Foramina of Luschka/Magendie.
* **Clinical Triage**: 🔴 **Urgent Neurosurgical Review** — placement of an External Ventricular Drain (EVD) is often life-saving.
"""
            )

        st.divider()

        with st.container():
            st.markdown(
                """
#### 5. Subarachnoid Hemorrhage (SAH)
* **Etiology**: Ruptured saccular (berry) intracranial aneurysm (85%, Circle of Willis); closed-head trauma; arteriovenous malformations.
* **CT Appearance**: High-density hyperattenuation filling cortical sulci, sylvian fissures, and basal cisterns ("star of David" sign in suprasellar cistern).
* **Classic Clinical Sign**: Sudden excruciating "thunderclap" headache ("worst headache of life"), neck stiffness, photophobia.
* **Clinical Complications**: Vasospasm (peak days 4-14) causing secondary ischemic stroke, delayed cerebral ischemia, rerupture risk.
"""
            )

        st.divider()

        with st.container():
            st.markdown(
                """
#### 6. Normal CT Head (No Hemorrhage)
* **Tissue Density**: Intact gray-white matter attenuation difference (gray matter ~35-40 HU, white matter ~25-30 HU).
* **Ventricles & Cisterns**: Symmetrical, age-appropriate lateral and 3rd ventricles; fully patent basal and perimesencephalic cisterns.
* **Sulcal Pattern**: Distinct bilateral cortical sulci without hyperdense effacement, midline shift, or focal hypoattenuation.
"""
            )


with tab_metrics:

    st.markdown(
        """
<div class="panel-heading">📊 Model Architecture, Evaluation & Transparency</div>
<div class="app-subtitle">Comprehensive performance breakdown from independent testing on 402 held-out CT slices with zero patient-level leakage.</div>
""",
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric("Test Accuracy", "79.35%", help="Overall accuracy on independent patient test split")
    with t2:
        st.metric("Model Backbone", "EfficientNetV2-S", help="Pretrained on ImageNet and fine-tuned")
    with t3:
        st.metric("Test Slices", "402", help="Strictly patient-level isolated CT images")
    with t4:
        st.metric("Patient Leakage", "0.0%", help="Patient-stratified split prevents slice-to-slice bias")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Detailed Classification Report (Independent Test Set)")

    rep_path = os.path.join(ROOT_DIR, "outputs", "plots", "classification_report.csv")
    if os.path.exists(rep_path):
        try:
            df_rep = pd.read_csv(rep_path)
            if "Unnamed: 0" in df_rep.columns:
                df_rep = df_rep.rename(columns={"Unnamed: 0": "Category"})

            for col in ["precision", "recall", "f1-score"]:
                if col in df_rep.columns:
                    df_rep[col] = df_rep[col].apply(lambda v: f"{v * 100:.2f}%" if pd.notnull(v) else "-")
            if "support" in df_rep.columns:
                df_rep["support"] = df_rep["support"].apply(lambda v: f"{int(v)}" if pd.notnull(v) else "-")

            st.dataframe(df_rep, use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Evaluation Visualizations & Diagnostic Curves")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        cm_path = os.path.join(ROOT_DIR, "outputs", "plots", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix across all 6 Hemorrhage Subtypes", use_container_width=True)

        loss_path = os.path.join(ROOT_DIR, "outputs", "plots", "loss_curve.png")
        if os.path.exists(loss_path):
            st.image(loss_path, caption="Cross-Entropy Loss Curve (Training vs. Validation)", use_container_width=True)

    with m_col2:
        dist_path = os.path.join(ROOT_DIR, "outputs", "plots", "class_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Medical Dataset Class Distribution (Severe Imbalance Addressed via Class Weights)", use_container_width=True)

        acc_path = os.path.join(ROOT_DIR, "outputs", "plots", "accuracy_curve.png")
        if os.path.exists(acc_path):
            st.image(acc_path, caption="Validation Accuracy Progression across Epochs", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
#### 🔬 Medical Deep Learning Integrity & Methodology
* **Patient-Level Stratified Splitting**: Slices belonging to the same patient were strictly restricted to either train, validation, or test sets. This prevents "data leakage" (the model memorizing skull shapes across consecutive slices), which is the single most common flaw in medical imaging benchmarks.
* **Class-Weighted Cross-Entropy**: Natural clinical populations feature a vast majority of normal scans (~80%) and rare severe cases (EDH, SAH <5%). The loss function penalizes errors on rare hemorrhage subtypes with inverse-frequency weighting `[0.17, 2.20, 5.80, 7.20, 15.0, 32.0]`.
* **Cosine Annealing Optimizer**: `AdamW` with a cosine learning rate scheduler gradually lowers the learning rate to allow delicate convergence on subtle hemorrhage margins.
"""
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
<div class="disclaimer">

<b>Important:</b>

This application is an AI-assisted screening
prototype intended for academic and research
demonstration.

Results should not be considered a definitive
medical diagnosis or a substitute for professional
clinical review.

</div>
""",
    unsafe_allow_html=True
)