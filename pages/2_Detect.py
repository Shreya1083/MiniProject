"""Detect page — upload an image, run the deepfake detector, view the result.

Requires login.  Every successful detection is saved to the user's history.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.database import init_db, save_search
from src.detector import DetectorError, load_image_model, predict_image_file
from src.frequency import FrequencyError, analyze_image, make_spectrum_figure
from src.ui_helpers import (
    CARD_BG, CARD_BRD, PRIMARY, TEXT_MAIN, TEXT_SUB,
    inject_global_css, is_logged_in, require_login, result_badge,
)

init_db()

st.set_page_config(
    page_title="DeepGuard — Detect",
    page_icon="🔍",
    layout="wide",
)
inject_global_css()

# ── Sidebar: user info + logout ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<p style='color:{TEXT_MAIN};font-weight:700;font-size:1.05rem'>"
        "🎭 DeepGuard</p>",
        unsafe_allow_html=True,
    )
    if is_logged_in():
        user = st.session_state["user"]
        st.success(f"👤 **{user['username']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()
    st.divider()
    st.markdown(
        f"<p style='color:{TEXT_SUB};font-size:0.82rem'>"
        "Upload a JPG/PNG face image.<br>"
        "The detector runs on CPU — allow a few seconds.</p>",
        unsafe_allow_html=True,
    )

# ── Auth guard ────────────────────────────────────────────────────────────────
if not require_login("Login"):
    st.stop()

user = st.session_state["user"]

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{TEXT_MAIN}'>🔍 Deepfake Detector</h1>"
    f"<p style='color:{TEXT_SUB}'>Upload a face image to check whether it is real "
    "or AI-generated. Results are saved to your Dashboard.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def get_model():
    return load_image_model()


try:
    model = get_model()
    model_ok = True
except (DetectorError, OSError, RuntimeError) as exc:
    model_ok = False
    st.error(f"Model failed to load: {exc}")

if not model_ok:
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Best results with a cropped face image (224-380 px resolution).",
)

if uploaded is None:
    # Show a placeholder when nothing is uploaded
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG};border:2px dashed {CARD_BRD};
            border-radius:16px;padding:3rem;text-align:center;
        ">
            <p style="font-size:3rem;margin:0">📂</p>
            <p style="color:{TEXT_MAIN};font-size:1.1rem;
                      font-weight:600;margin:0.5rem 0 0.3rem">
                No image selected</p>
            <p style="color:{TEXT_SUB};font-size:0.88rem;margin:0">
                Use the uploader above to analyse an image.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
suffix   = Path(uploaded.name).suffix.lower() or ".png"
tmp_path = None

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    with st.spinner("Running AI inference on CPU…"):
        result = predict_image_file(tmp_path, model)

except DetectorError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error: {exc}")
    st.stop()
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

# ── Save to history ───────────────────────────────────────────────────────────
save_search(
    user_id        = user["id"],
    filename       = uploaded.name,
    label          = result["label"],
    deepfake_score = result["deepfake_score"],
    architecture   = result["architecture"],
    preprocessing  = result["preprocessing"],
    score_note     = result["score_note"],
)

# ── Display results ───────────────────────────────────────────────────────────
col_img, col_result = st.columns([1, 1])

with col_img:
    st.markdown(
        f"<p style='color:{TEXT_MAIN};font-weight:600'>Uploaded image</p>",
        unsafe_allow_html=True,
    )
    st.image(result["rgb_image"], use_container_width=True)

with col_result:
    st.markdown(
        f"<p style='color:{TEXT_MAIN};font-weight:600;margin-bottom:0.8rem'>"
        "Detection result</p>",
        unsafe_allow_html=True,
    )
    result_badge(result["label"], result["deepfake_score"])
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Deepfake score", f"{result['deepfake_score']:.4f}")
    with c2:
        st.metric("Real score",     f"{1-result['deepfake_score']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:{CARD_BG};border:1px solid {CARD_BRD};"
        "border-radius:10px;padding:0.8rem'>"
        f"<p style='color:{TEXT_SUB};font-size:0.8rem;margin:0'>"
        f"<b style='color:{TEXT_MAIN}'>Architecture:</b> {result['architecture']}<br>"
        f"<b style='color:{TEXT_MAIN}'>Input:</b> {result['preprocessing']}<br>"
        f"<b style='color:{TEXT_MAIN}'>Note:</b> {result['score_note']}"
        "</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("✅ This result has been saved to your Dashboard.")

# ── FFT spectrum ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<h3 style='color:{TEXT_MAIN}'>🌊 Frequency Analysis (FFT Spectrum)</h3>",
    unsafe_allow_html=True,
)

try:
    fft_data = analyze_image(result["rgb_image"])
    fig      = make_spectrum_figure(result["rgb_image"], fft_data)
    st.pyplot(fig)
    st.caption(
        "Centre = low frequencies (smooth areas). "
        "Edges = high frequencies (fine details, edges, artefacts). "
        "Deepfakes often suppress high frequencies — visible as a 'cleaner' spectrum. "
        "This view is for human inspection; the classifier does NOT use it."
    )
except FrequencyError as exc:
    st.warning(f"Could not generate FFT spectrum: {exc}")
