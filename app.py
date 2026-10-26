"""Landing / Home page  ──  Deepfake Detection Platform.

Streamlit multi-page apps treat `app.py` (the entry point) as Page 1.
Additional pages live in the  `pages/`  folder and appear in the sidebar
automatically.

Flow:
  Home  →  Login / Sign Up  →  Detect  (requires login)
                              →  Dashboard  (requires login)
"""

from __future__ import annotations

import streamlit as st

from src.database import init_db
from src.ui_helpers import (
    CARD_BG,
    CARD_BRD,
    PRIMARY,
    SECONDARY,
    TEXT_MAIN,
    TEXT_SUB,
    card,
    hero,
    inject_global_css,
    is_logged_in,
)

# ── Bootstrap ───────────────────────────────────────────────────────────────
init_db()   # create tables if first run

st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

# ── Sidebar navigation hint ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.2rem 0 0.5rem">
            <span style="font-size:2.2rem">🎭</span>
            <p style="font-size:1.15rem;font-weight:700;
                      color:{TEXT_MAIN};margin:0.3rem 0 0">DeepGuard</p>
            <p style="font-size:0.78rem;color:{TEXT_SUB};margin:0">
                AI-powered deepfake detection
            </p>
        </div>
        <hr style="border-color:{CARD_BRD};margin:0.8rem 0">
        """,
        unsafe_allow_html=True,
    )
    if is_logged_in():
        user = st.session_state["user"]
        st.success(f"Logged in as **{user['username']}**")
    else:
        st.info("Not logged in — use **Login** to access the detector.")

# ── Hero banner ──────────────────────────────────────────────────────────────
hero(
    "🎭 DeepGuard",
    "AI-powered deepfake detection using EfficientNetV2-S and frequency-domain "
    "analysis.  Upload any image and know instantly whether it is real or "
    "synthetically manipulated.",
)

# ── Feature cards row ────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    card(
        "🤖 AI Detection",
        "EfficientNetV2-S backbone trained on deepfake datasets. "
        "Sigmoid output gives a continuous manipulation score (0 = real, 1 = fake).",
        "",
    )

with c2:
    card(
        "🌊 Frequency Analysis",
        "2D FFT magnitude-spectrum view reveals high-frequency artefacts "
        "that deepfake generators often suppress.  Shown alongside every result.",
        "",
    )

with c3:
    card(
        "📊 Personal Dashboard",
        "Every scan is saved to your account.  Review past predictions, "
        "compare scores, and download a history report — all in one place.",
        "",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ─────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:{TEXT_MAIN};margin-bottom:0.2rem'>How it works</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{TEXT_SUB};margin-bottom:1.2rem'>"
    "Four simple steps from upload to insight.</p>",
    unsafe_allow_html=True,
)

steps = st.columns(4)
step_data = [
    ("1️⃣", "Sign Up / Log In", "Create a free account or log in to access the detector and save your history."),
    ("2️⃣", "Upload Image",    "Go to the Detect page and upload any JPG or PNG face image from your device."),
    ("3️⃣", "AI Analysis",     "EfficientNetV2-S runs on-device (CPU). You get a REAL / DEEPFAKE label and score."),
    ("4️⃣", "View Report",     "The FFT spectrum is generated.  The result is saved to your Dashboard instantly."),
]
for col, (icon, title, desc) in zip(steps, step_data):
    with col:
        st.markdown(
            f"""
            <div style="
                background:{CARD_BG};border:1px solid {CARD_BRD};
                border-radius:14px;padding:1.2rem;text-align:center;
                height:100%;min-height:150px
            ">
                <p style="font-size:2rem;margin:0">{icon}</p>
                <p style="font-weight:700;color:{TEXT_MAIN};margin:0.4rem 0 0.3rem;
                          font-size:0.95rem">{title}</p>
                <p style="color:{TEXT_SUB};font-size:0.82rem;
                          line-height:1.4;margin:0">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Call to action ────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background:linear-gradient(135deg,{CARD_BG},{PRIMARY}33);
        border:1px solid {PRIMARY}55;
        border-radius:16px;padding:2rem;text-align:center;
    ">
        <h3 style="color:{TEXT_MAIN};margin:0 0 0.6rem">Ready to detect?</h3>
        <p style="color:{TEXT_SUB};margin:0 0 1.2rem;font-size:0.95rem">
            Log in or create a free account to start scanning images.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

cta1, cta2, _ = st.columns([1, 1, 3])
with cta1:
    if st.button("🔐  Login", use_container_width=True):
        st.switch_page("pages/1_Login.py")
with cta2:
    if st.button("✨  Sign Up", use_container_width=True):
        st.switch_page("pages/1_Login.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# ── About + disclaimer ────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        border-top:1px solid {CARD_BRD};padding-top:1.2rem;
        color:{TEXT_SUB};font-size:0.8rem;line-height:1.6;text-align:center;
    ">
        <b style="color:{TEXT_MAIN}">DeepGuard</b> — Mini Project,
        Computer Science Engineering · 2026 &nbsp;|&nbsp;
        Model: <i>Arko007/deepfake-image-detector</i> (Apache-2.0, HuggingFace).<br>
        <b>Disclaimer:</b> This tool is for research and education.
        The model score is not a calibrated probability.
        Do not use for forensic or legal conclusions without expert review.
    </div>
    """,
    unsafe_allow_html=True,
)
