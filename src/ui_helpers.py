"""Shared UI utilities: CSS injection, session-state helpers."""

from __future__ import annotations

import streamlit as st


# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
PRIMARY   = "#6C63FF"   # indigo-violet
SECONDARY = "#FF6584"   # coral-pink accent
BG_DARK   = "#0F0F1A"   # near-black page background
CARD_BG   = "#1A1A2E"   # card / panel background
CARD_BRD  = "#2D2D4E"   # card border
TEXT_MAIN = "#E8E8F0"   # primary text
TEXT_SUB  = "#9999BB"   # secondary / caption text
SUCCESS   = "#2ECC71"
DANGER    = "#E74C3C"


def inject_global_css() -> None:
    """Inject project-wide CSS once per page."""
    st.markdown(
        f"""
        <style>
        /* ── Page background ──────────────────────────── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {BG_DARK};
            color: {TEXT_MAIN};
        }}
        [data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid {CARD_BRD};
        }}
        /* ── Remove default top padding ───────────────── */
        [data-testid="stAppViewContainer"] > .main > div {{
            padding-top: 1rem;
        }}
        /* ── Hide Streamlit default nav header ─────────  */
        [data-testid="stHeader"] {{ display: none; }}
        header[data-testid="stHeader"] {{ display: none; }}
        /* ── Buttons ───────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, {PRIMARY}, #8B5CF6);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            transition: opacity 0.2s;
        }}
        .stButton > button:hover {{ opacity: 0.85; }}
        /* ── Text inputs ───────────────────────────────── */
        [data-testid="stTextInput"] input {{
            background-color: #0D0D1F;
            border: 1px solid {CARD_BRD};
            color: {TEXT_MAIN};
            border-radius: 8px;
        }}
        /* ── Metric cards ──────────────────────────────── */
        [data-testid="metric-container"] {{
            background-color: {CARD_BG};
            border: 1px solid {CARD_BRD};
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }}
        /* ── Divider ───────────────────────────────────── */
        hr {{ border-color: {CARD_BRD}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Reusable card component
# --------------------------------------------------------------------------- #
def card(title: str, body: str, icon: str = "") -> None:
    """Render a simple styled card block."""
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG};
            border:1px solid {CARD_BRD};
            border-radius:14px;
            padding:1.2rem 1.4rem;
            margin-bottom:0.8rem;
        ">
            <p style="font-size:1.05rem;font-weight:700;color:{TEXT_MAIN};margin:0 0 0.4rem">
                {icon} {title}
            </p>
            <p style="color:{TEXT_SUB};margin:0;font-size:0.9rem;line-height:1.5">
                {body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Full-width hero banner."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,{BG_DARK} 0%,#16213E 60%,#0F3460 100%);
            border: 1px solid {CARD_BRD};
            border-radius: 18px;
            padding: 3rem 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
        ">
            <h1 style="
                font-size: 2.6rem;
                font-weight: 800;
                background: linear-gradient(90deg,{PRIMARY},{SECONDARY});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 0.6rem;
            ">{title}</h1>
            <p style="color:{TEXT_SUB};font-size:1.1rem;max-width:600px;
                      margin:0 auto;line-height:1.6">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_badge(label: str, score: float) -> None:
    """Coloured result badge for REAL / DEEPFAKE."""
    colour = DANGER if label == "DEEPFAKE" else SUCCESS
    emoji  = "🚨" if label == "DEEPFAKE" else "✅"
    st.markdown(
        f"""
        <div style="
            background: {colour}22;
            border: 2px solid {colour};
            border-radius: 12px;
            padding: 1rem 1.4rem;
            text-align: center;
        ">
            <p style="font-size:2rem;margin:0">{emoji}</p>
            <p style="font-size:1.5rem;font-weight:800;color:{colour};margin:0.2rem 0">
                {label}
            </p>
            <p style="color:{TEXT_SUB};margin:0;font-size:0.9rem">
                Deepfake score: <b style="color:{TEXT_MAIN}">{score:.4f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Session-state helpers
# --------------------------------------------------------------------------- #
def is_logged_in() -> bool:
    return bool(st.session_state.get("user"))


def get_user() -> dict | None:
    return st.session_state.get("user")


def login_user(user: dict) -> None:
    st.session_state["user"] = user


def logout_user() -> None:
    st.session_state.pop("user", None)


def require_login(redirect_hint: str = "Login") -> bool:
    """Return True if logged in. Otherwise show a prompt and return False."""
    if is_logged_in():
        return True
    st.warning(f"Please log in first. Use the **{redirect_hint}** page in the sidebar.")
    return False
