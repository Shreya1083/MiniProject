"""Login / Sign-up page.

Login   : username + password
Sign up : email + username + password
"""

from __future__ import annotations

import streamlit as st

from src.database import authenticate, create_user, get_user_by_username, init_db
from src.ui_helpers import (
    CARD_BG, CARD_BRD, PRIMARY, SECONDARY, TEXT_MAIN, TEXT_SUB,
    inject_global_css, login_user, is_logged_in,
)

init_db()

st.set_page_config(
    page_title="DeepGuard — Login",
    page_icon="🔐",
    layout="centered",
)
inject_global_css()

# ── Already logged in ─────────────────────────────────────────────────────────
if is_logged_in():
    user = st.session_state["user"]
    st.markdown(
        f"""
        <div style="text-align:center;padding:2rem 0">
            <p style="font-size:2.5rem">👋</p>
            <h2 style="color:{TEXT_MAIN}">Welcome back, {user['username']}!</h2>
            <p style="color:{TEXT_SUB}">You are already logged in.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Go to Detect", use_container_width=True):
            st.switch_page("pages/2_Detect.py")
    with col2:
        if st.button("📊 Go to Dashboard", use_container_width=True):
            st.switch_page("pages/3_Dashboard.py")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="text-align:center;padding:1.5rem 0 1rem">
        <span style="font-size:3rem">🎭</span>
        <h1 style="
            background:linear-gradient(90deg,{PRIMARY},{SECONDARY});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-size:2rem;margin:0.3rem 0 0.2rem;font-weight:800
        ">DeepGuard</h1>
        <p style="color:{TEXT_SUB};font-size:0.9rem;margin:0">
            Sign in or create a new account
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs: Login / Sign Up ─────────────────────────────────────────────────────
tab_login, tab_signup = st.tabs(["🔐  Login", "✨  Sign Up"])

# ── LOGIN TAB ─────────────────────────────────────────────────────────────────
with tab_login:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("login_form"):
        st.markdown(
            f"<p style='color:{TEXT_MAIN};font-weight:600;margin-bottom:0.5rem'>"
            "Enter your credentials</p>",
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please fill in both fields.")
        else:
            user = authenticate(username, password)
            if user:
                login_user(user)
                st.success(f"Welcome back, **{user['username']}**! 🎉")
                st.balloons()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

# ── SIGN UP TAB ───────────────────────────────────────────────────────────────
with tab_signup:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("signup_form"):
        st.markdown(
            f"<p style='color:{TEXT_MAIN};font-weight:600;margin-bottom:0.5rem'>"
            "Create your free account</p>",
            unsafe_allow_html=True,
        )
        su_email    = st.text_input("Email",    placeholder="you@example.com",   key="su_email")
        su_username = st.text_input("Username", placeholder="choose_a_username", key="su_username")
        su_password = st.text_input("Password", type="password",
                                    placeholder="min 6 characters",             key="su_pass")
        su_confirm  = st.text_input("Confirm Password", type="password",
                                    placeholder="repeat password",               key="su_confirm")
        su_submit   = st.form_submit_button("Create Account", use_container_width=True)

    if su_submit:
        # Validation
        errors = []
        if not su_email or "@" not in su_email:
            errors.append("Enter a valid email address.")
        if not su_username or len(su_username) < 3:
            errors.append("Username must be at least 3 characters.")
        if not su_password or len(su_password) < 6:
            errors.append("Password must be at least 6 characters.")
        if su_password != su_confirm:
            errors.append("Passwords do not match.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # Check if username already exists
            if get_user_by_username(su_username):
                st.error("That username is already taken. Please choose another.")
            else:
                new_user = create_user(su_email, su_username, su_password)
                if new_user:
                    login_user(new_user)
                    st.success(
                        f"Account created! Welcome, **{new_user['username']}** 🎉"
                    )
                    st.balloons()
                    st.rerun()
                else:
                    st.error(
                        "That email is already registered. Try logging in instead."
                    )

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center;color:{TEXT_SUB};font-size:0.78rem'>"
    "Passwords are stored as SHA-256 hashes in a local SQLite database.<br>"
    "This app is for research use only — do not use real sensitive credentials.</p>",
    unsafe_allow_html=True,
)
