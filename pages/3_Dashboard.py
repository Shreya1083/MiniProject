"""Dashboard page — shows the logged-in user's full detection history.

Features:
  - Summary stats (total scans, deepfake %, highest score)
  - Searchable, sortable history table
  - Delete individual entries or clear all
  - CSV download of full history
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

import streamlit as st

from src.database import (
    clear_user_history,
    delete_history_entry,
    get_user_history,
    init_db,
)
from src.ui_helpers import (
    CARD_BG, CARD_BRD, DANGER, PRIMARY, SECONDARY, SUCCESS,
    TEXT_MAIN, TEXT_SUB,
    inject_global_css, is_logged_in, require_login,
)

init_db()

st.set_page_config(
    page_title="DeepGuard — Dashboard",
    page_icon="📊",
    layout="wide",
)
inject_global_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Auth guard ────────────────────────────────────────────────────────────────
if not require_login("Login"):
    st.stop()

user    = st.session_state["user"]
history = get_user_history(user["id"])

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{TEXT_MAIN}'>📊 Your Dashboard</h1>"
    f"<p style='color:{TEXT_SUB}'>All previous detections for "
    f"<b style='color:{PRIMARY}'>{user['username']}</b> · "
    f"Account created: {user['created_at'][:10]}</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if not history:
    st.markdown(
        f"""
        <div style="
            background:{CARD_BG};border:2px dashed {CARD_BRD};
            border-radius:16px;padding:3rem;text-align:center;
        ">
            <p style="font-size:3rem;margin:0">🗂️</p>
            <p style="color:{TEXT_MAIN};font-size:1.1rem;
                      font-weight:600;margin:0.5rem 0 0.3rem">No detections yet</p>
            <p style="color:{TEXT_SUB};font-size:0.88rem;margin:0">
                Go to the <b>Detect</b> page, upload an image, and your results
                will appear here automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🔍 Go to Detect", use_container_width=False):
        st.switch_page("pages/2_Detect.py")
    st.stop()

# ── Summary stats ─────────────────────────────────────────────────────────────
total      = len(history)
n_deepfake = sum(1 for h in history if h["label"] == "DEEPFAKE")
n_real     = total - n_deepfake
max_score  = max(h["deepfake_score"] for h in history)
avg_score  = sum(h["deepfake_score"] for h in history) / total

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Scans",     total)
s2.metric("Deepfakes Found", n_deepfake, delta=None)
s3.metric("Real Images",     n_real)
s4.metric("Highest Score",   f"{max_score:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Controls row ──────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

with ctrl1:
    search_term = st.text_input(
        "🔎 Filter by filename or label",
        placeholder="e.g.  face.jpg  or  DEEPFAKE",
    )

with ctrl2:
    sort_by = st.selectbox(
        "Sort by",
        ["Newest first", "Oldest first", "Highest score", "Lowest score"],
    )

with ctrl3:
    # CSV download
    csv_buf = io.StringIO()
    writer  = csv.DictWriter(
        csv_buf,
        fieldnames=["filename", "label", "deepfake_score", "timestamp", "architecture"],
    )
    writer.writeheader()
    for h in history:
        writer.writerow({
            "filename":       h["filename"],
            "label":          h["label"],
            "deepfake_score": h["deepfake_score"],
            "timestamp":      h["timestamp"],
            "architecture":   h["architecture"],
        })
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Export CSV",
        data=csv_buf.getvalue(),
        file_name=f"deepguard_history_{user['username']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ── Filter + sort ─────────────────────────────────────────────────────────────
filtered = history
if search_term:
    q = search_term.lower()
    filtered = [
        h for h in filtered
        if q in h["filename"].lower() or q in h["label"].lower()
    ]

if sort_by == "Oldest first":
    filtered = list(reversed(filtered))
elif sort_by == "Highest score":
    filtered = sorted(filtered, key=lambda h: h["deepfake_score"], reverse=True)
elif sort_by == "Lowest score":
    filtered = sorted(filtered, key=lambda h: h["deepfake_score"])

st.markdown(
    f"<p style='color:{TEXT_SUB};font-size:0.85rem'>"
    f"Showing <b style='color:{TEXT_MAIN}'>{len(filtered)}</b> of {total} results</p>",
    unsafe_allow_html=True,
)

# ── History cards ─────────────────────────────────────────────────────────────
to_delete = None

for h in filtered:
    label_colour = DANGER if h["label"] == "DEEPFAKE" else SUCCESS
    label_emoji  = "🚨" if h["label"] == "DEEPFAKE" else "✅"
    score_bar_w  = int(h["deepfake_score"] * 100)
    ts_display   = h["timestamp"][:19].replace("T", "  ")

    with st.container():
        st.markdown(
            f"""
            <div style="
                background:{CARD_BG};
                border:1px solid {CARD_BRD};
                border-left: 4px solid {label_colour};
                border-radius:12px;
                padding:1rem 1.2rem;
                margin-bottom:0.7rem;
            ">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <p style="font-weight:700;color:{TEXT_MAIN};margin:0 0 0.2rem;font-size:1rem">
                            📄 {h['filename']}
                        </p>
                        <p style="color:{TEXT_SUB};font-size:0.8rem;margin:0">
                            🕐 {ts_display}
                        </p>
                    </div>
                    <div style="text-align:right">
                        <span style="
                            background:{label_colour}22;
                            color:{label_colour};
                            border:1px solid {label_colour};
                            border-radius:20px;
                            padding:0.2rem 0.8rem;
                            font-size:0.85rem;font-weight:700
                        ">{label_emoji} {h['label']}</span>
                    </div>
                </div>
                <div style="margin-top:0.7rem">
                    <p style="color:{TEXT_SUB};font-size:0.8rem;margin:0 0 0.25rem">
                        Deepfake score: <b style="color:{TEXT_MAIN}">{h['deepfake_score']:.4f}</b>
                    </p>
                    <div style="background:{CARD_BRD};border-radius:4px;height:6px;width:100%">
                        <div style="
                            background:{label_colour};
                            width:{score_bar_w}%;
                            height:6px;border-radius:4px;
                        "></div>
                    </div>
                </div>
                <p style="color:{TEXT_SUB};font-size:0.75rem;margin:0.5rem 0 0">
                    Model: {h['architecture']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expand for full report
        with st.expander("View full report"):
            st.markdown(f"**File:** `{h['filename']}`")
            st.markdown(f"**Label:** `{h['label']}`")
            st.markdown(f"**Deepfake score:** `{h['deepfake_score']:.6f}`")
            st.markdown(f"**Real score:** `{1-h['deepfake_score']:.6f}`")
            st.markdown(f"**Timestamp:** `{h['timestamp']}`")
            st.markdown(f"**Architecture:** {h['architecture']}")
            st.markdown(f"**Preprocessing:** {h['preprocessing']}")
            st.markdown(f"**Score note:** _{h['score_note']}_")

            if st.button(f"🗑️ Delete this entry", key=f"del_{h['id']}"):
                to_delete = h["id"]

# Handle delete outside the loop to avoid rerun conflicts
if to_delete:
    delete_history_entry(to_delete, user["id"])
    st.rerun()

# ── Danger zone: clear all ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    f"<p style='color:{DANGER};font-weight:700'>⚠️ Danger Zone</p>",
    unsafe_allow_html=True,
)

confirm_clear = st.checkbox("I want to permanently delete all my scan history")
if confirm_clear:
    if st.button("🗑️ Clear all history", type="primary"):
        n = clear_user_history(user["id"])
        st.success(f"Deleted {n} entries.")
        st.rerun()
