"""DeepCheck AI — Streamlit frontend entrypoint."""
import streamlit as st
from components.page_compare import render_compare_page
from components.page_library import render_library_page
from components.page_scan import render_scan_page
from components.shared import (
    init_session_state,
    load_css,
    render_sidebar_navigation,
    render_advanced_settings_sidebar,
)

st.set_page_config(
    page_title="DeepCheck AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="padding-bottom:16px;font-size:1.1rem;font-weight:700"> DeepCheck AI</div>',
    unsafe_allow_html=True,
)
mode = render_sidebar_navigation()
render_advanced_settings_sidebar()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="dc-page-header">
        <div class="dc-brand-name">DeepCheck AI</div>
        <div class="dc-brand-sub">SBERT Semantic Detection &amp; Hybrid Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Route ─────────────────────────────────────────────────────────────────────
if mode == "1v1":
    render_compare_page()
elif mode == "db":
    render_scan_page()
else:
    render_library_page()

# Inject CSS after Streamlit's generated styles so our rules take precedence
load_css()
