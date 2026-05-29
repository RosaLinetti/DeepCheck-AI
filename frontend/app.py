"""DeepCheck AI Streamlit frontend entrypoint."""

import streamlit as st

from components.page_compare import render_compare_page
from components.page_library import render_library_page
from components.page_scan import render_scan_page
from components.shared import init_session_state, load_css, render_settings_modal

st.set_page_config(
    page_title="DeepCheck AI",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session_state()

st.markdown('<div class="dc-navbar-anchor"></div>', unsafe_allow_html=True)
with st.container():
    nav = st.radio(
        "Navigation",
        options=["compare", "scan", "library"],
        format_func=lambda x: {
            "compare": "⇄ 1v1 Compare",
            "scan": "⌕ DB Scan",
            "library": "⊞ Index Library",
        }[x],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_section",
    )

st.markdown(
    """
    <div class="dc-page-header">
        <div class="dc-brand-name">DeepCheck AI</div>
        <div class="dc-brand-sub">SBERT Semantic Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if nav == "compare":
    render_compare_page()
elif nav == "scan":
    render_scan_page()
else:
    render_library_page()

render_settings_modal()

# Inject CSS last so our rules appear after Streamlit's generated styles
load_css()
