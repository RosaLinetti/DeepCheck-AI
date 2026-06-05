import os
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT = f"{API_BASE}/document/analyze"
SEARCH_ENDPOINT  = f"{API_BASE}/document/analyze/search"
INGEST_ENDPOINT  = f"{API_BASE}/document/ingest"
STATS_ENDPOINT   = f"{API_BASE}/knowledge-base/stats"
DOCS_ENDPOINT    = f"{API_BASE}/knowledge-base/documents"
DELETE_ENDPOINT  = f"{API_BASE}/document/delete"

ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {e}")
    
    # Theme injection: Apply light theme class if selected
    if st.session_state.get("theme") == "light":
        st.markdown(
            '<script>try{document.querySelector(".stApp").classList.add("light-theme");}catch(e){}</script>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="light-theme">', unsafe_allow_html=True)
    else:
        st.markdown(
            '<script>try{document.querySelector(".stApp").classList.remove("light-theme");}catch(e){}</script>',
            unsafe_allow_html=True
        )


def init_session_state():
    defaults = {
        "results": None,
        "search_results": None,
        "chunk_strategy": "sentence",
        "window_size": 30,
        "overlap": 10,
        "algorithm_mode": "semantic",
        "settings_open": False,
        "dc_css_loaded": False,
        "current_mode": "1v1",
        "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar_navigation():
    # Theme toggle
    theme_mode = st.sidebar.toggle(
        label="Light Mode" if st.session_state.theme == "dark" else "Dark Mode",
        value=st.session_state.theme == "light",
        key="theme_toggle"
    )
    
    if theme_mode:
        if st.session_state.theme != "light":
            st.session_state.theme = "light"
            st.rerun()
    else:
        if st.session_state.theme != "dark":
            st.session_state.theme = "dark"
            st.rerun()
    
    # Display active theme indicator
    if st.session_state.theme == "light":
        st.sidebar.markdown("☀️ **Light Mode Active**", help="Switch to Dark Mode")
    else:
        st.sidebar.markdown("🌙 **Dark Mode Active**", help="Switch to Light Mode")
    
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["1v1", "db", "library"],
        format_func=lambda x: {
            "1v1": "1v1 Document Comparison",
            "db": "ChromaDB Knowledge Base Scan",
            "library": "Index Library",
        }[x],
        key="nav_mode",
    )
    st.session_state.current_mode = mode
    return mode


def render_advanced_settings_sidebar():
    with st.sidebar.expander("Advanced Token Settings", expanded=False):
        strategy = st.radio(
            "Chunking Strategy",
            ["sentence", "sliding_window"],
            format_func=lambda x: "Sentence-based" if x == "sentence" else "Sliding Window",
            index=0 if st.session_state.chunk_strategy == "sentence" else 1,
            key="settings_strategy",
        )
        st.session_state.chunk_strategy = strategy

        if strategy == "sliding_window":
            st.session_state.window_size = st.slider(
                "Window Size (tokens)", 10, 200, st.session_state.window_size, 5
            )
            max_overlap = st.session_state.window_size - 1
            st.session_state.overlap = st.slider(
                "Overlap (tokens)", 0, max_overlap,
                min(st.session_state.overlap, max_overlap), 5
            )


def render_algorithm_selector():
    st.markdown("**Detection Algorithm**")

    algo = st.radio(
        "Select processing method",
        ["semantic", "traditional"],
        format_func=lambda x: {
            "semantic": "AI Semantic (SBERT + ML hybrid classifier)",
            "traditional": "Traditional (Lexical overlap)",
        }[x],
        horizontal=True,
        key="algo_selector",
    )

    st.session_state.algorithm_mode = algo
    return algo


def render_db_status_cards(stats, title="Vector Database Status"):
    if not stats:
        st.warning("Backend not reachable.")
        return

    st.markdown(f"**{title}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Indexed Documents", stats.get("total_documents", 0))
    with c2:
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    with c3:
        st.metric("Collection", stats.get("collection_name", "—"))


def render_verdict_banner(overall_sim: float, is_1v1: bool = True):
    if overall_sim >= 0.7:
        cls, title, sub = (
            "high",
            "Critical Match Detected",
            f"{round(overall_sim*100,1)}% similarity detected",
        )
    elif overall_sim >= 0.4:
        cls, title, sub = (
            "medium",
            "Highly Paraphrased Content",
            f"{round(overall_sim*100,1)}% similarity detected",
        )
    else:
        cls, title, sub = (
            "low",
            "Clear",
            f"{round(overall_sim*100,1)}% similarity detected",
        )

    st.metric("Similarity Score", f"{round(overall_sim*100,1)}%")

    st.markdown(f"""
    <div class="verdict-banner {cls}">
        <div>
            <div class="banner-title">{title}</div>
            <div class="banner-sub">{sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(overall_sim, max_sim, total_chunks):
    c1, c2, c3 = st.columns(3)

    data = [
        ("Overall Similarity", f"{round(overall_sim*100,1)}%", c1),
        ("Max Similarity", f"{round(max_sim*100,1)}%", c2),
        ("Total Segments", str(total_chunks), c3),
    ]

    for label, value, col in data:
        with col:
            st.markdown(f"**{label}**")
            st.metric("", value)


def render_empty_state(icon, title, body=""):
    st.markdown(f"""
    <div class="empty-state">
        <h3>{title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def render_file_chip(file, label):
    size_kb = round(len(file.getvalue()) / 1024, 1)
    st.markdown(f"{file.name} · {size_kb} KB · {label}")


def severity_color(value):
    if value >= 0.7:
        return "#f43f5e"
    if value >= 0.4:
        return "#f59e0b"
    return "#34d399"


def verdict_badge_html(verdict):
    return f"{verdict}"


def highlight_matching_segments(text, reference, confidence):
    """
    Renders a pro-grade AI checker heatmap with dynamic opacity based on confidence.
    confidence < 0.4: returns text unstyled
    0.4 <= confidence < 0.7: amber/yellow tint with left border
    confidence >= 0.7: rose/red tint with left border
    """
    if confidence < 0.4:
        return text
    
    # Determine color and opacity based on confidence level
    if confidence >= 0.7:
        # Plagiarised (high match) - Rose/Red
        color_rgb = "244, 63, 94"  # rgba(244, 63, 94, ...)
        border_color = "#f43f5e"
    else:
        # Suspicious (medium match) - Amber/Yellow
        color_rgb = "234, 179, 8"  # rgba(234, 179, 8, ...)
        border_color = "#eab308"
    
    # Scale opacity dynamically: confidence 0.4-0.7 maps to 0.3-0.6, confidence 0.7+ maps to 0.6-0.9
    if confidence >= 0.7:
        alpha = min(0.9, 0.6 + (confidence - 0.7) * 0.5)
    else:
        alpha = 0.3 + (confidence - 0.4) * (0.6 - 0.3) / 0.3
    
    # Create a subtle left border highlight with background tint
    highlighted_html = f'<span style="background-color: rgba({color_rgb}, {alpha:.2f}); border-left: 4px solid {border_color}; padding-left: 4px; display: inline;">{text}</span>'
    return highlighted_html


# ---------------- API CALLS ----------------

def call_unified_api(source_text, suspicious_text, source_file, suspicious_file):
    data = {
        "chunk_strategy": st.session_state.chunk_strategy,
        "algorithm": st.session_state.algorithm_mode,
    }

    files = {}

    if source_file:
        files["source_file"] = (source_file.name, source_file.getvalue(), source_file.type)
    else:
        data["source_text"] = source_text

    if suspicious_file:
        files["suspicious_file"] = (suspicious_file.name, suspicious_file.getvalue(), suspicious_file.type)
    else:
        data["suspicious_text"] = suspicious_text

    resp = requests.post(UNIFIED_ENDPOINT, data=data, files=files or None, timeout=120)

    if not resp.ok:
        raise Exception(resp.text)

    return resp.json()


def call_chroma_search_api(file, top_k=5):
    data = {
        "chunk_strategy": st.session_state.chunk_strategy,
        "top_k": top_k,
        "algorithm": st.session_state.algorithm_mode,
    }

    files = {
        "file": (file.name, file.getvalue(), file.type)
    }

    resp = requests.post(SEARCH_ENDPOINT, data=data, files=files, timeout=120)

    if not resp.ok:
        raise Exception(resp.text)

    return resp.json()


def call_chroma_ingest_api(file):
    data = {"chunk_strategy": st.session_state.chunk_strategy}

    files = {
        "file": (file.name, file.getvalue(), file.type)
    }

    resp = requests.post(INGEST_ENDPOINT, data=data, files=files, timeout=120)

    if not resp.ok:
        raise Exception(resp.text)

    return resp.json()


def call_delete_api(filename):
    resp = requests.delete(DELETE_ENDPOINT, params={"filename": filename}, timeout=30)
    if not resp.ok:
        raise Exception(resp.text)
    return resp.json()


def get_knowledge_base_stats():
    try:
        r = requests.get(STATS_ENDPOINT, timeout=5)
        return r.json() if r.ok else {}
    except:
        return {}


def get_indexed_documents():
    try:
        r = requests.get(DOCS_ENDPOINT, timeout=5)
        return r.json().get("documents", []) if r.ok else []
    except:
        return []
