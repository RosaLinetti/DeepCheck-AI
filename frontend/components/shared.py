import os
import streamlit as st
import requests

API_BASE           = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT   = f"{API_BASE}/document/analyze"
SEARCH_ENDPOINT    = f"{API_BASE}/document/analyze/search"
INGEST_ENDPOINT    = f"{API_BASE}/document/ingest"
STATS_ENDPOINT     = f"{API_BASE}/knowledge-base/stats"
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    try:
        with open(css_path) as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {e}")


def init_session_state():
    defaults = {
        "results":        None,
        "search_results": None,
        "chunk_strategy": "sentence",
        "window_size":    30,
        "overlap":        10,
        "settings_open":  False,
        "dc_css_loaded":  False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_settings_modal():
    """Render a fixed settings FAB and a floating settings panel."""

    st.markdown('<div class="settings-fab-anchor"></div>', unsafe_allow_html=True)
    if st.button("⚙  Advanced token settings", key="fab_settings_toggle"):
        st.session_state.settings_open = not st.session_state.settings_open
        st.rerun()

    if not st.session_state.settings_open:
        return

    st.markdown('<div class="settings-panel-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="settings-panel-shell">', unsafe_allow_html=True)
    with st.container():
        st.markdown(
            '<div class="settings-panel-title">Advanced token settings</div>',
            unsafe_allow_html=True,
        )
        st.caption("Applied globally to all analysis modes.")

        strategy = st.radio(
            "Chunk strategy",
            options=["sentence", "sliding_window"],
            format_func=lambda x: "Sentence" if x == "sentence" else "Sliding window",
            index=0 if st.session_state.chunk_strategy == "sentence" else 1,
            horizontal=True,
            key="settings_strategy",
        )
        st.session_state.chunk_strategy = strategy

        if strategy == "sliding_window":
            st.session_state.window_size = st.slider(
                "Window size (tokens)",
                min_value=10,
                max_value=200,
                value=st.session_state.window_size,
                step=5,
                key="settings_window",
            )
            max_overlap = st.session_state.window_size - 1
            st.session_state.overlap = st.slider(
                "Overlap (tokens)",
                min_value=0,
                max_value=max_overlap,
                value=min(st.session_state.overlap, max_overlap),
                step=5,
                key="settings_overlap",
            )
            if st.session_state.overlap >= st.session_state.window_size:
                st.error("Overlap must be less than window size.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def call_unified_api(source_text, suspicious_text, source_file, suspicious_file) -> dict:
    data = {"chunk_strategy": st.session_state.chunk_strategy}
    if st.session_state.chunk_strategy == "sliding_window":
        data["window_size"] = st.session_state.window_size
        data["overlap"]     = st.session_state.overlap

    files = {}
    if source_file is not None:
        files["source_file"] = (source_file.name, source_file.getvalue(),
                                source_file.type or "application/octet-stream")
    elif source_text:
        data["source_text"] = source_text

    if suspicious_file is not None:
        files["suspicious_file"] = (suspicious_file.name, suspicious_file.getvalue(),
                                    suspicious_file.type or "application/octet-stream")
    elif suspicious_text:
        data["suspicious_text"] = suspicious_text

    resp = requests.post(UNIFIED_ENDPOINT, data=data, files=files if files else None, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Analyze failed ({resp.status_code}): {resp.text}")
    return resp.json()


def call_chroma_search_api(suspicious_file, top_k: int = 5) -> dict:
    data  = {"chunk_strategy": st.session_state.chunk_strategy, "top_k": top_k}
    files = {"file": (suspicious_file.name, suspicious_file.getvalue(),
                      suspicious_file.type or "application/octet-stream")}
    resp = requests.post(SEARCH_ENDPOINT, data=data, files=files, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Scan failed ({resp.status_code}): {resp.text}")
    return resp.json()


def call_chroma_ingest_api(file) -> dict:
    data  = {"chunk_strategy": st.session_state.chunk_strategy}
    files = {"file": (file.name, file.getvalue(),
                      file.type or "application/octet-stream")}
    resp = requests.post(INGEST_ENDPOINT, data=data, files=files, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Ingest failed ({resp.status_code}): {resp.text}")
    return resp.json()


def get_knowledge_base_stats() -> dict:
    try:
        resp = requests.get(STATS_ENDPOINT, timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def severity_color(value: float) -> str:
    if value >= 0.7:   return "#f43f5e"
    elif value >= 0.4: return "#f59e0b"
    return "#34d399"


def verdict_badge_html(verdict: str) -> str:
    css = f"verdict-{verdict.lower().replace(' ', '-')}"
    return f'<span class="verdict-badge {css}">{verdict}</span>'


def render_verdict_banner(overall_sim: float):
    if overall_sim >= 0.7:
        cls, icon, title, sub = "high", "🚨", "High plagiarism risk", \
            f"Overall similarity {round(overall_sim*100,1)}% — significant overlap detected."
    elif overall_sim >= 0.4:
        cls, icon, title, sub = "medium", "⚠️", "Suspicious content detected", \
            f"Overall similarity {round(overall_sim*100,1)}% — some segments warrant review."
    else:
        cls, icon, title, sub = "low", "✅", "Likely original", \
            f"Overall similarity {round(overall_sim*100,1)}% — no significant overlap found."
    st.markdown(f"""
    <div class="verdict-banner {cls}">
        <div class="banner-icon">{icon}</div>
        <div><div class="banner-title">{title}</div>
        <div class="banner-sub">{sub}</div></div>
    </div>""", unsafe_allow_html=True)


def render_empty_state(icon: str, title: str, body: str):
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <h3>{title}</h3>
        <p>{body}</p>
    </div>""", unsafe_allow_html=True)


def render_db_status_cards(stats: dict | None, title: str = "Vector database status"):
    if not stats:
        st.warning(
            "Cannot reach the backend — make sure the API server is running on port 8000.",
            icon="⚠️",
        )
        return

    total_chunks = stats.get("total_chunks", "—")
    collection_name = stats.get("collection_name", "—")

    st.markdown(f'<div class="panel-label">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="db-stat-grid">
            <div class="db-stat-card">
                <div class="db-stat-label">Collection</div>
                <div class="db-stat-value db-stat-mono">{collection_name}</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-label">Indexed chunks</div>
                <div class="db-stat-strong">{total_chunks}</div>
            </div>
            <div class="db-stat-card">
                <div class="db-stat-label">Status</div>
                <div class="db-stat-value db-stat-ok">Connected ✓</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(overall_sim: float, max_sim: float, total_chunks: int):
    k1, k2, k3 = st.columns(3)
    for col, label, value, color, sub in [
        (k1, "Overall similarity", f"{round(overall_sim*100,1)}%", severity_color(overall_sim), "Mean across segments"),
        (k2, "Max similarity",     f"{round(max_sim*100,1)}%",     severity_color(max_sim),     "Highest chunk match"),
        (k3, "Total segments",     str(total_chunks),               "#2dd4bf",                  "Evaluated blocks"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


def render_file_chip(file, label: str):
    size_kb = round(len(file.getvalue()) / 1024, 1)
    st.markdown(
        f'<div class="file-chip">📄 <span>{file.name}</span> · {size_kb} KB · {label}</div>',
        unsafe_allow_html=True)
