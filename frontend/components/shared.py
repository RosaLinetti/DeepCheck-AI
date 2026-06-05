import os
import streamlit as st
import requests

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT = f"{API_BASE}/document/analyze"
SEARCH_ENDPOINT  = f"{API_BASE}/document/analyze/search"
INGEST_ENDPOINT  = f"{API_BASE}/document/ingest"
STATS_ENDPOINT   = f"{API_BASE}/knowledge-base/stats"
DOCS_ENDPOINT    = f"{API_BASE}/knowledge-base/documents"
DELETE_ENDPOINT  = f"{API_BASE}/document/delete"

ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]


# ── THEME & SETUP LOGIC ───────────────────────────────────────────────────────
def init_session_state():
    """Initializes global data arrays and state variables uniformly."""
    defaults = {
        "results":          None,
        "search_results":   None,
        "chunk_strategy":   "sentence",
        "window_size":      30,
        "overlap":          10,
        "algorithm_mode":   "semantic",
        "settings_open":    False,
        "dc_css_loaded":    False,
        "current_mode":     "1v1",
        "theme":            "dark",
        "_theme_toggle":    False,   # Mirrors theme default initialization (Dark = False)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def inject_theme_css():
    """
    Injects :root token overrides AND Streamlit-specific selector fixes
    on every render. Only the light block does real work; dark is a no-op
    since :root already defaults to dark values in styles.css.
    """
    if st.session_state.get("theme") != "light":
        st.markdown("<style>/* dark — :root defaults active */</style>",
                    unsafe_allow_html=True)
        return
    light_css = """
    <style>
    /* ── 1. Root token override ─────────────────────────────────────── */
    :root {
        --bg-main:      #f1f5f9 !important;
        --bg-surface:   #ffffff !important;
        --text-main:    #0f172a !important;
        --text-muted:   #475569 !important;
        --border-color: #cbd5e1 !important;
    }
    /* ── 2. App + sidebar surfaces ──────────────────────────────────── */
    .stApp,
    .stApp > div,
    section.main,
    .block-container {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border-color) !important;
    }
    /* ── 3. Global text — every Streamlit markdown/paragraph node ───── */
    .stApp p,
    .stApp span,
    .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] li {
        color: var(--text-main) !important;
    }
    /* ── 4. Page header brand text ──────────────────────────────────── */
    .dc-brand-name { color: #2563eb !important; }
    .dc-brand-sub  { color: #475569 !important; }
    /* ── 5. Custom HTML component text (panel-label, chart-header) ──── */
    .panel-label,
    .chart-header,
    .banner-title  { color: var(--text-main) !important; }
    .banner-sub,
    .threshold-label { color: var(--text-muted) !important; }
    .chart-header  { color: var(--text-main) !important; }
    /* page-scan overrides its own panel-label to #e6eefc — flip it */
    .page-scan .panel-label,
    .page-scan .chart-header { color: var(--text-main) !important; }
    /* ── 6. Sidebar widget labels ────────────────────────────────────── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p {
        color: var(--text-main) !important;
    }
    /* Sidebar caption is intentionally slightly muted */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
    }
    /* Sidebar toggle label */
    [data-testid="stSidebar"] [data-testid="stToggle"] label,
    [data-testid="stSidebar"] [data-testid="stToggle"] p {
        color: var(--text-main) !important;
    }
    /* Radio option labels inside sidebar */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #334155 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        color: #2563eb !important;
        background: rgba(37, 99, 235, 0.08) !important;
    }
    /* ── 7. Main area widget labels ──────────────────────────────────── */
    label,
    label span,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stSlider label,
    .stSlider label p,
    .stRadio label,
    .stRadio label span,
    .stMultiSelect label,
    .stSelectbox label,
    .stTextInput label,
    .stTextArea label,
    .stNumberInput label {
        color: var(--text-main) !important;
    }
    /* st.caption */
    [data-testid="stCaptionContainer"] p { color: var(--text-muted) !important; }
    /* ── 8. Metric widget ─────────────────────────────────────────────── */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricValue"]    { color: var(--text-main) !important; }
    [data-testid="stMetricDelta"]    { color: var(--text-muted) !important; }
    /* ── 9. Inputs / textareas / selects ────────────────────────────── */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: var(--text-main) !important;
        border-color: var(--border-color) !important;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: var(--border-color) !important;
        color: var(--text-main) !important;
    }
    /* Dropdown menu that renders in a portal outside .stApp */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="option"] {
        background: #ffffff !important;
        color: var(--text-main) !important;
    }
    /* ── 10. File uploader — full light-mode restyle ────────────────── */
    div[data-testid="stFileUploader"] {
        background: #f8fafc !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px dashed #93c5fd !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        background: #eff6ff !important;
        border-color: #3b82f6 !important;
    }
    /* The ::before / ::after pseudo-elements carry the upload copy */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]::before {
        color: #1e40af !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]::after {
        color: #475569 !important;
    }
    /* Any surviving native Streamlit upload instruction text */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }
    /* ── 11. Expander ───────────────────────────────────────────────── */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    [data-testid="stExpanderToggleIcon"] { color: var(--text-main) !important; }
    [data-testid="stExpander"] { border-color: var(--border-color) !important; }
    /* ── 12. Slider track & thumb ───────────────────────────────────── */
    [data-baseweb="slider"] [data-testid="stSliderTrack"] {
        background-color: var(--border-color) !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        background: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"] { color: var(--text-muted) !important; }
    /* ── 13. st.info / st.success / st.warning / st.error boxes ──────── */
    [data-testid="stAlert"] { color: var(--text-main) !important; }
    [data-testid="stAlert"] p { color: var(--text-main) !important; }
    /* ── 14. Multiselect pills ──────────────────────────────────────── */
    [data-baseweb="tag"] {
        background: #dbeafe !important;
        color: #1e40af !important;
    }
    [data-baseweb="tag"] span { color: #1e40af !important; }
    /* ── 15. Chunk toggle buttons (cmp + scan) ──────────────────────── */
    div[data-testid="stElementContainer"][class*="st-key-cmp_toggle_"] .stButton > button,
    div[data-testid="stElementContainer"][class*="st-key-scan_toggle_"] .stButton > button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid var(--border-color) !important;
    }
    div[data-testid="stElementContainer"][class*="st-key-cmp_toggle_"] .stButton > button:hover,
    div[data-testid="stElementContainer"][class*="st-key-scan_toggle_"] .stButton > button:hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    /* ── 16. db-stat-mono and hardcoded-color utility classes ───────── */
    .db-stat-mono  { color: #3b82f6 !important; }
    .empty-state   { color: #475569 !important; }
    .empty-state h3 { color: #334155 !important; }
    /* ── 17. Active tab ─────────────────────────────────────────────── */
    .stTabs [aria-selected="true"] {
        background: #eff6ff !important;
        color: #2563eb !important;
    }
    .stTabs [data-baseweb="tab"] { color: #475569 !important; }
    /* ── 18. Spinner text ───────────────────────────────────────────── */
    [data-testid="stSpinner"] p { color: var(--text-main) !important; }
    /* ── 19. Number input arrows ────────────────────────────────────── */
    .stNumberInput button { color: var(--text-main) !important; }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)


def load_css():
    """Reads the local stylesheet and applies the theme cascade."""
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS layout: {e}")
    
    # Always inject variable tokens right after the base stylesheet
    inject_theme_css()


def _on_theme_change():
    """Synchronous callback handler to cycle values cleanly before render loops."""
    st.session_state.theme = "light" if st.session_state._theme_toggle else "dark"


def render_sidebar_navigation():
    """Renders the custom emoji theme toggle and the main route selectors."""
    is_light = st.session_state.theme == "light"
    toggle_label = "☀️ Switch to Light Mode" if not is_light else "🌙 Switch to Dark Mode"
    
    # Theme selector widget with state-locked callback
    st.sidebar.toggle(
        toggle_label,
        value=is_light,
        key="_theme_toggle",
        on_change=_on_theme_change,
    )
    
    if st.session_state.theme == "light":
        st.sidebar.caption("☀️ Light mode active")
    else:
        st.sidebar.caption("🌙 Dark mode active")
    
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["1v1", "db", "library"],
        format_func=lambda x: {
            "1v1":     "1v1 Document Comparison",
            "db":      "ChromaDB Knowledge Base Scan",
            "library": "Index Library",
        }[x],
        key="nav_mode",
    )
    st.session_state.current_mode = mode
    return mode


def plotly_theme_layout():
    """Generates real-time map overrides to keep charts in sync with global theme states."""
    is_light = st.session_state.get("theme") == "light"
    font_color = "#0f172a" if is_light else "#f8fafc"
    grid_color = "#e2e8f0" if is_light else "#334155"
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, family="Space Grotesk, Inter, sans-serif"),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
    )


# ── COMPONENT RENDERING CARDS ─────────────────────────────────────────────────
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
    """Renders an AI checker heatmap with dynamic opacity based on confidence."""
    if confidence < 0.4:
        return text
    
    if confidence >= 0.7:
        color_rgb = "244, 63, 94"
        border_color = "#f43f5e"
    else:
        color_rgb = "234, 179, 8"
        border_color = "#eab308"
    
    if confidence >= 0.7:
        alpha = min(0.9, 0.6 + (confidence - 0.7) * 0.5)
    else:
        alpha = 0.3 + (confidence - 0.4) * (0.6 - 0.3) / 0.3
    
    highlighted_html = f'<span style="background-color: rgba({color_rgb}, {alpha:.2f}); border-left: 4px solid {border_color}; padding-left: 4px; display: inline;">{text}</span>'
    return highlighted_html


# ── BACKEND API CONNECTIONS ───────────────────────────────────────────────────
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
    files = {"file": (file.name, file.getvalue(), file.type)}

    resp = requests.post(SEARCH_ENDPOINT, data=data, files=files, timeout=120)
    if not resp.ok:
        raise Exception(resp.text)
    return resp.json()


def call_chroma_ingest_api(file):
    data = {"chunk_strategy": st.session_state.chunk_strategy}
    files = {"file": (file.name, file.getvalue(), file.type)}

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