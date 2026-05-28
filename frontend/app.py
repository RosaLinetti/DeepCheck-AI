"""
DeepCheck-AI — Streamlit Frontend
Premium plagiarism-detection interface wired to unified 1v1 and ChromaDB search endpoints.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# Constants
API_BASE = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT = f"{API_BASE}/document/analyze"
SEARCH_ENDPOINT = f"{API_BASE}/document/analyze/search"
INGEST_ENDPOINT = f"{API_BASE}/document/ingest"
STATS_ENDPOINT = f"{API_BASE}/knowledge-base/stats"
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]

# Page Config
st.set_page_config(
    page_title="DeepCheck AI",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="expanded",  # Opened to show navigation modes
)

# CSS
st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Reduce Streamlit default top padding */
    .block-container {
        padding-top: 1rem !important;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 0rem 1rem 0.4rem;
    }
    .hero-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        /* CHANGED: Removed gradient, made it a simple solid blue */
        color: #3b82f6;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero-header p {
        color: #64748b;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0;
        letter-spacing: 0.2px;
    }

    /* Section titles (panel labels) */
    .panel-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        /* CHANGED: Simple solid gray border */
        border-bottom: 2px solid #334155;
    }

    /* File lock overlay */
    .file-lock-notice {
        /* CHANGED: Simplified background and border colors */
        background: #1e293b;
        border: 1px dashed #475569;
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }

    /* KPI metric cards */
    .kpi-card {
        /* CHANGED: Simple solid dark background and gray border */
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        /* CHANGED: Simple subtle gray shadow on hover */
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #475569;
        margin-top: 0.25rem;
    }

    /* Verdict badges (Kept clean red/yellow/green for messaging) */
    .verdict-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .verdict-plagiarised {
        background: rgba(244, 63, 94, 0.1);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.2);
    }
    .verdict-suspicious {
        background: rgba(234, 179, 8, 0.1);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.2);
    }
    .verdict-original {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    /* Analyse button */
    .stButton > button {
        /* CHANGED: Solid blue background with no gradient */
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.8rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        /* CHANGED: Simple dark drop shadow */
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
    }
    .stButton > button:hover {
        /* CHANGED: Shakes to a slightly darker solid blue on hover */
        background: #2563eb !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        /* CHANGED: Plain, subtle gray divider line */
        background: #334155;
        margin: 2rem 0;
    }

    /* Chunk detail cards */
    .chunk-text-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        /* CHANGED: Plain solid blue color */
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .chunk-text-content {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.55;
        /* CHANGED: Flat dark background and minimal gray border */
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 1rem;
    }

    /* Results section header */
    .results-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .results-header h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        /* CHANGED: Plain solid blue color */
        color: #3b82f6;
    }

    /* Chart section headers */
    .chart-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }

    /* Explorer header */
    .explorer-header h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
</style>""", unsafe_allow_html=True)

# Helper Functions

def severity_color(value: float) -> str:
    """Return a hex color based on similarity severity."""
    if value >= 0.7:
        return "#f43f5e"   # rose
    elif value >= 0.4:
        return "#f59e0b"   # amber
    else:
        return "#34d399"   # emerald


def verdict_badge_html(verdict: str) -> str:
    """Return styled HTML badge for a verdict string."""
    css_class = f"verdict-{verdict.lower().replace(' ', '-')}"
    return f'<span class="verdict-badge {css_class}">{verdict}</span>'


def call_unified_api(source_text, suspicious_text, source_file, suspicious_file, chunk_strategy: str) -> dict:
    """Send a multipart POST to the original unified endpoint."""
    data = {"chunk_strategy": chunk_strategy}
    files = {}

    if source_file is not None:
        files["source_file"] = (source_file.name, source_file.getvalue(), source_file.type or "application/octet-stream")
    elif source_text:
        data["source_text"] = source_text

    if suspicious_file is not None:
        files["suspicious_file"] = (suspicious_file.name, suspicious_file.getvalue(), suspicious_file.type or "application/octet-stream")
    elif suspicious_text:
        data["suspicious_text"] = suspicious_text

    resp = requests.post(UNIFIED_ENDPOINT, data=data, files=files if files else None)
    resp.raise_for_status()
    return resp.json()


def call_chroma_ingest_api(file, chunk_strategy: str) -> dict:
    """Upload a source file to the ChromaDB Reference Library."""
    data = {"chunk_strategy": chunk_strategy}
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    resp = requests.post(INGEST_ENDPOINT, data=data, files=files)
    resp.raise_for_status()
    return resp.json()


def call_chroma_search_api(suspicious_file, chunk_strategy: str, top_k: int = 5) -> dict:
    """Scan a suspicious file against the entire indexed database repository."""
    data = {"chunk_strategy": chunk_strategy, "top_k": top_k}
    files = {"file": (suspicious_file.name, suspicious_file.getvalue(), suspicious_file.type or "application/octet-stream")}
    resp = requests.post(SEARCH_ENDPOINT, data=data, files=files)
    resp.raise_for_status()
    return resp.json()


def get_knowledge_base_stats() -> dict:
    """Fetch analytics information about the database repository."""
    try:
        resp = requests.get(STATS_ENDPOINT)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


# Session State Init
for key in ["results", "source_file_uploaded", "suspicious_file_uploaded", "search_results"]:
    if key not in st.session_state:
        st.session_state[key] = None


# SIDEBAR APPLICATION CONTROLS
with st.sidebar:
    st.markdown("System Navigation")
    app_mode = st.radio(
        "Select Operation Mode:",
        options=["1v1 Document Compare", "Database Index Management", "Database Plagiarism Scan"]
    )
    
    # Render mini stats board inside sidebar if using DB features
    if app_mode in ["Database Index Management", "Database Plagiarism Scan"]:
        st.markdown("---")
        st.markdown("Vector Database Status")
        stats = get_knowledge_base_stats()
        if stats:
            st.caption(f"**Collection:** `{stats.get('collection_name')}`")
            st.caption(f"**Indexed Chunks:** `{stats.get('total_chunks')}`")
        else:
            st.caption("Status: Disconnected from API service")


# HEADER
st.markdown("""
<div class="hero-header">
    <h1>DeepCheck AI</h1>
    <p>Semantic Plagiarism Detection using SBERT Transformers</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


#mode 1: 1v1 document compare
if app_mode == "1v1 Document Compare":
    col_src, col_spacer, col_sus = st.columns([1, 0.05, 1])

    with col_src:
        st.markdown('<div class="panel-label">Reference / Source Text</div>', unsafe_allow_html=True)
        source_file = st.file_uploader("Upload source document", type=ALLOWED_EXTENSIONS, key="source_uploader")

        if source_file is not None:
            st.session_state.source_file_uploaded = source_file
            st.markdown('<div class="file-lock-notice">File uploaded — text input disabled</div>', unsafe_allow_html=True)
            source_text_input = st.text_area("Or paste source text", value="", height=140, key="source_text_area", disabled=True)
            source_text_value = None
        else:
            st.session_state.source_file_uploaded = None
            source_text_input = st.text_area("Or paste source text", height=140, key="source_text_area", placeholder="Paste reference text...")
            source_text_value = source_text_input if source_text_input else None

    with col_sus:
        st.markdown('<div class="panel-label">Suspected / Suspicious Text</div>', unsafe_allow_html=True)
        suspicious_file = st.file_uploader("Upload suspicious document", type=ALLOWED_EXTENSIONS, key="suspicious_uploader")

        if suspicious_file is not None:
            st.session_state.suspicious_file_uploaded = suspicious_file
            st.markdown('<div class="file-lock-notice">File uploaded — text input disabled</div>', unsafe_allow_html=True)
            suspicious_text_input = st.text_area("Or paste suspicious text", value="", height=140, key="suspicious_text_area", disabled=True)
            suspicious_text_value = None
        else:
            st.session_state.suspicious_file_uploaded = None
            suspicious_text_input = st.text_area("Or paste suspicious text", height=140, key="suspicious_text_area", placeholder="Paste suspicious text...")
            suspicious_text_value = suspicious_text_input if suspicious_text_input else None

    strat_col, btn_col = st.columns([4, 1])
    with strat_col:
        chunk_strategy = st.radio("Chunk Strategy", options=["sentence", "sliding_window"], format_func=lambda x: "Sentence" if x == "sentence" else "Sliding Window", horizontal=True)
    with btn_col:
        st.markdown("")
        analyse_clicked = st.button("Analyse 1v1", use_container_width=True)

    if analyse_clicked:
        has_source = source_file is not None or (source_text_value and source_text_value.strip())
        has_suspicious = suspicious_file is not None or (suspicious_text_value and suspicious_text_value.strip())

        if not has_source or not has_suspicious:
            st.error("Please provide input for both the source and suspicious sides.")
        else:
            with st.spinner("Analysing documents..."):
                try:
                    st.session_state.results = call_unified_api(source_text_value, suspicious_text_value, source_file, suspicious_file, chunk_strategy)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.results = None

    # Reference downstream visualizer handle
    active_results_payload = st.session_state.results


# mode 2: database index management
elif app_mode == "Database Index Management":
    st.markdown('<div class="panel-label">Database Management — Add Reference Material</div>', unsafe_allow_html=True)
    
    ingest_file = st.file_uploader("Select a document to permanently index into ChromaDB:", type=ALLOWED_EXTENSIONS, key="ingest_uploader")
    ingest_strat = st.radio("Indexing Chunk Strategy:", options=["sentence", "sliding_window"], horizontal=True, key="ingest_strat")
    
    if st.button("Index Document into Library", use_container_width=True):
        if ingest_file is None:
            st.error("Please upload a file to index.")
        else:
            with st.spinner(f"Splitting and vectorizing '{ingest_file.name}' into persistent memory..."):
                try:
                    res = call_chroma_ingest_api(ingest_file, ingest_strat)
                    st.success(f"Successfully processed and stored reference! ID: {res.get('document_id')} ({res.get('chunks_stored')} blocks added).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to commit document to ChromaDB: {e}")
                    
    active_results_payload = None


# mode 3: database plagiarism scan
elif app_mode == "Database Plagiarism Scan":
    st.markdown('<div class="panel-label">1-vs-Many Repository Plagiarism Sweep</div>', unsafe_allow_html=True)
    
    search_file = st.file_uploader("Upload a student submission to scan against entire reference collection:", type=ALLOWED_EXTENSIONS, key="search_uploader")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        search_strat = st.radio("Evaluation Step Chunking Strategy:", options=["sentence", "sliding_window"], horizontal=True, key="search_strat")
    with sc2:
        top_k_val = st.slider("Max verification matches fetched per block (Top K):", min_value=1, max_value=10, value=5)
        
    if st.button("Execute Repository Search Scan", use_container_width=True):
        if search_file is None:
            st.error("Please provide a file to evaluate.")
        else:
            with st.spinner("Running global vector lookup scan across collection memory..."):
                try:
                    st.session_state.search_results = call_chroma_search_api(search_file, search_strat, top_k_val)
                except Exception as e:
                    st.error(f"Scan interrupted: {e}")
                    st.session_state.search_results = None
                    
    active_results_payload = st.session_state.search_results



# unified render layer for results output
if active_results_payload:
    data = active_results_payload
    chunks = data.get("chunk_matches", [])
    overall_sim = data.get("overall_similarity", 0)
    max_sim = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="results-header"><h2>Analysis Results</h2></div>', unsafe_allow_html=True)

    # KPI Summary Metric Cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Overall Similarity</div><div class="kpi-value" style="color:{severity_color(overall_sim)}">{round(overall_sim * 100, 1)}%</div><div class="kpi-sub">Mean across segments</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Max Similarity</div><div class="kpi-value" style="color:{severity_color(max_sim)}">{round(max_sim * 100, 1)}%</div><div class="kpi-sub">Highest chunk match</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Segments</div><div class="kpi-value" style="color:#2dd4bf">{total_chunks}</div><div class="kpi-sub">Evaluated blocks</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Process Verdict Counter Charts
    verdict_counts = {"original": 0, "suspicious": 0, "plagiarised": 0}
    for c in chunks:
        v = c.get("verdict", "original").lower()
        if v in verdict_counts:
            verdict_counts[v] += 1

    chart_col, heatmap_col = st.columns(2)
    with chart_col:
        st.markdown('<div class="chart-header">Verdict Distribution</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Original", "Suspicious", "Plagiarised"],
            values=[verdict_counts["original"], verdict_counts["suspicious"], verdict_counts["plagiarised"]],
            hole=0.55,
            marker=dict(colors=["#34d399", "#fbbf24", "#f43f5e"], line=dict(color="rgba(0,0,0,0.2)", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340,
            font=dict(family="Inter", color="#e2e8f0"),
            annotations=[dict(text=f"<b>{total_chunks}</b><br><span style='font-size:11px;color:#7a8ba0'>chunks</span>", x=0.5, y=0.5, font_size=22, showarrow=False, font=dict(color="#e2e8f0", family="Space Grotesk"))],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with heatmap_col:
        st.markdown('<div class="chart-header">Chunk Similarity Scores</div>', unsafe_allow_html=True)
        if chunks:
            chunk_indices = list(range(len(chunks)))
            sim_scores = [round(c["similarity_score"] * 100, 1) for c in chunks]

            fig_bar = go.Figure(data=[go.Bar(
                x=chunk_indices, y=sim_scores,
                marker=dict(
                    color=sim_scores,
                    colorscale=[[0, "#34d399"], [0.4, "#34d399"], [0.55, "#fbbf24"], [0.7, "#fbbf24"], [0.85, "#f43f5e"], [1, "#f43f5e"]],
                    cmin=0, cmax=100, line=dict(width=0),
                    colorbar=dict(title=dict(text="Similarity %", font=dict(size=11, color="#7a8ba0")), tickfont=dict(color="#7a8ba0"), thickness=12, len=0.6),
                ),
                hovertemplate="<b>Evaluation Index %{x}</b><br>Similarity: %{y:.1f}%<br><extra></extra>",
            )])
            fig_bar.update_layout(
                xaxis=dict(title="Chunk Match Entry", color="#7a8ba0", gridcolor="rgba(20,184,166,0.06)"),
                yaxis=dict(title="Similarity %", color="#7a8ba0", range=[0, 105], gridcolor="rgba(20,184,166,0.06)"),
                margin=dict(t=20, b=50, l=50, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340,
                font=dict(family="Inter", color="#e2e8f0"), bargap=0.15,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confidence vs Similarity Scatter
    st.markdown('<div class="chart-header">Confidence vs Similarity</div>', unsafe_allow_html=True)
    scatter_data = {
        "Match Entry Index": list(range(len(chunks))),
        "Similarity (%)": [round(c["similarity_score"] * 100, 1) for c in chunks],
        "Confidence (%)": [round(c["confidence"] * 100, 1) for c in chunks],
        "Verdict": [c["verdict"].capitalize() for c in chunks],
    }
    fig_scatter = px.scatter(scatter_data, x="Similarity (%)", y="Confidence (%)", color="Verdict", color_discrete_map={"Original": "#34d399", "Suspicious": "#fbbf24", "Plagiarised": "#f43f5e"}, hover_data=["Match Entry Index"], size="Confidence (%)", size_max=14)
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, font=dict(family="Inter", color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(20,184,166,0.07)", range=[0, 105]), yaxis=dict(gridcolor="rgba(20,184,166,0.07)", range=[0, 105]),
        legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(20,184,166,0.15)", borderwidth=1), margin=dict(t=20, b=40, l=50, r=30),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chunk-wise Detail Explorer
    st.markdown('<div class="explorer-header"><h4>Chunk Detail Explorer</h4><p style="color:#7a8ba0; font-size:0.85rem; margin-top:-0.5rem;">Expand any segment panel to view text content overlays and tracking parameters.</p></div>', unsafe_allow_html=True)

    # Dynamic Filter Controls
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        verdict_filter = st.multiselect("Filter by verdict", options=["original", "suspicious", "plagiarised"], default=["original", "suspicious", "plagiarised"], format_func=str.capitalize)
    with filter_col2:
        sim_range = st.slider("Similarity range (%)", min_value=0, max_value=100, value=(0, 100))

    filtered = [c for c in chunks if c["verdict"].lower() in verdict_filter and sim_range[0] <= c["similarity_score"] * 100 <= sim_range[1]]
    st.caption(f"Showing {len(filtered)} matches inside dashboard workspace view.")

    for i, chunk in enumerate(filtered):
        idx = chunk.get("suspicious_chunk_index", i)
        sim = round(chunk["similarity_score"] * 100, 1)
        conf = round(chunk["confidence"] * 100, 1)
        verdict = chunk["verdict"]
        badge = verdict_badge_html(verdict)
        sim_color = severity_color(chunk["similarity_score"])
        
        # EXTRACT SOURCE METADATA DYNAMICALLY
        source_doc_origin = chunk.get("source_filename", "Direct Input Text Block")

        with st.expander(f"Match Entry · Block #{idx} | {sim}% similarity from [{source_doc_origin}]", expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"**Similarity Match:** <span style='color:{sim_color};font-weight:700'>{sim}%</span>", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"**Classifier Confidence:** <span style='font-weight:700'>{conf}%</span>", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"**System Flag Verdict:** {badge}", unsafe_allow_html=True)

            st.markdown("")

            text_left, text_right = st.columns(2)
            with text_left:
                st.markdown('<div class="chunk-text-label">Evaluated Submission Text</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chunk-text-content">{chunk["suspicious_chunk_text"]}</div>', unsafe_allow_html=True)
            with text_right:
                st.markdown(f'<div class="chunk-text-label">Best Reference Match Source Found</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chunk-text-content">{chunk["best_match_source_text"]}</div>', unsafe_allow_html=True)
