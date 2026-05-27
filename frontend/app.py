"""
DeepCheck-AI — Streamlit Frontend
Premium plagiarism-detection interface wired to the /document/analyze/unified endpoint.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# Constants
API_BASE = "http://127.0.0.1:8000"
UNIFIED_ENDPOINT = f"{API_BASE}/document/analyze/unified"
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx"]

# Page Config
st.set_page_config(
    page_title="DeepCheck AI",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    /* Hide sidebar & hamburger */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

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
        background: linear-gradient(135deg, #14b8a6, #2dd4bf, #5eead4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .hero-header p {
        color: #7a8ba0;
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
        border-bottom: 2px solid rgba(20, 184, 166, 0.25);
    }

    /* File lock overlay */
    .file-lock-notice {
        background: rgba(20, 184, 166, 0.06);
        border: 1px dashed rgba(20, 184, 166, 0.3);
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
        color: #5eead4;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }

    /* KPI metric cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(20, 184, 166, 0.12);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(20, 184, 166, 0.1);
    }
    .kpi-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #7a8ba0;
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
        color: #5a6b7d;
        margin-top: 0.25rem;
    }

    /* Verdict badges */
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
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.25);
    }
    .verdict-suspicious {
        background: rgba(251, 191, 36, 0.10);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    .verdict-original {
        background: rgba(52, 211, 153, 0.10);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }

    /* Analyse button */
    .stButton > button {
        background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.8rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #14b8a6, #2dd4bf) !important;
        box-shadow: 0 6px 22px rgba(20, 184, 166, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(20,184,166,0.2), transparent);
        margin: 2rem 0;
    }

    /* Chunk detail cards */
    .chunk-text-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #5eead4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .chunk-text-content {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.55;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(20, 184, 166, 0.08);
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
        background: linear-gradient(135deg, #14b8a6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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

    /* Hide Streamlit footer & menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
</style>
""", unsafe_allow_html=True)


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


def call_unified_api(
    source_text,
    suspicious_text,
    source_file,
    suspicious_file,
    chunk_strategy: str,
) -> dict:
    """Send a multipart POST to the unified endpoint."""
    data = {"chunk_strategy": chunk_strategy}
    files = {}

    # Source side
    if source_file is not None:
        files["source_file"] = (
            source_file.name,
            source_file.getvalue(),
            source_file.type or "application/octet-stream",
        )
    elif source_text:
        data["source_text"] = source_text

    # Suspicious side
    if suspicious_file is not None:
        files["suspicious_file"] = (
            suspicious_file.name,
            suspicious_file.getvalue(),
            suspicious_file.type or "application/octet-stream",
        )
    elif suspicious_text:
        data["suspicious_text"] = suspicious_text

    resp = requests.post(UNIFIED_ENDPOINT, data=data, files=files if files else None)
    resp.raise_for_status()
    return resp.json()


# Session State Init
for key in ["results", "source_file_uploaded", "suspicious_file_uploaded"]:
    if key not in st.session_state:
        st.session_state[key] = None


# HEADER

st.markdown("""
<div class="hero-header">
    <h1>DeepCheck AI</h1>
    <p>Semantic Plagiarism Detection · Powered by SBERT Transformers</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# INPUT PANELS

col_src, col_spacer, col_sus = st.columns([1, 0.05, 1])

# Source (left)
with col_src:
    st.markdown('<div class="panel-label">Reference / Source Text</div>', unsafe_allow_html=True)

    source_file = st.file_uploader(
        "Upload source document",
        type=ALLOWED_EXTENSIONS,
        key="source_uploader",
        help="Supported: .pdf, .txt, .docx",
    )

    if source_file is not None:
        st.session_state.source_file_uploaded = source_file
        st.markdown(
            '<div class="file-lock-notice">File uploaded — text input disabled</div>',
            unsafe_allow_html=True,
        )
        source_text_input = st.text_area(
            "Or paste source text",
            value="",
            height=140,
            key="source_text_area",
            disabled=True,
            placeholder="Text input locked while a file is uploaded...",
        )
        source_text_value = None
    else:
        st.session_state.source_file_uploaded = None
        source_text_input = st.text_area(
            "Or paste source text",
            height=140,
            key="source_text_area",
            placeholder="Paste or type the original / reference text here...",
        )
        source_text_value = source_text_input if source_text_input else None

# Suspicious (right)
with col_sus:
    st.markdown('<div class="panel-label">Suspected / Suspicious Text</div>', unsafe_allow_html=True)

    suspicious_file = st.file_uploader(
        "Upload suspicious document",
        type=ALLOWED_EXTENSIONS,
        key="suspicious_uploader",
        help="Supported: .pdf, .txt, .docx",
    )

    if suspicious_file is not None:
        st.session_state.suspicious_file_uploaded = suspicious_file
        st.markdown(
            '<div class="file-lock-notice">File uploaded — text input disabled</div>',
            unsafe_allow_html=True,
        )
        suspicious_text_input = st.text_area(
            "Or paste suspicious text",
            value="",
            height=140,
            key="suspicious_text_area",
            disabled=True,
            placeholder="Text input locked while a file is uploaded...",
        )
        suspicious_text_value = None
    else:
        st.session_state.suspicious_file_uploaded = None
        suspicious_text_input = st.text_area(
            "Or paste suspicious text",
            height=140,
            key="suspicious_text_area",
            placeholder="Paste or type the suspicious text here...",
        )
        suspicious_text_value = suspicious_text_input if suspicious_text_input else None


# STRATEGY SELECTOR + ANALYSE BUTTON

st.markdown("")
strat_col, btn_col = st.columns([4, 1])

with strat_col:
    chunk_strategy = st.radio(
        "Chunk Strategy",
        options=["sentence", "sliding_window"],
        format_func=lambda x: "Sentence" if x == "sentence" else "Sliding Window",
        horizontal=True,
        help="Sentence splits on punctuation; Sliding Window uses fixed-token windows with overlap.",
    )

with btn_col:
    st.markdown("")  # vertical spacing
    analyse_clicked = st.button("Analyse", use_container_width=True)


# API CALL

if analyse_clicked:
    # Validation
    has_source = source_file is not None or (source_text_value and source_text_value.strip())
    has_suspicious = suspicious_file is not None or (suspicious_text_value and suspicious_text_value.strip())

    if not has_source or not has_suspicious:
        st.error("Please provide input for both the source and suspicious sides (text or file).")
    else:
        with st.spinner("Analysing documents — this may take a moment..."):
            try:
                result = call_unified_api(
                    source_text=source_text_value,
                    suspicious_text=suspicious_text_value,
                    source_file=source_file,
                    suspicious_file=suspicious_file,
                    chunk_strategy=chunk_strategy,
                )
                st.session_state.results = result
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the DeepCheck API. Make sure the backend is running at http://127.0.0.1:8000.")
                st.session_state.results = None
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"API Error: {detail}")
                st.session_state.results = None
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state.results = None


# RESULTS VISUALISATION

if st.session_state.results:
    data = st.session_state.results
    chunks = data.get("chunk_matches", [])
    overall_sim = data.get("overall_similarity", 0)
    max_sim = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Section Title
    st.markdown("""
    <div class="results-header">
        <h2>Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    k1, k2, k3 = st.columns(3)

    with k1:
        ov_pct = round(overall_sim * 100, 1)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Similarity</div>
            <div class="kpi-value" style="color:{severity_color(overall_sim)}">{ov_pct}%</div>
            <div class="kpi-sub">Mean across all chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        mx_pct = round(max_sim * 100, 1)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Max Similarity</div>
            <div class="kpi-value" style="color:{severity_color(max_sim)}">{mx_pct}%</div>
            <div class="kpi-sub">Highest chunk match</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Chunks</div>
            <div class="kpi-value" style="color:#2dd4bf">{total_chunks}</div>
            <div class="kpi-sub">Total analysed segments</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Verdict Distribution Donut
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
            marker=dict(
                colors=["#34d399", "#fbbf24", "#f43f5e"],
                line=dict(color="rgba(0,0,0,0.2)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            font=dict(family="Inter", color="#e2e8f0"),
            annotations=[dict(
                text=f"<b>{total_chunks}</b><br><span style='font-size:11px;color:#7a8ba0'>chunks</span>",
                x=0.5, y=0.5, font_size=22, showarrow=False,
                font=dict(color="#e2e8f0", family="Space Grotesk"),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Similarity Bar Chart
    with heatmap_col:
        st.markdown('<div class="chart-header">Chunk Similarity Scores</div>', unsafe_allow_html=True)
        if chunks:
            chunk_indices = [c["suspicious_chunk_index"] for c in chunks]
            sim_scores = [round(c["similarity_score"] * 100, 1) for c in chunks]

            fig_bar = go.Figure(data=[go.Bar(
                x=chunk_indices,
                y=sim_scores,
                marker=dict(
                    color=sim_scores,
                    colorscale=[
                        [0, "#34d399"],
                        [0.4, "#34d399"],
                        [0.55, "#fbbf24"],
                        [0.7, "#fbbf24"],
                        [0.85, "#f43f5e"],
                        [1, "#f43f5e"],
                    ],
                    cmin=0,
                    cmax=100,
                    line=dict(width=0),
                    colorbar=dict(
                        title=dict(text="Similarity %", font=dict(size=11, color="#7a8ba0")),
                        tickfont=dict(color="#7a8ba0"),
                        thickness=12,
                        len=0.6,
                    ),
                ),
                hovertemplate=(
                    "<b>Chunk %{x}</b><br>"
                    "Similarity: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            )])
            fig_bar.update_layout(
                xaxis=dict(
                    title="Chunk Index",
                    color="#7a8ba0",
                    gridcolor="rgba(20,184,166,0.06)",
                ),
                yaxis=dict(
                    title="Similarity %",
                    color="#7a8ba0",
                    range=[0, 105],
                    gridcolor="rgba(20,184,166,0.06)",
                ),
                margin=dict(t=20, b=50, l=50, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                font=dict(family="Inter", color="#e2e8f0"),
                bargap=0.15,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No chunk data to display.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confidence vs Similarity Scatter
    if chunks:
        st.markdown('<div class="chart-header">Confidence vs Similarity</div>', unsafe_allow_html=True)
        scatter_data = {
            "Chunk Index": [c["suspicious_chunk_index"] for c in chunks],
            "Similarity (%)": [round(c["similarity_score"] * 100, 1) for c in chunks],
            "Confidence (%)": [round(c["confidence"] * 100, 1) for c in chunks],
            "Verdict": [c["verdict"].capitalize() for c in chunks],
        }
        color_map = {"Original": "#34d399", "Suspicious": "#fbbf24", "Plagiarised": "#f43f5e"}
        fig_scatter = px.scatter(
            scatter_data,
            x="Similarity (%)",
            y="Confidence (%)",
            color="Verdict",
            color_discrete_map=color_map,
            hover_data=["Chunk Index"],
            size="Confidence (%)",
            size_max=14,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            font=dict(family="Inter", color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(20,184,166,0.07)", range=[0, 105]),
            yaxis=dict(gridcolor="rgba(20,184,166,0.07)", range=[0, 105]),
            legend=dict(
                bgcolor="rgba(15,23,42,0.7)",
                bordercolor="rgba(20,184,166,0.15)",
                borderwidth=1,
                font=dict(size=12),
            ),
            margin=dict(t=20, b=40, l=50, r=30),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chunk-wise Detail Explorer
    st.markdown("""
    <div class="explorer-header" style="margin-bottom:1rem;">
        <h4>Chunk Detail Explorer</h4>
        <p style="color:#7a8ba0; font-size:0.85rem; margin-top:-0.5rem;">
            Expand any chunk to see the suspicious text alongside its best source match.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        verdict_filter = st.multiselect(
            "Filter by verdict",
            options=["original", "suspicious", "plagiarised"],
            default=["original", "suspicious", "plagiarised"],
            format_func=str.capitalize,
        )
    with filter_col2:
        sim_range = st.slider(
            "Similarity range (%)",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1,
        )

    filtered = [
        c for c in chunks
        if c["verdict"].lower() in verdict_filter
        and sim_range[0] <= c["similarity_score"] * 100 <= sim_range[1]
    ]

    st.caption(f"Showing {len(filtered)} of {len(chunks)} chunks")

    for chunk in filtered:
        idx = chunk["suspicious_chunk_index"]
        sim = round(chunk["similarity_score"] * 100, 1)
        conf = round(chunk["confidence"] * 100, 1)
        verdict = chunk["verdict"]
        badge = verdict_badge_html(verdict)
        sim_color = severity_color(chunk["similarity_score"])

        with st.expander(f"Chunk #{idx} · {sim}% similarity | {verdict.capitalize()}", expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"**Similarity:** <span style='color:{sim_color};font-weight:700'>{sim}%</span>", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"**Confidence:** <span style='font-weight:700'>{conf}%</span>", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)

            st.markdown("")

            text_left, text_right = st.columns(2)
            with text_left:
                st.markdown('<div class="chunk-text-label">Suspicious Text</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="chunk-text-content">{chunk["suspicious_chunk_text"]}</div>',
                    unsafe_allow_html=True,
                )
            with text_right:
                st.markdown('<div class="chunk-text-label">Best Source Match</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="chunk-text-content">{chunk["best_match_source_text"]}</div>',
                    unsafe_allow_html=True,
                )

    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:1rem; color:#5a6b7d; font-size:0.78rem;">
        DeepCheck AI · Semantic Plagiarism Detection · Powered by SBERT Transformers
    </div>
    """, unsafe_allow_html=True)
