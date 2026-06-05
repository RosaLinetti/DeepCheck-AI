import streamlit as st
import plotly.graph_objects as go

from components.shared import (
    ALLOWED_EXTENSIONS,
    call_chroma_search_api,
    get_knowledge_base_stats,
    render_verdict_banner,
    render_kpi_cards,
    render_empty_state,
    render_file_chip,
    render_db_status_cards,
    render_algorithm_selector,
    highlight_matching_segments,
    severity_color,
    verdict_badge_html,
)


def render_scan_page():
    st.markdown('<div class="page-scan">', unsafe_allow_html=True)
    st.markdown('<h2>ChromaDB Knowledge Base Scan</h2>', unsafe_allow_html=True)

    stats = get_knowledge_base_stats()

    # DB empty guard
    if not stats or stats.get("total_chunks", 0) == 0:
        st.info(
            "No documents indexed yet — go to the Index Library tab to add reference material before scanning."
        )
        return

    render_db_status_cards(stats)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Upload Submission to Scan</div>', unsafe_allow_html=True)

    scan_file = st.file_uploader(
        "Submission file",
        type=ALLOWED_EXTENSIONS,
        key="scan_uploader",
        label_visibility="collapsed",
    )

    if scan_file:
        render_file_chip(scan_file, "submission")

    col1, col2 = st.columns(2)

    with col1:
        top_k = st.slider(
            "Reference matches per chunk (top K)",
            min_value=1,
            max_value=10,
            value=5,
            key="scan_topk",
        )

    with col2:
        algo = render_algorithm_selector()

    _, btn_col = st.columns([5, 1])
    with btn_col:
        scan_clicked = st.button("Run scan", use_container_width=True, key="scan_btn")

    if scan_clicked:
        if scan_file is None:
            st.error("Please upload a submission file to scan.")
        else:
            algo_label = (
                "SBERT + ML hybrid"
                if algo == "semantic"
                else "Lexical / traditional"
            )
            with st.spinner(f"Chunking → Embedding → Searching ({algo_label})…"):
                try:
                    st.session_state.search_results = call_chroma_search_api(
                        scan_file, top_k
                    )
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    st.session_state.search_results = None

    if st.session_state.get("search_results"):
        _render_scan_results(st.session_state.search_results)
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_empty_state(
            "Ready to scan",
            "Upload a submission above and click Run scan",
            ""
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# RESULTS
# ---------------------------
def _render_scan_results(data: dict):
    chunks = data.get("chunk_matches", [])
    overall_sim = data.get("overall_similarity", 0)
    max_sim = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)
    suspicious_name = data.get("suspicious_filename", "Submission")
    auto_ingested = data.get("auto_ingested", False)
    db_chunks_searched = data.get("knowledge_base_chunks_searched", 0)
    algo_used = data.get("algorithm_used", "semantic")

    algo_label = (
        "AI Semantic (SBERT + ML hybrid classifier)"
        if algo_used == "semantic"
        else "Traditional (Lexical / string overlap)"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>Scan Results</h2>', unsafe_allow_html=True)

    if auto_ingested:
        st.success(
            f"Clean submission auto-ingested. {suspicious_name} added to knowledge base."
        )

    st.markdown(
        f"Database searched: {db_chunks_searched} chunks  \n"
        f"Algorithm: {algo_label}"
    )

    render_verdict_banner(overall_sim, is_1v1=False)
    render_kpi_cards(overall_sim, max_sim, total_chunks)

    threshold = st.slider(
        "Plagiarism threshold",
        0,
        100,
        70,
        key="scan_threshold",
    ) / 100

    def dynamic_verdict(score):
        if score >= threshold:
            return "plagiarised"
        elif score >= threshold * 0.55:
            return "suspicious"
        return "original"

    # Source breakdown
    source_scores = {}
    for c in chunks:
        src = c.get("source_filename", "Unknown")
        source_scores.setdefault(src, [])
        source_scores[src].append(c["similarity_score"])

    if len(source_scores) > 1:
        st.markdown("Matches by source document")

        labels = list(source_scores.keys())
        means = [sum(v) / len(v) * 100 for v in source_scores.values()]

        fig = go.Figure(go.Bar(
            x=means,
            y=labels,
            orientation="h",
            marker=dict(color="#3b82f6")
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Add verdict distribution chart
    verdict_counts = {"original": 0, "suspicious": 0, "plagiarised": 0}
    for c in chunks:
        verdict = dynamic_verdict(c["similarity_score"])
        verdict_counts[verdict] += 1
    
    # Create donut chart with semantic colors
    if sum(verdict_counts.values()) > 0:
        st.markdown("Verdict Distribution")
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=list(verdict_counts.keys()),
            values=list(verdict_counts.values()),
            hole=0.4,
            marker=dict(
                colors=[
                    "#22c55e",  # original - Emerald Green
                    "#eab308",  # suspicious - Warning Amber
                    "#f43f5e",  # plagiarised - Alert Red
                ]
            ),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Chunk explorer
    st.markdown("Chunk Detail Explorer")

    vf = st.multiselect(
        "Filter by verdict",
        ["original", "suspicious", "plagiarised"],
        default=["original", "suspicious", "plagiarised"],
        key="scan_verdict_filter",
    )

    sr = st.slider("Similarity range (%)", 0, 100, (0, 100), key="scan_sim_range")

    filtered = [
        c for c in chunks
        if dynamic_verdict(c["similarity_score"]) in vf
        and sr[0] <= c["similarity_score"] * 100 <= sr[1]
    ]

    st.caption(f"Showing {len(filtered)} of {len(chunks)} matches.")

    page_size = 15
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    page = 0
    if total_pages > 1:
        page = st.number_input("Page", 1, total_pages, 1, key="scan_page") - 1

    for i, chunk in enumerate(filtered[page * page_size:(page + 1) * page_size]):
        idx = chunk.get("suspicious_chunk_index", i)
        sim = round(chunk["similarity_score"] * 100, 1)
        conf = round(chunk["confidence"] * 100, 1)
        verdict = dynamic_verdict(chunk["similarity_score"])
        badge = verdict_badge_html(verdict)
        col = severity_color(chunk["similarity_score"])
        src = chunk.get("source_filename", "Library document")

        open_key = f"scan_block_open_{page * page_size + i}"
        is_open = st.session_state.get(open_key, False)

        if st.button(f"{'▼' if is_open else '▶'} Block {idx} · {sim}% · {src}",
                     key=f"scan_toggle_{page * page_size + i}"):
            st.session_state[open_key] = not is_open
            st.rerun()

        if st.session_state.get(open_key, False):
            st.markdown(f"Similarity: {sim}%")
            st.markdown(f"Confidence: {conf}%")
            st.markdown(f"Verdict: {badge}")

            st.markdown("Matched reference")
            st.markdown(
                highlight_matching_segments(
                    chunk["best_match_source_text"],
                    chunk["suspicious_chunk_text"],
                    conf / 100,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("Submitted text")
            st.markdown(
                highlight_matching_segments(
                    chunk["suspicious_chunk_text"],
                    chunk["best_match_source_text"],
                    conf / 100,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
