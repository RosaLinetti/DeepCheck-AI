import streamlit as st
import requests
import plotly.graph_objects as go

from components.shared import (
    ALLOWED_EXTENSIONS,
    call_unified_api,
    render_verdict_banner,
    render_kpi_cards,
    render_empty_state,
    render_file_chip,
    render_algorithm_selector,
    highlight_matching_segments,
    severity_color,
    verdict_badge_html,
)


def render_compare_page():
    st.markdown('<div class="page-compare">', unsafe_allow_html=True)
    st.markdown('<h2>1v1 Document Comparison</h2>', unsafe_allow_html=True)

    # Input panels
    col_src, col_sus = st.columns(2, gap="small")

    with col_src:
        st.markdown(
            '<div class="panel-label">Reference / Source Document</div>',
            unsafe_allow_html=True,
        )
        source_file = st.file_uploader(
            "Upload source",
            type=ALLOWED_EXTENSIONS,
            key="cmp_source_uploader",
            label_visibility="collapsed",
        )
        if source_file:
            render_file_chip(source_file, "source")
            source_text_value = None
        else:
            source_text_value = st.text_area(
                "Or paste source text",
                height=150,
                key="cmp_source_text",
                placeholder="Paste reference text here...",
            ) or None

    with col_sus:
        st.markdown(
            '<div class="panel-label">Suspicious Document</div>',
            unsafe_allow_html=True,
        )
        suspicious_file = st.file_uploader(
            "Upload suspicious",
            type=ALLOWED_EXTENSIONS,
            key="cmp_sus_uploader",
            label_visibility="collapsed",
        )
        if suspicious_file:
            render_file_chip(suspicious_file, "suspicious")
            suspicious_text_value = None
        else:
            suspicious_text_value = st.text_area(
                "Or paste suspicious text",
                height=150,
                key="cmp_sus_text",
                placeholder="Paste suspicious text here...",
            ) or None

    # Algorithm selection
    st.markdown("")
    algo = render_algorithm_selector()

    st.markdown("")
    _, btn_col = st.columns([5, 1], gap="small")
    with btn_col:
        analyse_clicked = st.button(
            "Analyse",
            use_container_width=True,
            key="cmp_analyse_btn",
        )

    if analyse_clicked:
        has_source = source_file is not None or bool(source_text_value)
        has_sus = suspicious_file is not None or bool(suspicious_text_value)

        if not has_source or not has_sus:
            st.error("Please provide input for both source and suspicious documents.")
        else:
            algo_label = (
                "SBERT + ML hybrid"
                if algo == "semantic"
                else "Lexical / traditional"
            )
            with st.spinner(f"Chunking → Embedding → Analysing ({algo_label})..."):
                try:
                    st.session_state.results = call_unified_api(
                        source_text_value,
                        suspicious_text_value,
                        source_file,
                        suspicious_file,
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not reach analysis server. Ensure backend is running on port 8000."
                    )
                    st.session_state.results = None
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.results = None

    # Results
    if st.session_state.results:
        _render_results(st.session_state.results)
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_empty_state(
            "",
            "No results yet",
            "Upload or paste two documents above, then click Analyse.",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# RESULTS RENDERER
# ---------------------------
def _render_results(data: dict):
    chunks = data.get("chunk_matches", [])
    overall_sim = data.get("overall_similarity", 0)
    max_sim = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)
    source_name = data.get("source_filename", "Original Source")
    suspicious_name = data.get("suspicious_filename", "Submitted Document")
    algo_used = data.get("algorithm_used", "semantic")

    algo_label = (
        "AI Semantic (SBERT + ML hybrid classifier)"
        if algo_used == "semantic"
        else "Traditional (Lexical / string overlap)"
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2>Directional Plagiarism Analysis</h2>', unsafe_allow_html=True)

    st.markdown(
        f"""
Analysis Verdict: The submitted file "{suspicious_name}" contains text blocks
matching the source file "{source_name}".

Algorithm used: {algo_label}
        """
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    render_verdict_banner(overall_sim, is_1v1=True)
    render_kpi_cards(overall_sim, max_sim, total_chunks)

    threshold = st.slider(
        "Plagiarism threshold",
        0,
        100,
        70,
        step=1,
        key="cmp_threshold",
    ) / 100

    def dynamic_verdict(score):
        if score >= threshold:
            return "plagiarised"
        elif score >= threshold * 0.55:
            return "suspicious"
        return "original"

    live_counts = {"original": 0, "suspicious": 0, "plagiarised": 0}
    for c in chunks:
        live_counts[dynamic_verdict(c["similarity_score"])] += 1

    # Charts
    chart_col, bar_col = st.columns(2, gap="small")

    with chart_col:
        st.markdown("Verdict distribution")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Original", "Suspicious", "Plagiarised"],
            values=[
                live_counts["original"],
                live_counts["suspicious"],
                live_counts["plagiarised"],
            ],
            hole=0.55,
        )])
        st.plotly_chart(fig_donut, use_container_width=True)

    with bar_col:
        st.markdown("Chunk similarity scores")

        sim_scores = [c["similarity_score"] * 100 for c in chunks]

        fig_bar = go.Figure(data=[go.Bar(
            x=list(range(len(chunks))),
            y=sim_scores,
        )])

        fig_bar.add_hline(y=threshold * 100)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chunk explorer
    st.markdown("Chunk Detail Explorer")

    verdict_filter = st.multiselect(
        "Filter by verdict",
        ["original", "suspicious", "plagiarised"],
        default=["original", "suspicious", "plagiarised"],
        key="cmp_verdict_filter",
    )

    sim_range = st.slider(
        "Similarity range (%)",
        0,
        100,
        (0, 100),
        key="cmp_sim_range",
    )

    filtered = [
        c for c in chunks
        if dynamic_verdict(c["similarity_score"]) in verdict_filter
        and sim_range[0] <= c["similarity_score"] * 100 <= sim_range[1]
    ]

    st.caption(f"Showing {len(filtered)} of {len(chunks)} segments.")

    page_size = 15
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    page = 0
    if total_pages > 1:
        page = st.number_input("Page", 1, total_pages, 1, key="cmp_page") - 1

    for i, chunk in enumerate(filtered[page * page_size:(page + 1) * page_size]):
        idx = chunk.get("suspicious_chunk_index", i)
        sim = round(chunk["similarity_score"] * 100, 1)
        conf = round(chunk["confidence"] * 100, 1)
        verdict = dynamic_verdict(chunk["similarity_score"])
        badge = verdict_badge_html(verdict)
        col = severity_color(chunk["similarity_score"])

        open_key = f"cmp_block_open_{page * page_size + i}"
        is_open = st.session_state.get(open_key, False)

        if st.button(
            f"{'▼' if is_open else '▶'} Block {idx} · {sim}%",
            key=f"cmp_toggle_{page * page_size + i}",
        ):
            st.session_state[open_key] = not is_open
            st.rerun()

        if st.session_state.get(open_key, False):
            st.markdown(f"Similarity: {sim}%")
            st.markdown(f"Confidence: {conf}%")
            st.markdown(f"Verdict: {badge}")

            st.markdown("Source text")
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
