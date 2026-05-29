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
    severity_color,
    verdict_badge_html,
)


def render_scan_page():
    st.markdown('<div class="page-scan">', unsafe_allow_html=True)
    stats = get_knowledge_base_stats()

    # ── DB empty guard ───────────────────────────────────────────────────────
    if not stats or stats.get("total_chunks", 0) == 0:
        st.info(
            "No documents indexed yet — go to the **Manage Library** tab "
            "to add reference material before scanning.",
            icon="📚",
        )
        return

    # ── DB status strip ─────────────────────────────────────────────────────
    render_db_status_cards(stats)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-label upload-label">Upload submission to scan</div>',
                unsafe_allow_html=True)

    scan_file = st.file_uploader(
        "Submission file", type=ALLOWED_EXTENSIONS, key="scan_uploader",
        label_visibility="collapsed",
    )
    if scan_file:
        render_file_chip(scan_file, "submission")

    top_k = st.slider(
        "Reference matches per chunk (top K)", min_value=1, max_value=10, value=5,
        key="scan_topk",
    )

    _, btn_col = st.columns([5, 1], gap="small")
    with btn_col:
        scan_clicked = st.button("Run scan", use_container_width=True, key="scan_btn")

    if scan_clicked:
        if scan_file is None:
            st.error("Please upload a submission file to scan.")
        else:
            with st.spinner("Chunking → Embedding → Searching library…"):
                try:
                    st.session_state.search_results = call_chroma_search_api(scan_file, top_k)
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                    st.session_state.search_results = None

    if st.session_state.get("search_results"):
        _render_scan_results(st.session_state.search_results)
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_empty_state(
            "🗄️", "Ready to scan",
            "Upload a submission above and click Run scan.",
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_scan_results(data: dict):
    chunks       = data.get("chunk_matches", [])
    overall_sim  = data.get("overall_similarity", 0)
    max_sim      = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="results-header"><h2>Scan results</h2></div>',
                unsafe_allow_html=True)

    render_verdict_banner(overall_sim)
    render_kpi_cards(overall_sim, max_sim, total_chunks)
    st.markdown("")

    threshold = st.slider(
        "Plagiarism threshold", 0, 100, 70, step=1, key="scan_threshold",
    ) / 100

    def dynamic_verdict(score):
        if score >= threshold:
            return "plagiarised"
        elif score >= threshold * 0.55:
            return "suspicious"
        return "original"

    # ── Source breakdown chart ───────────────────────────────────────────────
    source_scores: dict = {}
    for c in chunks:
        src = c.get("source_filename", "Unknown")
        source_scores.setdefault(src, [])
        source_scores[src].append(c["similarity_score"])

    if len(source_scores) > 1:
        st.markdown('<div class="chart-header">Matches by source document</div>',
                    unsafe_allow_html=True)
        src_labels = list(source_scores.keys())
        src_means  = [round(sum(v) / len(v) * 100, 1) for v in source_scores.values()]
        fig_src = go.Figure(go.Bar(
            x=src_means, y=src_labels, orientation="h",
            marker=dict(color="#3b82f6", line=dict(width=0)),
            hovertemplate="%{y}: %{x:.1f}% avg similarity<extra></extra>",
        ))
        fig_src.update_layout(
            xaxis=dict(title="Avg similarity %", color="#7a8ba0",
                       gridcolor="rgba(255,255,255,0.04)", range=[0, 105]),
            yaxis=dict(color="#7a8ba0"),
            margin=dict(t=10, b=40, l=10, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=max(160, len(src_labels) * 40),
            font=dict(family="Inter", color="#e2e8f0"),
        )
        st.plotly_chart(fig_src, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Chunk explorer ───────────────────────────────────────────────────────
    st.markdown("#### Chunk detail explorer")
    f1, f2 = st.columns([1, 2], gap="small")
    with f1:
        vf = st.multiselect(
            "Filter by verdict",
            ["original", "suspicious", "plagiarised"],
            default=["original", "suspicious", "plagiarised"],
            format_func=str.capitalize,
            key="scan_verdict_filter",
        )
    with f2:
        sr = st.slider("Similarity range (%)", 0, 100, (0, 100), key="scan_sim_range")

    filtered = [
        c for c in chunks
        if dynamic_verdict(c["similarity_score"]) in vf
        and sr[0] <= c["similarity_score"] * 100 <= sr[1]
    ]
    st.caption(f"Showing {len(filtered)} of {len(chunks)} matches.")

    page_size   = 15
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = 0
    if total_pages > 1:
        page = st.number_input("Page", 1, total_pages, 1, key="scan_page") - 1

    for i, chunk in enumerate(filtered[page * page_size: (page + 1) * page_size]):
        row_id = page * page_size + i
        idx     = chunk.get("suspicious_chunk_index", i)
        sim     = round(chunk["similarity_score"] * 100, 1)
        conf    = round(chunk["confidence"] * 100, 1)
        verdict = dynamic_verdict(chunk["similarity_score"])
        badge   = verdict_badge_html(verdict)
        col     = severity_color(chunk["similarity_score"])
        src     = chunk.get("source_filename", "Library document")
        open_key = f"scan_block_open_{row_id}"
        is_open = st.session_state.get(open_key, False)

        st.markdown('<div class="cmp-toggle-row">', unsafe_allow_html=True)
        if st.button(
            f"{'▼' if is_open else '▶'} Block #{idx} · {sim}% · matched in [{src}]",
            key=f"scan_toggle_{row_id}",
            use_container_width=True,
        ):
            st.session_state[open_key] = not is_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get(open_key, False):
            st.markdown('<div class="cmp-block">', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3, gap="small")
            with mc1:
                st.markdown(
                    f"**Similarity:** <span style='color:{col};font-weight:700'>{sim}%</span>",
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(f"**Confidence:** {conf}%")
            with mc3:
                st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
            st.markdown("")

            tl, tr = st.columns(2, gap="small")
            with tl:
                st.markdown('<div class="chunk-text-label">Submitted text</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chunk-text-content">{chunk["suspicious_chunk_text"]}</div>', unsafe_allow_html=True)
            with tr:
                st.markdown('<div class="chunk-text-label">Matched reference</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chunk-text-content">{chunk["best_match_source_text"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
