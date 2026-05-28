import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

from components.shared import (
    ALLOWED_EXTENSIONS,
    call_unified_api,
    render_verdict_banner,
    render_kpi_cards,
    render_empty_state,
    render_file_chip,
    severity_color,
    verdict_badge_html,
)


def render_compare_page():
    # ── Input panels ──────────────────────────────────────────────────────────
    col_src, col_sus = st.columns(2)

    with col_src:
        st.markdown('<div class="panel-label">Reference / source document</div>',
                    unsafe_allow_html=True)
        source_file = st.file_uploader(
            "Upload source", type=ALLOWED_EXTENSIONS, key="cmp_source_uploader",
            label_visibility="collapsed",
        )
        if source_file:
            render_file_chip(source_file, "source")
            source_text_value = None
        else:
            source_text_value = st.text_area(
                "Or paste source text", height=150, key="cmp_source_text",
                placeholder="Paste reference text here…",
            ) or None

    with col_sus:
        st.markdown('<div class="panel-label">Suspected / suspicious document</div>',
                    unsafe_allow_html=True)
        suspicious_file = st.file_uploader(
            "Upload suspicious", type=ALLOWED_EXTENSIONS, key="cmp_sus_uploader",
            label_visibility="collapsed",
        )
        if suspicious_file:
            render_file_chip(suspicious_file, "suspicious")
            suspicious_text_value = None
        else:
            suspicious_text_value = st.text_area(
                "Or paste suspicious text", height=150, key="cmp_sus_text",
                placeholder="Paste suspicious text here…",
            ) or None

    st.markdown("")
    _, btn_col = st.columns([5, 1])
    with btn_col:
        analyse_clicked = st.button("Analyse", use_container_width=True, key="cmp_analyse_btn")

    if analyse_clicked:
        has_source = source_file is not None or bool(source_text_value)
        has_sus    = suspicious_file is not None or bool(suspicious_text_value)
        if not has_source or not has_sus:
            st.error("Please provide input for both the source and suspicious sides.")
        else:
            with st.spinner("Chunking → Embedding → Comparing…"):
                try:
                    st.session_state.results = call_unified_api(
                        source_text_value, suspicious_text_value,
                        source_file, suspicious_file,
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the analysis server — make sure the backend is running on port 8000.")
                    st.session_state.results = None
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.results = None

    # ── Results ─────────────────────────────────────────────────────────────
    if st.session_state.results:
        _render_results(st.session_state.results)
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_empty_state(
            "🔍", "No results yet",
            "Upload or paste two documents above, then click Analyse.",
        )


def _render_results(data: dict):
    import requests  # local import to avoid circular if ever split further

    chunks       = data.get("chunk_matches", [])
    overall_sim  = data.get("overall_similarity", 0)
    max_sim      = data.get("max_similarity", 0)
    total_chunks = data.get("total_suspicious_chunks", 0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="results-header"><h2>Analysis results</h2></div>',
                unsafe_allow_html=True)

    render_verdict_banner(overall_sim)
    render_kpi_cards(overall_sim, max_sim, total_chunks)
    st.markdown("")

    # ── Live threshold slider ─────────────────────────────────────────────────
    threshold = st.slider(
        "Plagiarism threshold — drag to see verdict counts update live",
        min_value=0, max_value=100, value=70, step=1, key="cmp_threshold",
    ) / 100

    # Recompute verdicts against current threshold
    def dynamic_verdict(score):
        if score >= threshold:
            return "plagiarised"
        elif score >= threshold * 0.55:
            return "suspicious"
        return "original"

    live_counts = {"original": 0, "suspicious": 0, "plagiarised": 0}
    for c in chunks:
        live_counts[dynamic_verdict(c["similarity_score"])] += 1

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col, bar_col = st.columns(2)

    with chart_col:
        st.markdown('<div class="chart-header">Verdict distribution</div>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Original", "Suspicious", "Plagiarised"],
            values=[live_counts["original"], live_counts["suspicious"], live_counts["plagiarised"]],
            hole=0.55,
            marker=dict(
                colors=["#34d399", "#fbbf24", "#f43f5e"],
                line=dict(color="rgba(0,0,0,0.15)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, family="Inter"),
            hovertemplate="<b>%{label}</b><br>%{value} chunks<br>%{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            font=dict(family="Inter", color="#e2e8f0"),
            annotations=[dict(
                text=f"<b>{total_chunks}</b><br><span style='font-size:10px;color:#7a8ba0'>chunks</span>",
                x=0.5, y=0.5, font_size=20, showarrow=False,
                font=dict(color="#e2e8f0", family="Space Grotesk"),
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with bar_col:
        st.markdown('<div class="chart-header">Chunk similarity scores</div>',
                    unsafe_allow_html=True)
        if chunks:
            sim_scores = [round(c["similarity_score"] * 100, 1) for c in chunks]
            bar_colors = [
                "#f43f5e" if s / 100 >= threshold
                else "#fbbf24" if s / 100 >= threshold * 0.55
                else "#34d399"
                for s in sim_scores
            ]
            fig_bar = go.Figure(data=[go.Bar(
                x=list(range(len(chunks))),
                y=sim_scores,
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="<b>Chunk %{x}</b><br>Similarity: %{y:.1f}%<extra></extra>",
            )])
            fig_bar.add_hline(
                y=threshold * 100,
                line_dash="dash", line_color="#94a3b8", line_width=1,
                annotation_text=f"Threshold {round(threshold * 100)}%",
                annotation_font_color="#94a3b8", annotation_font_size=11,
            )
            fig_bar.update_layout(
                xaxis=dict(title="Chunk index", color="#7a8ba0",
                           gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Similarity %", color="#7a8ba0", range=[0, 105],
                           gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(t=20, b=50, l=50, r=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, font=dict(family="Inter", color="#e2e8f0"), bargap=0.15,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Sequence view ─────────────────────────────────────────────────────────
    st.markdown('<div class="chart-header">Similarity across document — where plagiarism occurs</div>',
                unsafe_allow_html=True)
    if chunks:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=list(range(len(chunks))),
            y=[round(c["similarity_score"] * 100, 1) for c in chunks],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(
                color=["#f43f5e" if c["similarity_score"] >= threshold
                       else "#fbbf24" if c["similarity_score"] >= threshold * 0.55
                       else "#34d399"
                       for c in chunks],
                size=7,
            ),
            hovertemplate="Chunk %{x} — %{y:.1f}%<extra></extra>",
        ))
        fig_line.add_hline(
            y=threshold * 100,
            line_dash="dash", line_color="#94a3b8", line_width=1,
        )
        fig_line.update_layout(
            xaxis=dict(title="Document position (chunk index)", color="#7a8ba0",
                       gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Similarity %", color="#7a8ba0", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=10, b=50, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, font=dict(family="Inter", color="#e2e8f0"),
        )
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Chunk detail explorer ─────────────────────────────────────────────────
    st.markdown("#### Chunk detail explorer")
    st.caption("Expand any segment to view text content and match parameters.")

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        verdict_filter = st.multiselect(
            "Filter by verdict",
            options=["original", "suspicious", "plagiarised"],
            default=["original", "suspicious", "plagiarised"],
            format_func=str.capitalize,
            key="cmp_verdict_filter",
        )
    with f_col2:
        sim_range = st.slider(
            "Similarity range (%)", 0, 100, (0, 100), key="cmp_sim_range",
        )

    filtered = [
        c for c in chunks
        if dynamic_verdict(c["similarity_score"]) in verdict_filter
        and sim_range[0] <= c["similarity_score"] * 100 <= sim_range[1]
    ]
    st.caption(f"Showing {len(filtered)} of {len(chunks)} segments.")

    # Paginate
    page_size = 15
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1,
                               step=1, key="cmp_page") - 1
    else:
        page = 0

    page_chunks = filtered[page * page_size: (page + 1) * page_size]

    for i, chunk in enumerate(page_chunks):
        row_id = page * page_size + i
        idx     = chunk.get("suspicious_chunk_index", i)
        sim     = round(chunk["similarity_score"] * 100, 1)
        conf    = round(chunk["confidence"] * 100, 1)
        verdict = dynamic_verdict(chunk["similarity_score"])
        badge   = verdict_badge_html(verdict)
        col     = severity_color(chunk["similarity_score"])
        src     = chunk.get("source_filename", "Direct input")
        open_key = f"cmp_block_open_{row_id}"
        is_open = st.session_state.get(open_key, False)

        if st.button(
            f"{'▼' if is_open else '▶'} Block #{idx} · {sim}% match · {src}",
            key=f"cmp_toggle_{row_id}",
        ):
            st.session_state[open_key] = not is_open
            st.rerun()

        if st.session_state.get(open_key, False):
            with st.container(border=True):
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"**Similarity:** <span style='color:{col};font-weight:700'>{sim}%</span>",
                                unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"**Confidence:** {conf}%")
                with mc3:
                    st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
                st.markdown("")

                tl, tr = st.columns(2)
                with tl:
                    st.markdown('<div class="chunk-text-label">Submitted text</div>',
                                unsafe_allow_html=True)
                    highlighted = _highlight_overlap(
                        chunk["suspicious_chunk_text"], chunk["best_match_source_text"]
                    )
                    st.markdown(f'<div class="chunk-text-content">{highlighted}</div>',
                                unsafe_allow_html=True)
                with tr:
                    st.markdown('<div class="chunk-text-label">Best reference match</div>',
                                unsafe_allow_html=True)
                    highlighted_src = _highlight_overlap(
                        chunk["best_match_source_text"], chunk["suspicious_chunk_text"]
                    )
                    st.markdown(f'<div class="chunk-text-content">{highlighted_src}</div>',
                                unsafe_allow_html=True)


def _highlight_overlap(text: str, other: str) -> str:
    """Bold shared words between two strings."""
    other_words = set(other.lower().split())
    result = []
    for word in text.split():
        clean = word.strip(".,;:!?\"'()")
        if clean.lower() in other_words:
            result.append(f"<strong style='color:#fbbf24'>{word}</strong>")
        else:
            result.append(word)
    return " ".join(result)
