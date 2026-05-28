import streamlit as st

from components.shared import (
    ALLOWED_EXTENSIONS,
    call_chroma_ingest_api,
    get_knowledge_base_stats,
    render_empty_state,
    render_file_chip,
)


def render_library_page():
    st.markdown('<div class="panel-label">Vector database status</div>',
                unsafe_allow_html=True)

    stats = get_knowledge_base_stats()

    if stats:
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Collection", stats.get("collection_name", "—"))
        with s2:
            st.metric("Indexed chunks", stats.get("total_chunks", 0))
        with s3:
            st.metric("Status", "Connected ✓")
    else:
        st.warning(
            "Cannot reach the backend — make sure the API server is running on port 8000.",
            icon="⚠️",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Add reference document</div>',
                unsafe_allow_html=True)
    st.caption(
        "Documents indexed here are permanently stored in ChromaDB and will be "
        "searched during every library scan."
    )

    ingest_file = st.file_uploader(
        "Select document to index", type=ALLOWED_EXTENSIONS, key="lib_ingest_uploader",
        label_visibility="collapsed",
    )

    if ingest_file:
        render_file_chip(ingest_file, "to be indexed")

    _, btn_col = st.columns([5, 1])
    with btn_col:
        index_clicked = st.button("Index document", use_container_width=True, key="lib_index_btn")

    if index_clicked:
        if ingest_file is None:
            st.error("Please upload a file to index.")
        else:
            with st.spinner(f"Vectorizing '{ingest_file.name}' into ChromaDB…"):
                try:
                    res = call_chroma_ingest_api(ingest_file)
                    st.success(
                        f"Indexed **{ingest_file.name}** successfully — "
                        f"{res.get('chunks_stored')} chunks stored "
                        f"(document ID: `{res.get('document_id')}`)."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if not stats or stats.get("total_chunks", 0) == 0:
        render_empty_state(
            "📚", "Library is empty",
            "Index your first reference document above to enable library scanning.",
        )
