import streamlit as st

from components.shared import (
    ALLOWED_EXTENSIONS,
    call_chroma_ingest_api,
    call_delete_api,
    get_knowledge_base_stats,
    get_indexed_documents,
    render_empty_state,
    render_file_chip,
    render_db_status_cards,
)


def render_library_page():
    st.markdown('<div class="page-library">', unsafe_allow_html=True)
    st.markdown('<h2>Index Library</h2>', unsafe_allow_html=True)

    # Live stats
    stats = get_knowledge_base_stats()
    render_db_status_cards(stats)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Upload + index
    st.markdown(
        '<div class="panel-label">Add Reference Document</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Documents indexed here are permanently stored in ChromaDB and used during scanning."
    )

    ingest_file = st.file_uploader(
        "Select document to index",
        type=ALLOWED_EXTENSIONS,
        key="lib_ingest_uploader",
        label_visibility="collapsed",
    )

    if ingest_file:
        render_file_chip(ingest_file, "to be indexed")

    _, btn_col = st.columns([5, 1])
    with btn_col:
        index_clicked = st.button(
            "Index document",
            use_container_width=True,
            key="lib_index_btn",
        )

    if index_clicked:
        if ingest_file is None:
            st.error("Please upload a file to index.")
        else:
            with st.spinner(f"Vectorizing '{ingest_file.name}' into ChromaDB…"):
                try:
                    res = call_chroma_ingest_api(ingest_file)
                    st.success(
                        f"Indexed {ingest_file.name} successfully — "
                        f"{res.get('chunks_stored')} chunks stored "
                        f"(document ID: {res.get('document_id')})."
                    )
                    st.rerun()
                except Exception as e:
                    if "409" in str(e) or "already exists" in str(e).lower():
                        st.warning(
                            f"{ingest_file.name} is already in the database — skipping duplicate."
                        )
                    else:
                        st.error(f"Indexing failed: {e}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Indexed documents
    st.markdown(
        '<div class="panel-label">Indexed Documents</div>',
        unsafe_allow_html=True,
    )

    docs = get_indexed_documents()

    if not docs:
        render_empty_state(
            "Library is empty",
            "No documents indexed yet",
            "Upload your first reference document above to enable scanning.",
        )
    else:
        st.caption(
            f"{len(docs)} document(s) in the knowledge base. "
            "You can delete a document to remove all its chunks."
        )

        # Table header
        h1, h2, h3, h4 = st.columns([4, 1, 1, 1])
        with h1:
            st.markdown("Filename")
        with h2:
            st.markdown("Chunks")
        with h3:
            st.markdown("IDs")
        with h4:
            st.markdown("Action")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        for doc in docs:
            fname = doc.get("filename", "—")
            chunks = doc.get("chunks", 0)
            n_ids = doc.get("documents", 1)
            del_key = f"del_confirm_{fname}"

            c1, c2, c3, c4 = st.columns([4, 1, 1, 1])

            with c1:
                st.markdown(fname)

            with c2:
                st.markdown(str(chunks))

            with c3:
                st.markdown(str(n_ids))

            with c4:
                if st.session_state.get(del_key):
                    confirmed = st.button("Confirm", key=f"del_yes_{fname}")
                    if confirmed:
                        try:
                            call_delete_api(fname)
                            st.success(f"Removed {fname} from database.")
                            st.session_state.pop(del_key, None)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

                    if st.button("Cancel", key=f"del_no_{fname}"):
                        st.session_state.pop(del_key, None)
                        st.rerun()
                else:
                    if st.button("Delete", key=f"del_btn_{fname}"):
                        st.session_state[del_key] = True
                        st.rerun()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
