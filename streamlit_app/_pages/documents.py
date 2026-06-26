"""
Documents page — upload files or paste text into the RAG vector store,
and view store statistics.  Mirrors the Gradio "Documents" tab.
"""
import sys, os
import streamlit as st

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for p in (_APP_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read_uploaded(uf) -> tuple[str, str]:
    """Return (text, filename) from a Streamlit UploadedFile."""
    name = uf.name
    raw = uf.read()
    ext = os.path.splitext(name)[1].lower()

    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n\n".join(p.get_text() for p in doc)
            doc.close()
        except ImportError:
            text = raw.decode("utf-8", errors="replace")
    elif ext == ".ipynb":
        import json as _json
        try:
            nb = _json.loads(raw.decode("utf-8", errors="replace"))
            parts = [
                f"[{c['cell_type'].upper()}]\n{''.join(c.get('source', []))}"
                for c in nb.get("cells", [])
                if ''.join(c.get("source", [])).strip()
            ]
            text = "\n\n".join(parts)
        except Exception:
            text = raw.decode("utf-8", errors="replace")
    elif ext == ".docx":
        try:
            from docx import Document
            from io import BytesIO
            doc = Document(BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            text = raw.decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    return text, name


def render() -> None:
    st.title("📄 Documents")
    st.caption("Upload reference material into the RAG vector store used by the Strategist agent.")

    import adaptevolve_core as ae

    col_left, col_right = st.columns(2)

    # ── Left: file upload ─────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Upload a File")
        uf = st.file_uploader(
            "Supported: .py .txt .pdf .ipynb .docx .tex .md",
            type=["py", "txt", "pdf", "ipynb", "docx", "tex", "md"],
            key="rag_file_upload",
        )
        if st.button("Upload File to RAG", use_container_width=True, type="primary"):
            if uf:
                text, fname = _read_uploaded(uf)
                ae.session.upload_document(text, source=fname)
                st.success(f"Ingested **{fname}** — {len(text):,} chars")
            else:
                st.warning("Please select a file first.")

    # ── Right: paste text ─────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Paste Document Content")
        doc_text = st.text_area("Document text", height=200, key="rag_paste_text")
        doc_source = st.text_input("Source name (optional)", placeholder="e.g. paper_notes.txt")
        if st.button("Upload Text to RAG", use_container_width=True):
            if doc_text.strip():
                source = doc_source.strip() or "pasted"
                ae.session.upload_document(doc_text, source=source)
                st.success(f"Ingested pasted text ({len(doc_text):,} chars) as **{source}**")
            else:
                st.warning("Please paste some text first.")

    # ── RAG stats ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Vector Store Statistics")
    if st.button("Refresh Stats"):
        pass  # just reruns the section below
    stats = ae.rag_system.vector_store.get_stats()
    col_a, col_b = st.columns(2)
    col_a.metric("Documents / Chunks", stats.get("total_chunks", 0))
    col_b.metric("Total Documents", stats.get("total_documents", 0))
    st.json(stats)
