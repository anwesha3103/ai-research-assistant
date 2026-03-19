# ============================================================
# components/uploader.py — File Upload UI Component
# ============================================================
# Separating UI components into their own files keeps app.py
# clean and makes each piece independently readable/testable.
# This is standard practice in production Streamlit apps.

import streamlit as st
from utils.file_validator import validate_files
from utils.logger import get_logger

logger = get_logger()


def render_uploader():
    """
    Renders the file upload widget and validates uploaded files.

    Returns:
      valid_files  → list of validated UploadedFile objects
                     ready to be passed to the RAG pipeline
    """
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Max 20MB per file. Multiple files supported.",
    )

    if not uploaded_files:
        return []

    # ── Validate all uploaded files ──────────────────────────
    valid_files, errors = validate_files(uploaded_files)

    # Show errors for invalid files
    for error in errors:
        st.error(error)

    # Show success count
    if valid_files:
        st.success(f"✅ {len(valid_files)} file(s) ready to index")

    return valid_files