# ============================================================
# utils/file_validator.py — File Validation
# ============================================================
# Before we process any uploaded file, we validate it.
#
# 📖 WHY VALIDATE?
# ─────────────────
# Without validation, users could upload:
#   • Unsupported formats (.exe, .mp4) → crashes the loader
#   • Massive files (500MB PDF) → kills memory / API costs
#   • Empty files → confusing errors deep in the pipeline
#
# Validating early = failing fast with a clear message
# instead of a cryptic error 5 steps later.

from typing import Tuple, List
import os

from utils.logger import get_logger
import config

logger = get_logger()


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate a single Streamlit UploadedFile object.

    Returns:
        (True, "")           → file is valid, proceed
        (False, "reason")    → file is invalid, show reason to user
    """

    # ── Check 1: File extension ──────────────────────────────
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        msg = (f"❌ '{filename}' is not supported. "
               f"Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}")
        logger.warning(f"Invalid file type: {ext} for file {filename}")
        return False, msg

    # ── Check 2: File size ───────────────────────────────────
    # uploaded_file.size gives size in bytes
    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > config.MAX_FILE_SIZE_MB:
        msg = (f"❌ '{filename}' is {size_mb:.1f}MB — "
               f"max allowed is {config.MAX_FILE_SIZE_MB}MB")
        logger.warning(f"File too large: {size_mb:.1f}MB for file {filename}")
        return False, msg

    # ── Check 3: Empty file ──────────────────────────────────
    if uploaded_file.size == 0:
        msg = f"❌ '{filename}' is empty."
        logger.warning(f"Empty file uploaded: {filename}")
        return False, msg

    logger.info(f"✅ File validated: {filename} ({size_mb:.2f}MB)")
    return True, ""


def validate_files(uploaded_files) -> Tuple[List, List[str]]:
    """
    Validate a list of uploaded files.

    Returns:
        valid_files   → list of files that passed validation
        errors        → list of error messages for invalid files
    """
    valid_files = []
    errors = []

    for f in uploaded_files:
        is_valid, error_msg = validate_file(f)
        if is_valid:
            valid_files.append(f)
        else:
            errors.append(error_msg)

    logger.info(f"Validation complete: {len(valid_files)} valid, {len(errors)} invalid")
    return valid_files, errors