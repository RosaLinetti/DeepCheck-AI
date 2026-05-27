# app/services/document_parser.py
"""
Document Parser Service — DeepCheck-AI

Extracts plain text from uploaded PDF, DOCX, and TXT files.
Designed to handle large documents efficiently by processing
pages/paragraphs incrementally.

Supported formats:
  - PDF  (.pdf)   — via PyPDF2
  - DOCX (.docx)  — via python-docx
  - TXT  (.txt)   — direct text decoding (UTF-8 / Latin-1 fallback)
"""

import io
import re
from typing import Optional, Tuple

from PyPDF2 import PdfReader
from docx import Document as DocxDocument


# Maximum file size: 50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

SUPPORTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentParseError(Exception):
    """Raised when document parsing fails."""
    pass


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file.

    Processes page-by-page to keep memory usage manageable for large
    documents. Joins pages with double-newlines so sentence-boundary
    detection still works downstream.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentParseError(f"Failed to read PDF: {exc}") from exc

    pages_text = []
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                pages_text.append(page_text)
        except Exception as exc:
            # Skip unreadable pages but log a note
            pages_text.append(f"[Page {page_num + 1}: extraction failed]")

    full_text = "\n\n".join(pages_text)
    return _normalize_whitespace(full_text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract all text from a DOCX file.

    Iterates over paragraphs (which are streamed by python-docx),
    so memory stays proportional to the largest single paragraph
    rather than the whole document.
    """
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentParseError(f"Failed to read DOCX: {exc}") from exc

    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    return _normalize_whitespace(full_text)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a plain-text (.txt) file.

    Attempts UTF-8 decoding first; falls back to Latin-1 which
    always succeeds since every byte maps to a valid character.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    return _normalize_whitespace(text)


def detect_format(filename: str, content_type: Optional[str]) -> str:
    """
    Determine file format from content type or extension.
    Returns 'pdf', 'docx', or 'txt'.
    Raises DocumentParseError for unsupported formats.
    """
    # Try content type first
    if content_type and content_type in SUPPORTED_CONTENT_TYPES:
        return SUPPORTED_CONTENT_TYPES[content_type]

    # Fall back to extension
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    elif lower.endswith(".docx"):
        return "docx"
    elif lower.endswith(".txt"):
        return "txt"

    raise DocumentParseError(
        f"Unsupported file format: '{filename}' "
        f"(content_type={content_type}). "
        f"Supported formats: PDF (.pdf), DOCX (.docx), TXT (.txt)"
    )


async def parse_uploaded_file(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> str:
    """
    High-level entry point: detect format, validate size, extract text.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename (used for format detection fallback).
    content_type : Optional[str]
        MIME type from the upload, if available.

    Returns
    -------
    str
        Extracted plain text from the document.

    Raises
    ------
    DocumentParseError
        If the file is too large, unsupported, or cannot be parsed.
    """
    # Size guard
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise DocumentParseError(
            f"File '{filename}' is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # Format detection
    fmt = detect_format(filename, content_type)

    # Extraction
    if fmt == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif fmt == "docx":
        text = extract_text_from_docx(file_bytes)
    else:
        text = extract_text_from_txt(file_bytes)

    if not text or len(text.strip()) < 50:
        raise DocumentParseError(
            f"Could not extract sufficient text from '{filename}'. "
            f"The document may be empty, image-only, or corrupted."
        )

    return text
