"""Turn uploaded files into paragraph "pages" ready for ingestion.

Every parser emits text blocks (optionally tagged with a real page number); each block then goes
through the same `split_pages` rule, so a given paragraph always produces the same content ID
regardless of file type. `split_pages` is deliberately unchanged: content IDs and the recorded
eval baselines depend on it.

Supported: plain text / Markdown (read as text), HTML (visible text via the stdlib parser),
PDF (text layer only) and DOCX (body paragraphs only). There is no OCR, so scanned PDFs fail
with 422. A block with no blank lines becomes one chunk, which can exceed the embedding model's
input limit; real chunking is a separate piece of work.
"""

import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Splits one parsed block of text into chunks. None means the default blank-line split.
Chunker = Callable[[str], list[str]]

_TEXT_EXTS = {".txt", ".text", ".md", ".markdown"}
_HTML_EXTS = {".html", ".htm"}
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIME_FORMATS = {
    "text/plain": "text",
    "text/markdown": "text",
    "text/html": "html",
    "application/pdf": "pdf",
    _DOCX_MIME: "docx",
}


class UnsupportedFormat(Exception):
    """The file type is not one we can parse (HTTP 415)."""


class UnparseableDocument(Exception):
    """The file is a supported type but could not be read, or has no text (HTTP 422)."""


@dataclass(frozen=True)
class Page:
    text: str
    chunk_idx: int  # position among all chunks of the document, before any dedup
    page_nr: int | None = None  # real page number (1-based); set for PDFs only


def split_pages(text: str) -> list[str]:
    """Split text into paragraph "pages": blank-line separated, stripped, empties dropped."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def detect_format(filename: str, content_type: str | None) -> str:
    """Pick a parser from the stored file name's extension, else the declared content type."""
    ext = PurePosixPath(filename).suffix.lower()
    if ext in _TEXT_EXTS:
        return "text"
    if ext in _HTML_EXTS:
        return "html"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _MIME_FORMATS:
        return _MIME_FORMATS[mime]
    raise UnsupportedFormat(
        "Unsupported file type; supported: .txt, .md, .html, .pdf, .docx (UTF-8 text only)"
    )


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise UnsupportedFormat("Only UTF-8 text files are supported") from None


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "head"}
    _BLOCK = {
        "address", "article", "aside", "blockquote", "dd", "details", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
        "td", "th", "tr", "ul",
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "br":
            self.parts.append(" ")
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(re.sub(r"\s+", " ", data))


def _html_blocks(raw: bytes) -> list[tuple[str, int | None]]:
    extractor = _TextExtractor()
    extractor.feed(_decode(raw))
    extractor.close()
    return [("".join(extractor.parts), None)]


def _pdf_blocks(raw: bytes) -> list[tuple[str, int | None]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnparseableDocument("PDF is password-protected")
        return [(page.extract_text() or "", n) for n, page in enumerate(reader.pages, start=1)]
    except UnparseableDocument:
        raise
    except Exception as exc:  # pypdf raises many types on malformed input
        raise UnparseableDocument("Could not read PDF") from exc


def _docx_blocks(raw: bytes) -> list[tuple[str, int | None]]:
    from docx import Document

    try:
        doc = Document(io.BytesIO(raw))
        return [("\n\n".join(p.text for p in doc.paragraphs), None)]
    except Exception as exc:  # bad zip / missing parts / not a docx
        raise UnparseableDocument("Could not read DOCX") from exc


def parse_document(
    filename: str, content_type: str | None, raw: bytes, chunker: Chunker | None = None
) -> list[Page]:
    """Parse bytes into numbered pages. Synchronous and CPU-bound: call from a thread.

    ``chunker`` replaces the blank-line split, applied per parsed block (so a PDF page keeps its
    ``page_nr`` and ``chunk_idx`` still counts across the whole document).
    """
    fmt = detect_format(filename, content_type)
    blocks: list[tuple[str, int | None]]
    if fmt == "text":
        blocks = [(_decode(raw), None)]
    elif fmt == "html":
        blocks = _html_blocks(raw)
    elif fmt == "pdf":
        blocks = _pdf_blocks(raw)
    else:
        blocks = _docx_blocks(raw)

    pages: list[Page] = []
    for text, page_nr in blocks:
        for chunk in chunker(text) if chunker else split_pages(text):
            pages.append(Page(text=chunk, chunk_idx=len(pages), page_nr=page_nr))
    if not pages and fmt != "text":
        raise UnparseableDocument("No extractable text (scanned or image-only document?)")
    return pages


async def read_pages(file: UploadFile, filename: str, chunker: Chunker | None = None) -> list[Page]:
    """Read an upload and parse it, mapping failures to 413 / 415 / 422."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 10 MiB)")
    try:
        return await run_in_threadpool(parse_document, filename, file.content_type, raw, chunker)
    except UnsupportedFormat as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from None
    except UnparseableDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
