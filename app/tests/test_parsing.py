import io
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter
from rag_app.parsing import (
    MAX_UPLOAD_BYTES,
    UnparseableDocument,
    UnsupportedFormat,
    detect_format,
    parse_document,
    split_pages,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = Path(__file__).parents[1] / "eval" / "corpus.txt"


def texts(pages) -> list[str]:
    return [p.text for p in pages]


# --- split_pages is pinned: content IDs and the eval baselines depend on it ---


def test_split_pages_rule():
    assert split_pages("a\n\n\n\n  b  \n\nc\nd\n\n") == ["a", "b", "c\nd"]
    assert split_pages("") == [] and split_pages("\n\n  \n\n") == []


def test_split_pages_still_yields_the_eval_corpus_paragraphs():
    assert len(split_pages(CORPUS.read_text())) == 90


# --- dispatch ---


@pytest.mark.parametrize(
    ("name", "ctype", "fmt"),
    [
        ("a.TXT", None, "text"),
        ("a.md", None, "text"),
        ("a.html", None, "html"),
        ("a.htm", "application/octet-stream", "html"),
        ("a.pdf", None, "pdf"),
        ("a.docx", None, "docx"),
        ("noext", "text/plain; charset=utf-8", "text"),
        ("noext", "application/pdf", "pdf"),
    ],
)
def test_detect_format(name, ctype, fmt):
    assert detect_format(name, ctype) == fmt


def test_extension_wins_over_content_type():
    assert detect_format("a.pdf", "text/plain") == "pdf"


@pytest.mark.parametrize(("name", "ctype"), [("bin.dat", "application/octet-stream"), ("x", None)])
def test_unknown_types_are_unsupported(name, ctype):
    with pytest.raises(UnsupportedFormat):
        detect_format(name, ctype)


# --- text / markdown ---


def test_text_and_markdown_split_on_blank_lines_with_chunk_idx():
    pages = parse_document("n.md", None, b"# Title\n\nfirst para\n\nsecond para\n")
    assert texts(pages) == ["# Title", "first para", "second para"]
    assert [p.chunk_idx for p in pages] == [0, 1, 2]
    assert all(p.page_nr is None for p in pages)


def test_empty_text_file_is_zero_pages_not_an_error():
    assert parse_document("e.txt", None, b"") == []


def test_non_utf8_text_is_unsupported():
    with pytest.raises(UnsupportedFormat):
        parse_document("a.txt", None, b"\xff\xfe\x00bad")


# --- html ---


def test_html_extracts_visible_text_by_block():
    html = (
        b"<html><head><title>skip me</title><style>p{}</style></head><body>"
        b"<h1>Heading</h1><p>First   paragraph\n with  wrap.</p>"
        b"<script>var x = 1;</script><ul><li>one</li><li>two</li></ul>"
        b"<p>Line<br>break &amp; entity</p></body></html>"
    )
    assert texts(parse_document("p.html", None, html)) == [
        "Heading",
        "First paragraph with wrap.",
        "one",
        "two",
        "Line break & entity",
    ]


def test_html_without_text_is_unparseable():
    with pytest.raises(UnparseableDocument):
        parse_document("p.html", None, b"<html><body><script>1</script></body></html>")


# --- pdf (fixture written by reportlab, not by the parsing library) ---


def test_pdf_pages_carry_real_page_numbers():
    pages = parse_document("two_page.pdf", None, (FIXTURES / "two_page.pdf").read_bytes())
    assert [p.page_nr for p in pages] == [1, 2]
    assert [p.chunk_idx for p in pages] == [0, 1]
    assert "espresso descaling" in pages[0].text and "Descale every three months" in pages[0].text
    assert "sourdough starters" in pages[1].text


def test_corrupt_pdf_is_unparseable():
    with pytest.raises(UnparseableDocument):
        parse_document("x.pdf", None, b"%PDF-1.4 definitely not a pdf")


def test_scanned_style_pdf_with_no_text_is_unparseable():
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(UnparseableDocument, match="No extractable text"):
        parse_document("scan.pdf", None, buf.getvalue())


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
def test_pdf_with_owner_password_only_still_parses(algorithm):
    # "Permissions-only" PDFs: owner password set, blank user password. Common in the wild.
    writer = PdfWriter(clone_from=str(FIXTURES / "two_page.pdf"))
    writer.encrypt(user_password="", owner_password="owner", algorithm=algorithm)
    buf = io.BytesIO()
    writer.write(buf)
    pages = parse_document("perm.pdf", None, buf.getvalue())
    assert [(p.chunk_idx, p.page_nr) for p in pages] == [(0, 1), (1, 2)]


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-256"])
def test_password_protected_pdf_is_unparseable(algorithm):
    writer = PdfWriter(clone_from=str(FIXTURES / "two_page.pdf"))
    writer.encrypt("secret", algorithm=algorithm)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(UnparseableDocument, match="password"):
        parse_document("locked.pdf", None, buf.getvalue())


# --- docx ---


def _docx(*paragraphs: str) -> bytes:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_docx_body_paragraphs_become_pages():
    pages = parse_document("d.docx", None, _docx("Intro text", "", "Second paragraph"))
    assert texts(pages) == ["Intro text", "Second paragraph"]
    assert all(p.page_nr is None for p in pages)


def test_docx_written_by_macos_textutil():
    # Independent of python-docx: produced with `textutil -convert docx` from a two-paragraph file.
    pages = parse_document("textutil.docx", None, (FIXTURES / "textutil.docx").read_bytes())
    assert texts(pages) == ["Independent intro paragraph", "Second paragraph from textutil"]


def test_corrupt_docx_is_unparseable():
    with pytest.raises(UnparseableDocument):
        parse_document("d.docx", None, b"PK not really a zip")


def test_empty_docx_is_unparseable():
    with pytest.raises(UnparseableDocument):
        parse_document("d.docx", None, _docx())


def test_upload_limit_constant_is_ten_mib():
    assert MAX_UPLOAD_BYTES == 10 * 1024 * 1024
