"""Uploaded files (Word, PDF, images) as newsletter sources (no network)."""

import io
import zipfile

import pytest

from src.common import uploads as up

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx() -> bytes:
    doc = f"""<w:document {W}><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Board memo</w:t></w:r></w:p>
<w:p><w:r><w:t>Delay is </w:t></w:r><w:r><w:t>8 years.</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>US</w:t></w:r></w:p></w:tc>
<w:tc><w:p><w:r><w:t>8.1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", doc)
    return buf.getvalue()


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(pdf.tell())
        pdf.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    xref = pdf.tell()
    pdf.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offsets:
        pdf.write(b"%010d 00000 n \n" % off)
    pdf.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objs) + 1, xref)
    )
    return pdf.getvalue()


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "state_path", lambda name: str(tmp_path / name))
    return tmp_path


def test_docx_and_pdf_text(state):
    r = up.read_upload("memo.docx", _docx())
    assert r["kind"] == "docx" and r["title"] == "Board memo"
    assert r["text"] == "# Board memo\nDelay is 8 years.\nUS | 8.1"
    p = up.read_upload("pipeline.pdf", _pdf("Phase 2 data for compound X in 2027."))
    assert p["kind"] == "pdf" and p["method"] == "text" and p["pages"] == 1
    assert "compound X in 2027" in p["text"] and p["title"] == "pipeline"


def test_scanned_pdf_and_image_use_ai(state, monkeypatch):
    seen = []

    def fake_ai(block, instruction, data_uri=""):
        seen.append(block["type"])
        return "Transcribed by AI: values 8.1, 7.5"

    monkeypatch.setattr(up, "_ai_read", fake_ai)
    r = up.read_upload("scan.pdf", _pdf(""))
    assert r["method"] == "ai" and r["note"] == "Scanned PDF read by AI"
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (3000, 1000), "white").save(buf, "PNG")
    img = up.read_upload("chart.png", buf.getvalue())
    assert img["kind"] == "image" and img["text"].startswith("Transcribed by AI")
    assert seen == ["document", "image"]
    # cached by content: no second AI call
    up.read_upload("chart-copy.png", buf.getvalue())
    assert seen == ["document", "image"]


def test_upload_errors(state):
    with pytest.raises(up.UploadError, match="unsupported file type"):
        up.read_upload("notes.txt", b"hi")
    with pytest.raises(up.UploadError, match="larger than"):
        up.read_upload("big.pdf", b"0" * (up.MAX_BYTES + 1))
    with pytest.raises(up.UploadError, match="couldn't be read"):
        up.read_upload("broken.docx", b"not a zip")


def test_uploads_reach_prompt_and_chat(state):
    from src.ui.newsletter_from_briefing import (
        _article_block,
        citation_text,
        sources_block,
    )

    a = up.as_source(up.read_upload("memo.docx", _docx()))
    assert a["origin"] == "upload" and a["key"].startswith("upload_")
    block = _article_block(1, a)
    assert "Document text (uploaded by the editor: memo.docx)" in block
    assert "      US | 8.1" in block
    assert citation_text(a) == "memo.docx (document provided by the editor)."
    src = [
        {
            "role": "reference",
            "title": "Board memo",
            "source": "",
            "date": "",
            "url": "",
            "summary": "x",
            "full_text": "US | 8.1",
            "full_text_license": "uploaded by the editor",
        }
    ]
    assert "Document text (uploaded by the editor):" in sources_block(src)


def test_uploads_skip_full_text_lookup(monkeypatch):
    from src.common import fulltext as ft

    monkeypatch.setattr(ft, "lookup", lambda a: (_ for _ in ()).throw(AssertionError))
    assert ft.availability({"origin": "upload", "title": "x" * 60}) == ""
    assert ft.enrich_articles([{"origin": "upload", "snippet": "s"}], [100]) == 0
