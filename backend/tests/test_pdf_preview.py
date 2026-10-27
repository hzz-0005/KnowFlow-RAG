from io import BytesIO
from pathlib import Path

from knowflow.app.structured_parsing import render_pdf_page
from PIL import Image


def test_runtime_preview_keeps_ink_for_an_unembedded_cjk_sentence():
    # This region contains Chinese plus a single '4'. Without runtime CJK fonts,
    # PDFium rendered only that digit while extracted text still looked correct.
    fixture = (
        Path(__file__).resolve().parents[2]
        / "evaluation/complex_documents/controlled/cross-page-references.pdf"
    )
    image = Image.open(BytesIO(render_pdf_page(fixture, 8))).convert("L")
    sentence = image.crop((80, 260, 640, 300))
    ink = sum(count for value, count in enumerate(sentence.histogram()) if value < 180)
    assert ink > 500, "Chinese source-page text vanished; check runtime fallback fonts"
