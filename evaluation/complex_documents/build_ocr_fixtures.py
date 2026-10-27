"""Controlled OCR fixtures, not real operational instructions. Run with bundled Python."""

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "controlled"
OUTPUT.mkdir(exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def scan_image():
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 32)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 43)
    draw.text((95, 110), "扫描页：备用电源检查记录", font=title, fill="black")
    lines = [
        "测试样本，非真实设备操作规范。",
        "",
        "记录编号：BAT-042",
        "设备：备用电源 B",
        "检查日期：2026-10-20",
        "复检周期：每隔 17 天",
        "容量低于 83% 时需安排维护。",
        "",
        "以上关键内容只存在于图片中。",
    ]
    for index, line in enumerate(lines):
        draw.text((95, 225 + index * 64), line, font=font, fill="black")
    return image


def native_page(pdf):
    pdf.setFont("STSong-Light", 22)
    pdf.drawString(48, 770, "复杂 PDF 回归样本：设备巡检")
    pdf.setFont("STSong-Light", 13)
    for index, line in enumerate(
        [
            "测试样本，非真实设备操作规范。",
            "本页包含可选择的原生文字。",
            "主泵 A 的检查周期为每隔 11 天。",
            "备用电源 B 的规则记录在下一页扫描记录中。",
            "核对时必须保留记录编号、周期和数值单位。",
        ]
    ):
        pdf.drawString(48, 715 - index * 31, line)
    pdf.showPage()


scan = scan_image()
for name, include_native in [("mixed-ocr.pdf", True), ("scan-only.pdf", False)]:
    result = canvas.Canvas(str(OUTPUT / name), pagesize=(595, 842))
    result.setTitle("Controlled OCR regression fixture")
    if include_native:
        native_page(result)
    result.drawImage(ImageReader(scan), 0, 0, width=595, height=842)
    result.showPage()
    result.save()

# Preview both physical pages from the saved mixed PDF, rather than the source image.
import pypdfium2 as pdfium  # noqa: E402

with pdfium.PdfDocument(OUTPUT / "mixed-ocr.pdf") as saved:
    thumbnails = []
    for page in saved:
        thumbnails.append(page.render(scale=0.8).to_pil().convert("RGB"))
    contact = Image.new(
        "RGB",
        (sum(img.width for img in thumbnails), max(img.height for img in thumbnails)),
        "#dce5ec",
    )
    left = 0
    for thumb in thumbnails:
        contact.paste(thumb, (left, 0))
        left += thumb.width
    buffer = io.BytesIO()
    contact.save(buffer, format="PNG")
    (ROOT.parent / "results" / "ocr-fixture-contact.png").write_bytes(buffer.getvalue())
print("Created mixed/native+scan and scan-only fixtures; source-page preview saved.")
