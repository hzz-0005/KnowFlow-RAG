"""Controlled distant-reference and duplicate-number fixtures. Run with bundled Python."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def build(name, ambiguous=False):
    path = ROOT / "controlled" / name
    pdf = canvas.Canvas(str(path), pagesize=(595, 842))
    pdf.setTitle("Controlled cross-page references - synthetic facts")
    for page in range(1, 9):
        title = f"{page} 独立附件"
        lines = ["此页是占位附件，不包含设备事实。", "样本仅用于软件回归，不是真实业务资料。"]
        if page == 1:
            title = "1 回归样本说明"
            lines = [
                "虚构设备资料，用于验证远距离文内引用。",
                "核对时必须保留引用链中的条件与适用范围。",
            ]
        elif page == 2:
            title = "2 部署范围"
            lines = ["编号 G1 对应 Atlas 网关。", "部署范围：仅适用于测试环境，生产环境不适用。"]
        elif page == 4:
            title = "4 执行条件"
            lines = ["归档任务的执行范围遵循第2节。", "复核周期：每隔17天。"]
        elif page == 5 and ambiguous:
            title = "2 补充部署范围"
            lines = [
                "本附件使用了重复的章节号。",
                "部署范围：仅适用于隔离环境。不可自动判定其与前一范围的优先级。",
            ]
        elif page == 8:
            title = "8 归档路由"
            lines = ["G1 的归档任务写入 Cedar 索引。", "复核周期与执行范围均见第4节。"]
        pdf.setFont("STSong-Light", 23)
        pdf.drawString(55, 750, title)
        pdf.setFont("STSong-Light", 14)
        for i, line in enumerate(lines):
            pdf.drawString(55, 690 - i * 38, line)
        pdf.setFont("STSong-Light", 10)
        pdf.drawString(55, 45, f"受控回归样本 - 物理页 {page} / 8")
        pdf.showPage()
    pdf.save()
    with pdfium.PdfDocument(path) as saved:
        thumbs = []
        for index in [1, 3, 7]:
            page = saved[index]
            bitmap = page.render(scale=0.6)
            thumbs.append(bitmap.to_pil().convert("RGB"))
            bitmap.close()
            page.close()
        contact = Image.new(
            "RGB", (sum(im.width for im in thumbs), max(im.height for im in thumbs)), "white"
        )
        left = 0
        for im in thumbs:
            contact.paste(im, (left, 0))
            left += im.width
        contact.save(ROOT.parent / "results" / (path.stem + "-contact.png"))
    print(f"Created {name}: 8 pages, duplicate section={ambiguous}")


if __name__ == "__main__":
    build("cross-page-references.pdf")
    build("ambiguous-references.pdf", ambiguous=True)
