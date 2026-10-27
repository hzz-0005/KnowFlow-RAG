"""Authored numeric/annotation controls; no bank report or private gold required."""

import json
from copy import deepcopy

from knowflow.app.evaluation_replay.common import digest
from knowflow.app.structured_parsing import render_pdf_page
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def build_control(root):
    base = root / "prep"
    base.mkdir()
    pdf = root / "control.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    lines = [
        "Fictional Control Bank. Synthetic software tests, not financial data.",
        "Currency CNY. Amounts in millions. Current and previous fiscal years.",
        "Revenue 100. Profit current 80, previous 60. Difference 20.",
        "ROE current 8 percent, previous 10 percent. Change -2 percentage points.",
    ]
    stream = DecodedStreamObject()
    stream.set_data(
        (
            "BT /F1 10 Tf 40 740 Td 16 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        ).encode("ascii")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    with pdf.open("wb") as output:
        writer.write(output)
    text = PdfReader(pdf).pages[0].extract_text()
    snapshot = root / "page.txt"
    snapshot.write_text(text, encoding="utf-8")
    preview = root / "page.png"
    preview.write_bytes(render_pdf_page(pdf, 1))

    def save(name, value):
        (base / name).write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")

    did = "synthetic-control-bank"
    assignments = [
        {
            "document_id": did,
            "group_id": "control-bank",
            "split": "development",
            "content_inspection_allowed": True,
        }
    ]
    save("split-assignment.json", {"schema": "knowflow.m2.split.v1", "assignments": assignments})
    save(
        "corpus-manifest.json",
        {
            "schema": "knowflow.m2.corpus.v1",
            "split_manifest_sha256": digest(base / "split-assignment.json"),
            "documents": [
                {
                    **assignments[0],
                    "path": "control.pdf",
                    "sha256": digest(pdf),
                    "format": "pdf",
                    "physical_pages": 1,
                    "download_status": "completed",
                    "origin": "authored synthetic test fixture; never downloaded",
                }
            ],
        },
    )
    evidence = []
    for index, line in enumerate(lines):
        start = text.index(line)
        evidence.append(
            {
                "id": f"e{index}",
                "document_id": did,
                "document_sha256": digest(pdf),
                "physical_page": 1,
                "char_start": start,
                "char_end": start + len(line),
                "quote": line,
                "text_snapshot": "page.txt",
                "text_snapshot_sha256": digest(snapshot),
                "page_preview": "page.png",
                "page_preview_sha256": digest(preview),
                "review_status": "draft",
            }
        )

    def number(nid, value, metric="profit", unit="million", period="FY2024"):
        return {
            "id": nid,
            "value": value,
            "entity": "FictionalControlBank",
            "metric": metric,
            "unit": unit,
            "period": period,
            "basis": "consolidated",
            "currency": "CNY",
        }

    def case(cid, numbers, group=None):
        return {
            "id": cid,
            "split": "development",
            "question_type": "synthetic_contract",
            "question": f"Software control question {cid}",
            "scope_document_ids": [did],
            "answerability": "answerable",
            "annotation_status": "draft",
            "expected_answer": "Synthetic answer; not semantic gold.",
            "required_claims": [
                {
                    "id": "fact",
                    "text": "synthetic fact",
                    "evidence_alternatives": [{"all_of": ["e1", "e2"]}],
                }
            ],
            "numeric_assertions": numbers,
            "equivalence_group": group or cid,
        }

    first = case("control-revenue", [number("revenue", "100", "revenue")])
    equivalent = deepcopy(first)
    equivalent.update(id="control-revenue-paraphrase", question="A second wording of the same fact")
    tasks = [first, equivalent]
    for cid, values, unit, metric in [
        ("control-profit-difference", ["80", "60", "20"], "million", "profit"),
        ("control-roe-change", ["8", "10", "-2"], "percent", "roe"),
    ]:
        numbers = [
            number(nid, v, metric, unit)
            for nid, v in zip(["a", "b", "result"], values, strict=True)
        ]
        if unit == "percent":
            numbers[-1]["unit"] = "percentage_points"
        row = case(cid, numbers)
        row["calculation"] = {
            "operator": "subtract",
            "operand_ids": ["a", "b"],
            "result_id": "result",
        }
        tasks.append(row)
    tasks += [case(f"control-extra-{i}", []) for i in range(3)]
    for name, rows in [("evidence-draft.jsonl", evidence), ("tasks-draft.jsonl", tasks)]:
        (base / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    save(
        "baseline-protocol.json",
        {
            "schema": "knowflow.m2.protocol.v1",
            "corpus_manifest_sha256": digest(base / "corpus-manifest.json"),
            "execution_enabled": False,
            "m1_external_review_status": "pending",
            "runtime_snapshot": None,
        },
    )
    return root, base
