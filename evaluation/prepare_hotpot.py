"""Create an immutable, deterministic HotpotQA subset without changing official labels."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

SEED = "knowflow-hotpot-external-v1"
ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select_rows(payload, per_type=2):
    if per_type < 1:
        raise ValueError("per_type must be positive")
    rows = payload["rows"]
    selected = []
    for kind in ("bridge", "comparison"):
        group = [entry for entry in rows if entry["row"]["type"] == kind]
        group.sort(key=lambda entry: digest((SEED + entry["row"]["id"]).encode()))
        if len(group) < per_type:
            raise ValueError("Insufficient rows for the predeclared stratified selection")
        selected.extend(group[:per_type])
    if any(entry.get("truncated_cells") for entry in selected):
        raise ValueError("Selected rows contain truncated cells; do not silently replace them")
    return selected


def convert_rows(selected):
    documents, provenance, cases = {}, {}, []
    for entry in selected:
        row = entry["row"]
        context = row["context"]
        by_title = {}
        for title, sentences in zip(context["title"], context["sentences"], strict=True):
            if title in by_title or not sentences or not all(isinstance(s, str) for s in sentences):
                raise ValueError("Invalid context structure")
            body = "".join(sentences)
            filename = "hotpot-" + digest((title + "\0" + body).encode())[:20] + ".md"
            documents[filename] = f"# {title}\n\n{body}\n"
            provenance[filename] = {
                "title": title,
                "wikipedia_page": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                "note": (
                    "Historical paragraph snapshot distributed by HotpotQA; not live Wikipedia."
                ),
            }
            by_title[title] = (filename, sentences)
        facts = []
        pairs = zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True
        )
        for title, sentence_id in dict.fromkeys(pairs):
            filename, sentences = by_title[title]
            if not isinstance(sentence_id, int) or not 0 <= sentence_id < len(sentences):
                raise ValueError("Invalid official supporting sentence index")
            text = sentences[sentence_id].strip()
            if not text:
                raise ValueError("Empty official supporting sentence")
            facts.append(
                {"id": f"support-{len(facts) + 1}", "filename": filename, "contains": text}
            )
        cases.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "question_type": row["type"],
                "should_refuse": False,
                "relevant_evidence": facts,
                "official_level": row["level"],
                "original_row_index": entry["row_idx"],
            }
        )
    return documents, provenance, cases


def prepare(input_path, output_dir, revision=None, per_type=2):
    raw = input_path.read_bytes()
    payload = json.loads(raw)
    if [entry["row_idx"] for entry in payload["rows"]] != list(range(100)):
        raise ValueError("This protocol requires the unchanged first 100 validation rows")
    selected = select_rows(payload, per_type)
    documents, provenance, cases = convert_rows(selected)
    # Never overwrite a previous trial after observing its results.
    output_dir.mkdir(parents=True, exist_ok=False)
    corpus = output_dir / "corpus"
    corpus.mkdir()
    for filename, content in documents.items():
        (corpus / filename).write_text(content, encoding="utf-8")
    dataset = {
        "version": "hotpotqa-validation-fixed4-v1"
        if per_type == 2
        else f"hotpotqa-fixed{2 * per_type}-v1",
        "source": (
            "HotpotQA distractor validation, fixed stratified subset of the first 100 rows. "
            "Official questions, answers and support sentences unchanged. All ten contexts per "
            "question retained and pooled across this subset. Internal RAG metrics, not official "
            "leaderboard EM/F1 or fullwiki evaluation; public data may be in model training."
        ),
        "cases": cases,
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "selected-original.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "document-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths = list(corpus.glob("*.md")) + [
        output_dir / "dataset.json",
        output_dir / "selected-original.json",
    ]
    core = [
        ROOT / "backend/knowflow/app" / (name + ".py")
        for name in (
            "providers",
            "answer_review",
            "graph_index",
            "graph_semantics",
            "retrieval",
            "relevance",
            "quality_metrics",
            "evaluate_quality",
            "api",
        )
    ]
    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_url": "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=0&length=100",
        "repository": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "repository_revision_metadata": revision,
        "raw_response_sha256": digest(raw),
        "selection": {
            "seed": SEED,
            "rule": (
                "Sort SHA256(seed + official id) within each type; "
                f"first {per_type} per type; no answer/score filtering."
            ),
            "per_type": per_type,
            "pool": "validation rows 0..99",
        },
        "official_ids": [row["row"]["id"] for row in selected],
        "corpus_documents": len(documents),
        "license": "CC BY-SA 4.0 (dataset); preserve attribution and share-alike terms.",
        "artifact_sha256": {str(p.relative_to(output_dir)): digest(p.read_bytes()) for p in paths},
        "implementation_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in core},
        "adapter_sha256": digest(Path(__file__).read_bytes()),
        "limitation": (
            "Not a uniform draw from all validation, not a full benchmark, and not proof "
            "of unseen model-training data or complex-document parsing."
        ),
    }
    (output_dir / "frozen-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--per-type", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.input, args.output_dir, args.revision, args.per_type)
    print(
        json.dumps(
            {
                "cases": len(result["official_ids"]),
                "documents": result["corpus_documents"],
                "ids": result["official_ids"],
            }
        )
    )
