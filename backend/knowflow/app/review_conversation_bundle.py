"""Offline CLI: check snapshots and declared annotation witnesses, not answer correctness."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from knowflow.app import conversation_review
from knowflow.app.conversation_review import digest, load_json, validate_annotations


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--require-annotations",
        action="store_true",
        help="Exit 3 if any dimension remains unreviewed; not a quality gate",
    )
    args = parser.parse_args(argv)
    try:
        bundle = load_json(args.bundle)
        annotations = load_json(args.annotations) if args.annotations else bundle["review_template"]
        report = validate_annotations(bundle, annotations)
        try:
            args.out.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError("Output already exists; refusing overwrite") from exc

        def write(name, value):
            (args.out / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )

        write("report.json", report)
        write("annotations.json", annotations)
        (args.out / "summary.md").write_text(
            "# 多轮对话复核包校验\n\n本报告没有准确率或任务成功率。unknown不是0分。\n\n"
            f"截至轮次：{bundle['snapshot']['through_sequence']}；人工标注来源仅为声明：{report['declared_method']}。\n\n"
            "状态："
            + report["annotation_status"]
            + "；独立复核未建立。字面节选通过不证明语义正确。\n\n"
            + "\n".join(
                f"- {r['dimension']}: {r['verdict']}（{r['observation']}）"
                for r in report["criteria"]
            )
            + "\n\n仅包含已保存引用节选，不能据此判整篇文档没有答案或证据完整。\n",
            encoding="utf-8",
        )
        write(
            "audit.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "argv": [sys.executable, *sys.orig_argv[1:]] if argv is None else argv,
                "input_files": [
                    {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (args.bundle, args.annotations)
                    if p is not None
                ],
                "code": [
                    {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (Path(__file__), Path(conversation_review.__file__))
                ],
                "outputs": [
                    {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sorted(args.out.iterdir())
                    if p.is_file()
                ],
                "bundle_sha256": digest(bundle),
                "annotations_sha256": digest(annotations),
                "snapshot_sha256": bundle["snapshot_sha256"],
                "semantic_scoring_performed": False,
            },
        )
        return 3 if args.require_annotations and report["unreviewed_dimensions"] else 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"Review package error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
