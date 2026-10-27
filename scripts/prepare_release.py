"""Build an immutable allowlisted source tree and archive, never copy live data."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

if __package__:
    from .public_files import files
else:
    from public_files import files


def build(root, out):
    root, out = root.resolve(), out.resolve()
    if out == root or any(
        out.is_relative_to(root / name)
        for name in (
            "backend",
            "frontend",
            "scripts",
            "docs",
            "evaluation",
            "migrations",
            ".github",
        )
    ):
        raise ValueError("Output must be outside exported source folders")
    out.mkdir(parents=True, exist_ok=False)
    tree = out / "knowflow-rag"
    tree.mkdir()
    rows = []
    for source in files(root):
        if source.is_symlink():
            raise ValueError("Symlinks are not accepted in public sources")
        relative = source.relative_to(root)
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if source.suffix == ".md":
            text = data.decode("utf-8")
            # Private evidence remains local; remove dangling private-link targets only.
            text = re.sub(
                r"\[([^\]]+)\]\((?:[^)]*/)?(?:review|总结)/[^)]*\)",
                r"\1（本地研究记录，未随公开包分发）",
                text,
            )
            data = text.encode("utf-8")
        target.write_bytes(data)
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "original_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": len(data),
            }
        )
    manifest = {
        "schema": "knowflow-public-source-v1",
        "scope": "research-software-source-not-independent-quality-benchmark",
        "files": sorted(rows, key=lambda r: r["path"]),
    }
    (tree / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    archive = out / "knowflow-rag-source.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    path.relative_to(out).as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                zipped.writestr(info, path.read_bytes())
    (out / "SHA256SUMS").write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n",
        encoding="ascii",
    )
    return {"files": len(rows), "archive": str(archive), "bytes": archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                build(Path(__file__).resolve().parents[1], args.out.resolve()), ensure_ascii=True
            )
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Source package failed: {exc}\n")


if __name__ == "__main__":
    main()
