"""Explicit public-source policy. Local research never enters a release by accident."""

from pathlib import Path

ROOT_FILES = {
    ".gitattributes",
    ".dockerignore",
    ".gitignore",
    ".env.example",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    "alembic.ini",
    "compose.yml",
    "compose.pdf.yml",
    "compose.smoke.yml",
}
CONTROL_DIRS = {
    "evaluation/corpus",
    "evaluation/diagnostics_v1/corpus",
    "evaluation/diagnostics_v2/corpus",
    "evaluation/complex_documents/controlled",
}
CONTROL_FILES = {
    "evaluation/dataset.json",
    "evaluation/quality-template.json",
    "evaluation/relevance_cases.json",
    "evaluation/scoped_answer_cases.json",
    "evaluation/context_routing_v1/cases.json",
    "evaluation/routing_input_v1/cases.json",
    "evaluation/conversation_v1/scenarios.json",
    "evaluation/diagnostics_v1/cases.json",
    "evaluation/diagnostics_v1/corpus-manifest.json",
    "evaluation/diagnostics_v2/cases.json",
    "evaluation/diagnostics_v2/corpus-manifest.json",
    "evaluation/README.md",
}
EVALUATION_SCRIPTS = {
    "evaluation/prepare_hotpot.py",
    "evaluation/diagnostics_v1/run_controls.py",
    "evaluation/diagnostics_v1/prepare_controls.py",
    "evaluation/m2_v1/validate_dataset.py",
    "evaluation/complex_documents/fetch_chunk_tokenizer.py",
    "evaluation/complex_documents/build_reference_fixtures.py",
    "evaluation/complex_documents/build_ocr_fixtures.py",
}
ASSET_FILES = {
    "backend/knowflow/app/assets/README.md",
    "backend/knowflow/app/assets/QWEN-LICENSE.txt",
    "backend/knowflow/app/assets/qwen3-tokenizer.json",
    "backend/knowflow/app/assets/qwen3-tokenizer.json.gz",
}


def allowed(relative):
    p = Path(relative)
    name = p.as_posix()
    if (
        p.is_absolute()
        or ".." in p.parts
        or any(
            x in p.parts
            for x in ("__pycache__", "node_modules", ".git", ".venv", "dist", "results")
        )
    ):
        return False
    if name in ROOT_FILES or name in CONTROL_FILES:
        return True
    if name.startswith("evaluation/unified_example/"):
        return p.suffix in {".json", ".md"}
    if p.parent.as_posix() in CONTROL_DIRS:
        return p.suffix.lower() in {".md", ".pdf", ".docx", ".csv", ".xlsx"}
    if name.startswith(("backend/", "migrations/", "scripts/")):
        return p.suffix in {".py", ".mako"} or name in ASSET_FILES
    if name.startswith("evaluation/"):
        return name in EVALUATION_SCRIPTS
    if name.startswith("docs/"):
        return p.suffix == ".md"
    if name.startswith(".github/"):
        return p.suffix in {".yml", ".yaml", ".md"}
    if name.startswith("frontend/"):
        return p.suffix in {".js", ".mjs", ".css", ".html", ".json", ".yaml"} or p.name in {
            "Dockerfile",
            ".dockerignore",
            "nginx.conf",
        }
    return False


def files(root):
    # Do not walk private data/caches merely to exclude them afterwards.
    for name in sorted(ROOT_FILES | CONTROL_FILES):
        path = root / name
        if path.is_file():
            yield path
    for folder in ("backend", "migrations", "scripts", "docs", ".github", "frontend", "evaluation"):
        import os

        for directory, dirs, names in os.walk(root / folder):
            dirs[:] = sorted(
                d
                for d in dirs
                if d
                not in {
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "dist",
                    "results",
                    "data",
                    "draft-history",
                    "baselines",
                    "sources",
                    "page-previews",
                    "pages",
                    "reserved",
                }
            )
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(root)
                if relative.as_posix() not in ROOT_FILES | CONTROL_FILES and allowed(relative):
                    yield path
