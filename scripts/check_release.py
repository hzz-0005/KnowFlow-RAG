"""Check an exported public tree without printing matched sensitive values."""

import argparse
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .public_files import allowed
else:
    from public_files import allowed

RULES = {
    "provider_credential": re.compile(
        rb"\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b"
    ),
    "private_key": re.compile(rb"^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.M),
    "local_home_path": re.compile(rb"[A-Za-z]:[/\\]Users[/\\][A-Za-z0-9_-]+[/\\]"),
}


def scan(root, private_env=None):
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    names = [r["path"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate manifest path")
    secrets = set()
    if private_env is not None and private_env.is_file():
        for line in private_env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and (key.endswith(("API_KEY", "PASSWORD")) or key == "JWT_SECRET"):
                value = value.strip().strip('"').strip("'")
                if len(value) >= 16 and not value.startswith("replace-this"):
                    secrets.add(value.encode())
    findings = []
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            findings.append({"path": path.relative_to(root).as_posix(), "rule": "symlink"})
            continue
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        actual.add(name)
        if name == "SOURCE_MANIFEST.json":
            continue
        if not allowed(name):
            findings.append({"path": name, "rule": "not_allowlisted"})
        if path.stat().st_size >= 90 * 1024 * 1024:
            findings.append({"path": name, "rule": "github_file_size"})
        data = path.read_bytes()
        for secret in secrets:
            if secret in data:
                findings.append({"path": name, "rule": "matches_local_secret"})
        if path.suffix.lower() in {".pdf", ".docx", ".xlsx", ".gz"}:
            continue
        for rule, pattern in RULES.items():
            for match in pattern.finditer(data):
                findings.append(
                    {"path": name, "rule": rule, "line": data[: match.start()].count(b"\n") + 1}
                )
    if actual != set(names) | {"SOURCE_MANIFEST.json"}:
        findings.append({"rule": "manifest_inventory_mismatch"})
    for row in records:
        name = row["path"]
        if not allowed(name) or Path(name).is_absolute() or ".." in Path(name).parts:
            findings.append({"path": name, "rule": "unsafe_manifest_path"})
            continue
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            findings.append({"path": name, "rule": "fingerprint_mismatch"})
    for required in (
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "uv.lock",
        "frontend/pnpm-lock.yaml",
        "scripts/run_tests.py",
        ".github/workflows/ci.yml",
    ):
        if required not in actual:
            findings.append({"path": required, "rule": "required_file_missing"})
    return {
        "schema": "knowflow-release-check-v1",
        "files": len(names),
        "local_secret_values_checked": len(secrets),
        "findings": findings,
        "passed": not findings,
        "scope": (
            "Source/credential policy checks, not an independent security audit "
            "or semantic benchmark"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--private-env", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = scan(args.root.resolve(), args.private_env)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=True, indent=2)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "files": result["files"],
                    "findings": len(result["findings"]),
                }
            )
        )
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Release policy check failed: {type(exc).__name__}\n")


if __name__ == "__main__":
    raise SystemExit(main())
