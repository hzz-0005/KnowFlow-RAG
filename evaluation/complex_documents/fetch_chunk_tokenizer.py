"""Fetch only a pinned public tokenizer (no model weights), for offline runtime use."""

import gzip
import hashlib
import json
from pathlib import Path

import httpx

REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
URL = f"https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/{REVISION}/tokenizer.json"
DEST = Path(__file__).resolve().parents[2] / "backend/knowflow/app/assets"


def main():
    response = httpx.get(URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    data = response.content
    assert isinstance(json.loads(data)["model"], dict)
    DEST.mkdir(parents=True, exist_ok=True)
    license_response = httpx.get(
        "https://www.apache.org/licenses/LICENSE-2.0.txt",
        follow_redirects=True,
        timeout=60,
    )
    license_response.raise_for_status()
    if "Apache License" not in license_response.text:
        raise ValueError("Unexpected upstream license")
    (DEST / "QWEN-LICENSE.txt").write_text(license_response.text, encoding="utf-8")
    (DEST / "qwen3-tokenizer.json.gz").write_bytes(gzip.compress(data, mtime=0))
    manifest = {
        "source": URL,
        "revision": REVISION,
        "sha256": hashlib.sha256(data).hexdigest(),
        "license": "Apache-2.0",
        "source_model": "Qwen/Qwen3-Embedding-0.6B",
        "mode": "qwen3_family_estimator",
        "note": "公开 Qwen3 tokenizer；未证明与托管 qwen3.7 embedding 分词完全一致。",
    }
    (DEST / "qwen3-tokenizer.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({"bytes": len(data), "sha256": manifest["sha256"]})


if __name__ == "__main__":
    main()
