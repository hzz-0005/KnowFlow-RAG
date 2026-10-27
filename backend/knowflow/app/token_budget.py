"""Offline tokenizer adapter. A related-family count is explicitly an estimate."""

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

ASSETS = Path(__file__).parent / "assets"


class TokenCounter:
    def __init__(self, path: str | None = None):
        if path:
            raw = Path(path).read_bytes()
            self.metadata = {"mode": "configured_local_tokenizer", "model_match": "unverified"}
        else:
            raw = gzip.decompress((ASSETS / "qwen3-tokenizer.json.gz").read_bytes())
            self.metadata = json.loads(
                (ASSETS / "qwen3-tokenizer.json").read_text(encoding="utf-8")
            )
            if hashlib.sha256(raw).hexdigest() != self.metadata["sha256"]:
                raise ValueError("Bundled tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.metadata = {**self.metadata, "sha256": hashlib.sha256(raw).hexdigest()}

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prefix_end(self, text: str, budget: int) -> int:
        if budget <= 0:
            return 0
        # Bound the work per fragment. Encoding the entire remaining document
        # for every prefix is quadratic for a large TXT/Markdown section.
        window = min(len(text), max(1024, budget * 8))
        encoded = self.tokenizer.encode(text[:window], add_special_tokens=False)
        while len(encoded.ids) <= budget and window < len(text):
            window = min(len(text), window * 2)
            encoded = self.tokenizer.encode(text[:window], add_special_tokens=False)
        if len(encoded.ids) <= budget:
            return len(text)
        # Never cut inside a multi-token Unicode character (several byte tokens
        # can share one character offset). Verify the actual sliced string too.
        end = encoded.offsets[budget - 1][1]
        while end and self.count(text[:end]) > budget:
            end -= 1
        return end


@lru_cache(maxsize=4)
def get_token_counter(path: str | None = None):
    return TokenCounter(path)
