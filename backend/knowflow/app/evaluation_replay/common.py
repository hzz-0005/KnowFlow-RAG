from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ReplayError(ValueError):
    """Invalid input/contract, distinct from a measured quality failure."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_object,
            parse_constant=lambda value: invalid(f"Nonfinite JSON constant: {value}"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"Cannot read {path.name}: {type(exc).__name__}") from exc


def invalid(message):
    raise ReplayError(message)


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReplayError("Input path must be relative to manifest root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ReplayError(f"Input escapes root: {relative}")
    return path


def json_write(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
