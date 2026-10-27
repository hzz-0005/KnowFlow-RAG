"""Create a local configuration with a unique JWT secret; never overwrite."""

import argparse
import secrets
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    example = Path(__file__).resolve().parents[1] / ".env.example"
    text = example.read_text(encoding="utf-8")
    placeholder = "replace-this-with-a-random-secret-of-at-least-32-characters"
    if text.count(placeholder) != 1:
        parser.error("Example JWT placeholder missing or ambiguous")
    text = text.replace(placeholder, secrets.token_urlsafe(48))
    try:
        with args.out.open("x", encoding="utf-8") as output:
            output.write(text)
    except FileExistsError:
        parser.error("Configuration already exists; it was not overwritten")
    print("Configuration created. Fill provider keys locally; secrets are not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
