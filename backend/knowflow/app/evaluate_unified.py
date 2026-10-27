"""Offline ledger CLI: exit 0=replayed, 2=invalid input, 3=unpaired, 4=explicit gate."""

import argparse
import sys
from pathlib import Path

from .evaluation_replay.common import ReplayError
from .evaluation_replay.report import replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("evaluation/unified_v2/manifest.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-comparable", action="store_true")
    parser.add_argument("--gates", type=Path)
    args = parser.parse_args(argv)
    try:
        result = replay(
            args.manifest,
            args.out,
            require_comparable=args.require_comparable,
            gates=args.gates,
            argv=(
                list(sys.orig_argv)
                if argv is None
                else [sys.executable, "-m", "knowflow.app.evaluate_unified", *argv]
            ),
        )
    except (ReplayError, OSError, KeyError, TypeError) as exc:
        message = str(exc) if isinstance(exc, ReplayError) else type(exc).__name__
        print(f"M1 input/output error: {message}", file=sys.stderr)
        return 2
    sys.stdout.write(result["stdout"])
    sys.stderr.write(result["stderr"])
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
