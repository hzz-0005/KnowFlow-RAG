"""Run tests against an isolated generated PostgreSQL database, then remove only it.

Usage: python scripts/run_tests.py [-q] [pytest arguments ...]
Optional KNOWFLOW_TEST_EVIDENCE_DIR records subprocess output fingerprints.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from knowflow.app.config import get_settings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main():
    url = make_url(get_settings().database_url)
    name = "knowflow_tests_" + uuid4().hex
    assert name != url.database and name.startswith("knowflow_tests_")
    assert name.replace("_", "").isalnum()
    evidence = os.environ.get("KNOWFLOW_TEST_EVIDENCE_DIR")
    out = Path(evidence).resolve() if evidence else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records = []
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    created = False
    result = {
        "database": name,
        "created": False,
        "dropped": False,
        "application_database_used_for_tests": False,
    }
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
            created = True
            result["created"] = True
        # Credentials stay in the child environment, never argv or evidence.
        env = dict(
            os.environ, DATABASE_URL=url.set(database=name).render_as_string(hide_password=False)
        )
        for index, argv in enumerate(
            [
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                [sys.executable, "-m", "pytest", *(sys.argv[1:] or ["-q"])],
            ]
        ):
            started = datetime.now(UTC).isoformat()
            run = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True)
            row = {
                "argv": argv,
                "cwd": str(ROOT),
                "started_at": started,
                "completed_at": datetime.now(UTC).isoformat(),
                "exit_code": run.returncode,
                "database_scope": "disposable_generated_database",
            }
            if out:
                for channel, data in (("stdout", run.stdout), ("stderr", run.stderr)):
                    path = out / f"isolated-{stamp}-{index}-{channel}.txt"
                    path.write_bytes(data)
                    row[channel] = {
                        "path": str(path.relative_to(ROOT))
                        if path.is_relative_to(ROOT)
                        else str(path),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                records.append(row)
                (out / f"commands-isolated-{stamp}.json").write_text(
                    json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8"
                )
            sys.stdout.buffer.write(run.stdout)
            sys.stderr.buffer.write(run.stderr)
            if run.returncode:
                return run.returncode
        return 0
    finally:
        try:
            if created:
                # Only the exact database created by this invocation is removable.
                assert result["database"] == name and name != url.database
                with admin.connect() as connection:
                    connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
                result["dropped"] = True
        finally:
            admin.dispose()
            if out:
                (out / f"isolation-{stamp}.json").write_text(
                    json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
                )


if __name__ == "__main__":
    sys.exit(main())
