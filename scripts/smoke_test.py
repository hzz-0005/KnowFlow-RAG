"""Exercise an isolated compose.smoke.yml stack, never a semantic benchmark.

Uses only Python's standard library. The normal application's ports are refused;
the caller must obtain the random contract-stack port from Docker Compose.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


def run(base):
    parsed = urlparse(base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.port in {None, 80, 443, 8000, 8080}
        or parsed.path not in {"", "/"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use the isolated contract stack's random localhost HTTP port")
    base = base.rstrip("/")
    opener = build_opener(ProxyHandler({}))
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "status": "passed"})

    def call(method, path, payload=None, token=None, expected=200, raw=None, headers=None):
        headers = dict(headers or {})
        if token:
            headers["Authorization"] = "Bearer " + token
        data = raw
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        try:
            with opener.open(
                Request(base + path, data=data, method=method, headers=headers), timeout=60
            ) as response:
                status, content = response.status, response.read()
        except HTTPError as exc:
            status, content = exc.code, exc.read()
        if status != expected:
            raise AssertionError(f"{method} {path}: expected {expected}, received {status}")
        return json.loads(content) if content else None

    def poll(path, token, done):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = call("GET", path, token=token)
            if done(result):
                return result
            time.sleep(0.3)
        raise TimeoutError(f"Contract stack did not finish {path}")

    health = call("GET", "/health")
    check("health", health.get("status") == "ok")
    with opener.open(base, timeout=15) as response:
        check("frontend_html", b"FinCite" in response.read())
    tokens = []
    for _ in range(2):
        result = call(
            "POST",
            "/api/v1/auth/register",
            {
                "email": f"contract-{uuid4().hex}@example.com",
                "password": uuid4().hex + "-contract",
            },
            expected=201,
        )
        tokens.append(result["access_token"])
    token, other = tokens
    kb = call(
        "POST",
        "/api/v1/knowledge-bases",
        {"name": "Software contract fixture"},
        token,
        expected=201,
    )["id"]
    call("GET", f"/api/v1/knowledge-bases/{kb}/documents", token=other, expected=404)
    check("knowledge_base_owner_isolation", True)

    boundary = "contract-" + uuid4().hex
    parts = []
    for filename, content in [
        (
            "orion.md",
            b"# Orion\nOrion depends on Atlas for authentication.\n"
            b"Orion retains logs for 17 days.\n",
        ),
        ("retention.csv", b"service,retention_days\nOrion,17\n"),
    ]:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            + content
            + b"\r\n"
        )
    multipart = b"".join(parts) + f"--{boundary}--\r\n".encode()
    headers = {
        "Content-Type": "multipart/form-data; boundary=" + boundary,
        "Idempotency-Key": str(uuid4()),
    }
    path = f"/api/v1/knowledge-bases/{kb}/ingestion-batches"
    batch = call("POST", path, token=token, raw=multipart, headers=headers, expected=202)
    replay = call("POST", path, token=token, raw=multipart, headers=headers, expected=202)
    check("batch_request_idempotency", batch["id"] == replay["id"])
    batch = poll(f"/api/v1/ingestion-batches/{batch['id']}", token, lambda b: b["active"] == 0)
    check("md_csv_ingestion", batch["counts"]["completed"] == 2)
    documents = call("GET", f"/api/v1/knowledge-bases/{kb}/documents", token=token)
    doc = next(d for d in documents if d["filename"] == "orion.md")
    check(
        "published_documents",
        len(documents) == 2 and all(d["status"] == "processed" for d in documents),
    )

    answer = call(
        "POST",
        f"/api/v1/knowledge-bases/{kb}/query",
        {
            "question": "Orion authentication",
            "strategy": "keyword",
            "top_k": 3,
        },
        token,
    )
    check(
        "stub_answer_and_citations",
        not answer["refused"]
        and bool(answer["citations"])
        and "本地软件测试桩" in answer["answer"],
    )
    missing = call(
        "POST",
        f"/api/v1/knowledge-bases/{kb}/query",
        {
            "question": "zqxv_unrelated_term_987",
            "strategy": "keyword",
        },
        token,
    )
    check("no_evidence_refusal", missing["refused"] and not missing["citations"])

    conv = call("POST", f"/api/v1/knowledge-bases/{kb}/conversations", token=token, expected=201)
    turn_body = {
        "question": "Orion authentication",
        "strategy": "keyword",
        "request_id": str(uuid4()),
        "expected_revision": 0,
    }
    path = f"/api/v1/conversations/{conv['id']}/turns"
    turn = call("POST", path, turn_body, token)
    replay = call("POST", path, turn_body, token)
    check("conversation_turn_idempotency", turn == replay and turn["status"] == "completed")
    turn_body = {
        "question": "What is Orion authentication?",
        "strategy": "keyword",
        "request_id": str(uuid4()),
        "expected_revision": 1,
    }
    second = call("POST", path, turn_body, token)
    stored = call("GET", f"/api/v1/conversations/{conv['id']}", token=token)
    check(
        "two_turn_persistence",
        stored["revision"] == 2 and len(stored["turns"]) == 2 and second["status"] == "completed",
    )
    call("GET", f"/api/v1/conversations/{conv['id']}", token=other, expected=404)
    check("conversation_owner_isolation", True)
    task = call(
        "POST",
        f"/api/v1/knowledge-bases/{kb}/tasks",
        {"source_turn_id": turn["id"]},
        token,
        expected=201,
    )
    memory = call(
        "POST",
        f"/api/v1/knowledge-bases/{kb}/memories",
        {
            "kind": "alias",
            "key": "contract-alias",
            "content": "Orion",
            "status": "candidate",
        },
        token,
        expected=201,
    )
    check("task_and_candidate_memory", bool(task["id"]) and memory["status"] == "candidate")

    call("POST", f"/api/v1/documents/{doc['id']}/graph-index", token=token, expected=202)
    indexed = poll(
        f"/api/v1/documents/{doc['id']}",
        token,
        lambda d: d["graph_status"] in {"processed", "failed"},
    )
    check("graph_job_completion", indexed["graph_status"] == "processed")
    graph = call("GET", f"/api/v1/knowledge-bases/{kb}/graph", token=token)
    check("graph_inspection_contract", bool(graph.get("edges")))
    check(
        "unknown_graph_semantics_not_silently_approved",
        all(
            "retrieval_eligible" in edge and not edge["retrieval_eligible"]
            for edge in graph["edges"]
        ),
    )
    return {
        "schema": "knowflow-software-contract-v1",
        "passed": True,
        "checks": checks,
        "semantic_quality_measured": False,
        "external_model_calls": False,
        "note": "Deterministic local stub; not evidence of model, retrieval or graph quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing report")
    started = datetime.now(UTC).isoformat()
    try:
        result = run(args.base_url)
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)[:250]}
    result.update(started_at=started, completed_at=datetime.now(UTC).isoformat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "report": str(args.out)}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
