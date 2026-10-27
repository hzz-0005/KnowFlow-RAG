"""Local deterministic protocol stub, not a language model or semantic benchmark."""

import hashlib
import json
import math
import re

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="KnowFlow software contract stub — not a real model")


@app.get("/health")
def health():
    return {"status": "ok", "kind": "software_contract_stub"}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body["input"]
    inputs = [inputs] if isinstance(inputs, str) else inputs
    dimensions = body.get("dimensions", 2560)
    if dimensions != 2560:
        return {"error": {"message": "Stub expects 2560 dimensions"}}
    rows = []
    for i, text in enumerate(inputs):
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            vector[
                int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % dimensions
            ] += 1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        rows.append({"object": "embedding", "index": i, "embedding": [v / norm for v in vector]})
    return {
        "object": "list",
        "model": "local-contract-stub",
        "data": rows,
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body["messages"]
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "requirement_sources" in user:
        content = json.dumps({"requirements": []})
    elif "current_question" in user:
        data = json.loads(user)
        content = json.dumps(
            {"status": "standalone", "question": data["current_question"], "reference_turn_ids": []}
        )
    elif "explicit entities and relationships" in system:
        evidence = "Orion depends on Atlas for authentication."
        facts = {"entities": [], "relations": []}
        if evidence in user:
            facts = {
                "entities": [{"name": n, "evidence": evidence} for n in ("Orion", "Atlas")],
                "relations": [
                    {
                        "source": "Orion",
                        "target": "Atlas",
                        "predicate": "depends_on",
                        "description": evidence,
                        "evidence": evidence,
                        "confidence": 1.0,
                        "qualifiers": [],
                    }
                ],
            }
        content = json.dumps(facts)
    else:
        match = re.search(r"\[KF1\]\s*(.*?)(?=\n\n\[KF\d+\]|\n</evidence>)", user, re.S)
        content = (
            ("本地软件测试桩复述来源，不代表模型理解质量：\n\n" + match.group(1).strip() + " [KF1]")
            if match
            else "Local contract stub has no supplied evidence."
        )
    return {
        "id": "local-contract-stub",
        "object": "chat.completion",
        "model": "local-contract-stub",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9100)
