"""Whole-message operational budgets for context resolution and query rewriting."""

import hashlib
import json

from knowflow.app.request_budget import estimate_messages, output_limit
from knowflow.app.token_budget import get_token_counter

VERSION = "understanding-whole-request-v1"


def contract(settings, stage):
    return {
        "version": VERSION,
        "stage": stage,
        "packing": "all_inputs_or_no_request",
        "completion_policy": "stop_only",
        "thinking": "disabled",
        "tokenizer_sha256": get_token_counter(settings.chunk_tokenizer_path).metadata["sha256"],
        **{
            key: getattr(settings, stage + "_" + key)
            for key in (
                "total_token_budget",
                "output_token_budget",
                "safety_tokens",
                "output_parameter",
            )
        },
    }


def measure(messages, settings, stage):
    spec = contract(settings, stage)
    counter = get_token_counter(settings.chunk_tokenizer_path)
    used = estimate_messages(messages, counter)
    reserved = spec["output_token_budget"]
    safety = spec["safety_tokens"]
    feasible = used + reserved + safety <= spec["total_token_budget"]
    parameter, scope = output_limit(
        settings.llm_model, settings.llm_base_url, spec["output_parameter"]
    )
    return {
        "version": VERSION,
        "stage": stage,
        "status": "ready" if feasible else "blocked",
        "packing": "all_inputs_or_no_request",
        "input_tokens_estimate": used,
        "output_reserved_tokens": reserved,
        "safety_tokens": safety,
        "estimated_total_tokens": used + reserved + safety,
        "configured_total_tokens": spec["total_token_budget"],
        "estimated_feasible": feasible,
        "measurement": "local_tokenizer_and_estimated_chat_framing",
        "tokenizer_sha256": spec["tokenizer_sha256"],
        "provider_tokenizer_match": "unverified",
        "model": settings.llm_model,
        "thinking": "disabled",
        "output_limit_parameter": parameter,
        "output_limit_scope": scope,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "request_attempted": False,
        "response_received": False,
        "provider_usage": None,
        "finish_reason": None,
        "reason_code": "request_budget_exceeded" if not feasible else None,
    }


def record_result(trace, result):
    """Keep observed response facts even when validation fails; never serialize errors."""
    trace.update(
        status="blocked" if not trace["estimated_feasible"] else result["status"],
        request_attempted=result["request_attempted"],
        response_received=result["response_received"],
        provider_usage=result["provider_usage"],
        finish_reason=result.get("finish_reason"),
        reason_code=result.get("reason_code", result.get("reason", "validated_response")),
    )
