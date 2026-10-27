"""All-or-none review/repair request budgets; never trim conditions or audit targets."""

import hashlib
import json

from knowflow.app.request_budget import estimate_messages, output_limit
from knowflow.app.token_budget import get_token_counter

VERSION = "answer-guard-request-window-v1"


def measure(messages, settings, stage, model, base_url, thinking):
    prefix = "answer_repair" if stage == "repair_generation" else "answer_review"
    total = getattr(settings, prefix + "_total_token_budget")
    output = getattr(settings, prefix + "_output_token_budget")
    safety = getattr(settings, prefix + "_safety_tokens")
    parameter, scope = output_limit(
        model, base_url, getattr(settings, prefix + "_output_parameter")
    )
    counter = get_token_counter(settings.chunk_tokenizer_path)
    used = estimate_messages(messages, counter)
    feasible = used + output + safety <= total
    return {
        "version": VERSION,
        "stage": stage,
        "status": "ready" if feasible else "blocked",
        "input_tokens_estimate": used,
        "output_reserved_tokens": output,
        "safety_tokens": safety,
        "estimated_total_tokens": used + output + safety,
        "configured_total_tokens": total,
        "estimated_feasible": feasible,
        "measurement": "local_tokenizer_and_estimated_chat_framing",
        "tokenizer_sha256": counter.metadata["sha256"],
        "provider_tokenizer_match": "unverified",
        "output_limit_parameter": parameter,
        "output_limit_scope": scope,
        "model": model,
        "thinking": thinking,
        "packing": "all_inputs_or_no_request",
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "request_attempted": False,
        "response_received": False,
        "provider_usage": None,
        "finish_reason": None,
        "phase": "budget",
        "reason_code": None,
    }


def receive(trace, response):
    trace.update(phase="response", response_received=True)
    usage = getattr(response, "usage", None)
    trace["provider_usage"] = (
        usage.model_dump(mode="json")
        if hasattr(usage, "model_dump")
        else (usage if isinstance(usage, dict) else None)
    )
    trace["reported_model"] = getattr(response, "model", None)
    choice = response.choices[0] if response.choices else None
    finish = getattr(choice, "finish_reason", None)
    trace["finish_reason"] = finish
    if finish != "stop":
        trace["reason_code"] = {
            "length": "output_limit_reached",
            "content_filter": "content_filtered",
            None: "finish_reason_missing",
        }.get(finish, "nonfinal_model_output")
        raise ValueError("Nonfinal audited pipeline response")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        trace["reason_code"] = "empty_content"
        raise ValueError("Empty audited pipeline response")
    return content


def budget_contract(settings):
    from knowflow.app.answer_guard import REPAIR_PROMPT, REPAIR_TRIGGER_VERSION
    from knowflow.app.answer_language import POLICY as LANGUAGE_POLICY
    from knowflow.app.answer_language import VERSION as LANGUAGE_VERSION
    from knowflow.app.answer_review import REVIEW_PROMPT

    contract = {
        "version": VERSION,
        "repair_trigger_version": REPAIR_TRIGGER_VERSION,
        "repair_language_contract": LANGUAGE_VERSION,
        "repair_language_policy_sha256": hashlib.sha256(LANGUAGE_POLICY.encode()).hexdigest(),
        "packing": "all_inputs_or_no_request",
        "completion_policy": "stop_only",
        "review_unit_limit": 60,
        "review_context_char_limit": 64000,
        "review_prompt_sha256": hashlib.sha256(REVIEW_PROMPT.encode()).hexdigest(),
        "repair_prompt_sha256": hashlib.sha256(REPAIR_PROMPT.encode()).hexdigest(),
        "tokenizer_sha256": get_token_counter(settings.chunk_tokenizer_path).metadata["sha256"],
    }
    for prefix in ("answer_review", "answer_repair"):
        contract[prefix] = {
            key: getattr(settings, prefix + "_" + key)
            for key in (
                "total_token_budget",
                "output_token_budget",
                "safety_tokens",
                "output_parameter",
            )
        }
    contract.update(
        guard_enabled=settings.answer_guard_enabled,
        repair_enabled=settings.answer_repair_enabled,
        guard_thinking=settings.answer_guard_thinking,
        offline_review_thinking=settings.eval_judge_thinking,
    )
    return contract
