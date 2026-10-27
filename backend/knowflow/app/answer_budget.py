"""Generation-stage estimated budget using the exact prompt builder and whole sources."""

import hashlib
import json

from knowflow.app.intent_context import RoutingInput
from knowflow.app.providers import answer_messages, task_route
from knowflow.app.request_budget import estimate_messages, output_limit
from knowflow.app.token_budget import get_token_counter

VERSION = "answer-window-whole-sources-v1"


class AnswerInputBudgetError(ValueError):
    pass


class AnswerOutputLimit(RuntimeError):
    pass


def budget_contract(settings):
    counter = get_token_counter(getattr(settings, "chunk_tokenizer_path", None))
    return {
        "version": VERSION,
        "total_tokens": settings.rag_answer_total_token_budget,
        "output_tokens": settings.rag_answer_output_token_budget,
        "safety_tokens": settings.rag_answer_safety_tokens,
        "evidence_tokens": settings.rag_context_token_budget,
        "output_parameter": settings.rag_answer_output_parameter,
        "tokenizer_sha256": counter.metadata["sha256"],
        "provider_tokenizer_match": "unverified",
        "packing": "whole_source_or_skip",
    }


def _estimate(messages, counter):
    # Chat framing is provider-specific; this is deliberately labeled an estimate.
    return estimate_messages(messages, counter)


def measure_request(question, evidence, settings, *, routing_input: RoutingInput | None = None):
    counter = get_token_counter(getattr(settings, "chunk_tokenizer_path", None))
    messages = answer_messages(question, evidence)
    fixed = _estimate(answer_messages(question, []), counter)
    used = _estimate(messages, counter)
    total = getattr(settings, "rag_answer_total_token_budget", 32768)
    output = getattr(settings, "rag_answer_output_token_budget", 8192)
    safety = getattr(settings, "rag_answer_safety_tokens", 1024)
    route_input = routing_input or RoutingInput(question, "question_text")
    route = task_route("answer", settings, route_input.text, len(evidence))
    parameter, scope = output_limit(
        route.model,
        getattr(settings, "llm_base_url", None),
        getattr(settings, "rag_answer_output_parameter", "auto"),
    )
    cap = getattr(settings, "rag_context_token_budget", 6000)
    feasible = used + output + safety <= total and used - fixed <= cap
    return {
        "version": VERSION,
        "stage": "answer_draft_generation",
        "status": "ready" if feasible else "blocked",
        "measurement": "local_tokenizer_and_estimated_chat_framing",
        "tokenizer_sha256": counter.metadata["sha256"],
        "provider_tokenizer_match": "unverified",
        "configured_total_tokens": total,
        "fixed_input_tokens_estimate": fixed,
        "input_tokens_estimate": used,
        "evidence_tokens_estimate": used - fixed,
        "evidence_cap_tokens": cap,
        "output_reserved_tokens": output,
        "safety_tokens": safety,
        "estimated_total_tokens": used + output + safety,
        "estimated_feasible": feasible,
        "output_limit_parameter": parameter,
        "output_limit_scope": scope,
        "model": route.model,
        "model_route": route.public(),
        "routing_input_basis": route_input.basis,
        "routing_input_sha256": hashlib.sha256(route_input.text.encode()).hexdigest(),
        "routing_input_chars": len(route_input.text),
        "provider_usage": None,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "note": (
            "本地估算与配置的运行预算；不代表供应商真实窗口或分词精确匹配。"
            "其他调用阶段仍有独立预算。"
        ),
    }


def plan_answer_context(
    question, candidates, settings, *, routing_input: RoutingInput | None = None
):
    selected, decisions, seen = [], [], set()
    base = measure_request(question, [], settings, routing_input=routing_input)
    for item in candidates:
        content = item["content"]
        key = content
        reason = "empty" if not content.strip() else "duplicate" if key in seen else None
        if reason is None:
            seen.add(key)
            checked = measure_request(
                question, [*selected, item], settings, routing_input=routing_input
            )
            if not checked["estimated_feasible"]:
                reason = (
                    "fixed_input_exhausted"
                    if not base["estimated_feasible"]
                    else (
                        "evidence_cap"
                        if checked["evidence_tokens_estimate"] > checked["evidence_cap_tokens"]
                        else "window_budget"
                    )
                )
            else:
                selected.append(item)
        decisions.append(
            {
                "id": item["id"],
                "decision": "skipped" if reason else "selected",
                "reason": reason,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    trace = measure_request(question, selected, settings, routing_input=routing_input)
    trace.update(
        selected_count=len(selected),
        skipped_count=len(candidates) - len(selected),
        decisions=decisions,
        truncation_policy="whole_source_or_skip",
    )
    if candidates and not selected and base["estimated_feasible"]:
        trace["status"] = "no_evidence_fits"
    return selected, trace
