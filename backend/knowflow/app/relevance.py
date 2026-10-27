from __future__ import annotations

import json
import re
from collections import Counter
from time import perf_counter
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from knowflow.app.config import Settings
from knowflow.app.providers import (
    chat_client,
    is_quota_error,
    mark_quota_blocked,
    structured_request_options,
    task_route,
)
from knowflow.app.relevance_budget import messages, plan
from knowflow.app.round_budget import RoundBudgetExceeded, chat_request
from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics

PROMPT_VERSION = "relevance-v3-source-identity"
ACCEPTED_VERDICTS = {"direct", "partial"}


def retain_supported_context(
    context: list[dict[str, str]], assessments: dict[str, RelevanceAssessment]
) -> list[dict[str, str]]:
    """Do not mark a truncated passage usable if its validated support was cut away."""
    return [
        row
        for row in context
        if all(
            " ".join(quote.split()) in " ".join(row["content"].split())
            for quote in assessments[row["id"]].supporting_quotes
        )
    ]


class AssessedRow(BaseModel):
    chunk_id: UUID
    verdict: str = Field(pattern="^(direct|partial|background|irrelevant|unknown)$")
    reason: str = Field(min_length=1, max_length=500)
    quotes: list[str] = Field(default_factory=list, max_length=3)


class AssessedBatch(BaseModel):
    passages: list[AssessedRow] = Field(max_length=30)


def validate_assessments(
    raw: str, evidence: list[dict[str, str]]
) -> dict[str, RelevanceAssessment]:
    raw = re.sub(r"^```(?:json)?\s*\n([\s\S]*?)\n```$", r"\1", raw.strip())
    try:
        batch = AssessedBatch.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as exc:
        raise ValueError("Invalid relevance response schema") from exc
    counts = Counter(str(row.chunk_id) for row in batch.passages)
    rows = {str(row.chunk_id): row for row in batch.passages}
    assessments = {}
    for item in evidence:
        chunk_id = item["id"]
        row = rows.get(chunk_id)
        if row is None or counts[chunk_id] != 1:
            assessments[chunk_id] = RelevanceAssessment(reason="检测结果缺失或重复，尚未核验。")
            continue
        content = " ".join(item["content"].split())
        quotes = list(dict.fromkeys(quote.strip() for quote in row.quotes if quote.strip()))
        valid_quotes = [
            quote for quote in quotes if len(quote) >= 4 and " ".join(quote.split()) in content
        ]
        # A model cannot promote a passage using invented or absent supporting text.
        if len(valid_quotes) != len(quotes) or (
            row.verdict in ACCEPTED_VERDICTS and not valid_quotes
        ):
            assessments[chunk_id] = RelevanceAssessment(
                reason="检测声称相关，但提供的依据未通过原文校验；暂不采用。"
            )
            continue
        assessments[chunk_id] = RelevanceAssessment(
            verdict=row.verdict, reason=row.reason, supporting_quotes=valid_quotes
        )
    return assessments


async def assess_retrieval(
    question: str, evidence: list[dict[str, str]], settings: Settings
) -> tuple[dict[str, RelevanceAssessment], RetrievalDiagnostics]:
    started = perf_counter()
    route = task_route(
        "relevance", settings, question + "\n" + "\n".join(r["content"] for r in evidence)
    )
    defaults = {item["id"]: RelevanceAssessment(reason="相关性尚未核验。") for item in evidence}
    diagnostics = RetrievalDiagnostics(
        status="empty" if not evidence else "unavailable",
        assessor_model=route.model,
        model_route=route.public(),
        prompt_version=PROMPT_VERSION,
    )
    if not evidence:
        return defaults, diagnostics
    if not settings.retrieval_assessment_enabled:
        diagnostics.status = "disabled"
        defaults = {
            key: RelevanceAssessment(reason="相关性检查已关闭，仅展示原始检索分数。")
            for key in defaults
        }
        return defaults, diagnostics
    inspected, trace = plan(question, evidence, settings, route)
    diagnostics.context_budget = trace
    reasons = {
        "empty": "候选为空，尚未核验。",
        "duplicate_id": "候选身份重复，尚未核验。",
        "passage_limit": "超出单次30片段限制，尚未核验。",
    }
    for decision in trace["decisions"]:
        if decision["decision"] == "skipped":
            defaults[decision["id"]] = RelevanceAssessment(
                reason=reasons.get(
                    decision["reason"], "超出本次完整请求预算，未截断送检；尚未核验。"
                )
            )
    if not inspected:
        trace["reason_code"] = (
            "fixed_input_exhausted" if not trace["estimated_feasible"] else "no_whole_passage_fits"
        )
        diagnostics.assessment_latency_ms = round((perf_counter() - started) * 1000, 2)
        return defaults, diagnostics
    try:
        trace["phase"] = "configuration"
        client = chat_client()
        trace.update(phase="request", request_attempted=True)

        async def request_batch(group, model=None, thinking=None):
            target = model or route.model or ""
            return await chat_request(
                "relevance_assessment",
                client,
                model=target,
                temperature=0,
                response_format={"type": "json_object"},
                **structured_request_options(
                    target,
                    settings.llm_base_url,
                    settings,
                    thinking=thinking or route.thinking,
                ),
                **{trace["output_limit_parameter"]: trace["output_reserved_tokens"]},
                messages=messages(question, group),
            )

        # Long candidate lists are reviewed in bounded batches: one huge reply can
        # hit the provider timeout or the output cap and lose every verdict.
        batch_size = 10
        groups = [inspected[i : i + batch_size] for i in range(0, len(inspected), batch_size)]
        trace["batches"] = len(groups)
        usages = []
        for index, group in enumerate(groups, 1):
            trace["batch_index"] = index
            try:
                response = await request_batch(group)
            except RoundBudgetExceeded:
                raise
            except Exception as exc:
                if is_quota_error(exc) and settings.llm_model and settings.llm_model != route.model:
                    # The larger model's free quota is gone; finish on the configured model.
                    trace["model_fallback"] = settings.llm_model
                    mark_quota_blocked(route.model)
                    response = await request_batch(
                        group, model=settings.llm_model, thinking="disabled"
                    )
                else:
                    # A transient provider error must not discard an otherwise usable batch.
                    trace["provider_retry"] = trace.get("provider_retry", 0) + 1
                    response = await request_batch(group)
            trace.update(phase="response", response_received=True)
            usage = getattr(response, "usage", None)
            usages.append(
                usage.model_dump(mode="json")
                if hasattr(usage, "model_dump")
                else (usage if isinstance(usage, dict) else None)
            )
            trace["provider_usage"] = usages[-1]
            trace["provider_usage_batches"] = usages
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
                raise ValueError("Nonfinal relevance response")
            raw = choice.message.content
            if not raw:
                trace["reason_code"] = "empty_content"
                raise ValueError("Empty relevance response")
            trace["phase"] = "validation"
            trace["reason_code"] = "invalid_response_schema"
            try:
                defaults.update(validate_assessments(raw, group))
            except ValueError:
                # One malformed JSON reply must not discard an otherwise usable batch.
                trace["validation_attempts"] = trace.get("validation_attempts", 0) + 1
                retry = await request_batch(group)
                retry_choice = retry.choices[0] if retry.choices else None
                if getattr(retry_choice, "finish_reason", None) != "stop" or not getattr(
                    retry_choice, "message", None
                ):
                    raise
                raw = retry_choice.message.content or ""
                if not raw:
                    raise
                defaults.update(validate_assessments(raw, group))
        unknown = any(row.verdict == "unknown" for row in defaults.values())
        diagnostics.status = "partial" if unknown else "complete"
        trace.update(
            phase="complete",
            status="completed",
            reason_code="validated_response",
            observed_count=sum(row.verdict != "unknown" for row in defaults.values()),
            unknown_count=sum(row.verdict == "unknown" for row in defaults.values()),
        )
    except Exception as exc:
        if isinstance(exc, RoundBudgetExceeded):
            trace.update(request_attempted=False, reason_code="round_budget_exhausted")
        # Do not leak provider exceptions, source content, or credentials to the UI.
        for item in inspected:
            defaults[item["id"]] = RelevanceAssessment(
                reason="相关性检测暂不可用，此片段尚未核验。"
            )
        trace["status"] = "unavailable"
        trace["reason_code"] = trace["reason_code"] or (
            "client_configuration_error" if trace["phase"] == "configuration" else "provider_error"
        )
        diagnostics.status = "unavailable"
    diagnostics.assessment_latency_ms = round((perf_counter() - started) * 1000, 2)
    return defaults, diagnostics
