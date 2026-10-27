"""Durable conversation turns; claims are short, completion is fenced and idempotent."""

import asyncio
import hashlib
import json
import logging
import re
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field
from sqlalchemy import select

from knowflow.app.api import UserDep, owned_knowledge_base, query_knowledge_base
from knowflow.app.clarification_hints import (
    document_hints,
    memory_invalidation,
    restatement_reasons,
)
from knowflow.app.config import get_settings
from knowflow.app.conversation_context import aggregate_list_question, resolve_question
from knowflow.app.db import SessionLocal
from knowflow.app.document_reference import (
    allows_prior_document,
    named_document_resolution,
    prior_document_followup,
    resolve_document_reference,
)
from knowflow.app.models import Conversation, ConversationTurn, Document, utc_now
from knowflow.app.round_budget import CURRENT, RoundBudget
from knowflow.app.schemas import QueryRequest, QueryResponse
from knowflow.app.security import SessionDep

# The client only ever sees "this turn did not complete"; the cause stays in logs.
logger = logging.getLogger("knowflow.turn")

router = APIRouter(prefix="/api/v1", tags=["conversations"])


class TurnRequest(QueryRequest):
    request_id: UUID
    expected_revision: int = Field(ge=0)
    task_id: UUID | None = None
    task_revision: int | None = Field(None, ge=0)


async def prior_document_ids(session, conversation_id):
    """Document identities the completed turn cited; used only to resolve '这份文档'.

    Read through a fresh scalar query: the caller may have rolled back or committed,
    which expires the ORM instance and would otherwise trigger a lazy load.
    """
    state = await session.scalar(
        select(Conversation.context_state).where(Conversation.id == conversation_id)
    )
    raw = (state or {}).get("source_document_ids") or []
    try:
        return tuple(UUID(str(value)) for value in raw if value)
    except (TypeError, ValueError):
        return None


def conversation_view(row):
    return {
        "id": row.id,
        "knowledge_base_id": row.knowledge_base_id,
        "title": row.title,
        "revision": row.revision,
        "active_turn_id": row.active_turn_id,
        "active_until": row.active_until,
        "context_state": row.context_state,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def turn_view(row):
    return {
        key: getattr(row, key)
        for key in (
            "id",
            "conversation_id",
            "sequence",
            "request_id",
            "question",
            "resolved_question",
            "resolution",
            "status",
            "response",
            "error",
            "created_at",
            "finished_at",
        )
    }


async def owned_conversation(session, cid, user_id, *, lock=False):
    statement = select(Conversation).where(Conversation.id == cid, Conversation.owner_id == user_id)
    if lock:
        statement = statement.with_for_update()
    row = await session.scalar(statement)
    if row is None:
        raise HTTPException(404, "Conversation not found")
    await owned_knowledge_base(session, row.knowledge_base_id, user_id)
    return row


@router.post("/knowledge-bases/{kb_id}/conversations", status_code=201)
async def create_conversation(kb_id: UUID, user: UserDep, session: SessionDep):
    await owned_knowledge_base(session, kb_id, user.id)
    row = Conversation(owner_id=user.id, knowledge_base_id=kb_id)
    session.add(row)
    await session.commit()
    return conversation_view(row)


@router.get("/knowledge-bases/{kb_id}/conversations")
async def list_conversations(kb_id: UUID, user: UserDep, session: SessionDep):
    await owned_knowledge_base(session, kb_id, user.id)
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.owner_id == user.id, Conversation.knowledge_base_id == kb_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id)
        .limit(50)
    )
    return {"items": [conversation_view(r) for r in rows], "limit": 50}


@router.delete("/conversations/{cid}", status_code=204)
async def delete_conversation(cid: UUID, user: UserDep, session: SessionDep):
    """Delete one owned conversation and its turns.

    A turn still holding the execution lease is refused (409) so an in-flight
    answer cannot write into a deleted conversation. Task checkpoints sourced
    from these turns cascade with them; user-level memories are not touched.
    """
    row = await owned_conversation(session, cid, user.id, lock=True)
    if row.active_turn_id is not None and row.active_until and row.active_until > utc_now():
        raise HTTPException(409, "这个对话还有正在生成的回答，完成后再删除")
    await session.delete(row)
    await session.commit()
    return None


@router.get("/conversations/{cid}")
async def get_conversation(
    cid: UUID,
    user: UserDep,
    session: SessionDep,
    before: int | None = Query(None, ge=1),
    limit: int = Query(10, ge=1, le=20),
):
    row = await owned_conversation(session, cid, user.id)
    statement = select(ConversationTurn).where(ConversationTurn.conversation_id == cid)
    if before is not None:
        statement = statement.where(ConversationTurn.sequence < before)
    turns = list(
        await session.scalars(statement.order_by(ConversationTurn.sequence.desc()).limit(limit + 1))
    )
    return {
        **conversation_view(row),
        "turns": [turn_view(t) for t in reversed(turns[:limit])],
        "has_more": len(turns) > limit,
    }


def fingerprint(body):
    return hashlib.sha256(
        json.dumps(
            body.model_dump(mode="json", exclude={"request_id", "expected_revision"}),
            ensure_ascii=True,
            sort_keys=True,
        ).encode()
    ).hexdigest()


async def claim_turn(cid, body, user_id, session):
    row = await owned_conversation(session, cid, user_id, lock=True)
    existing = await session.scalar(
        select(ConversationTurn).where(
            ConversationTurn.conversation_id == cid, ConversationTurn.request_id == body.request_id
        )
    )
    digest = fingerprint(body)
    if existing:
        if existing.request_sha256 != digest:
            raise HTTPException(409, "同一请求标识不能用于不同的问题或参数。")
        if (
            existing.status == "pending"
            and row.active_turn_id == existing.id
            and row.active_until
            and row.active_until <= utc_now()
        ):
            existing.status, existing.finished_at = "interrupted", utc_now()
            existing.error = {
                "code": "lease_expired",
                "message": "上轮未能确认完成，结果未知；未重复调用模型。",
            }
            row.active_turn_id, row.active_until = None, None
        await session.commit()
        return row, existing, False
    now = utc_now()
    if row.revision != body.expected_revision:
        raise HTTPException(409, "对话已更新，请刷新对话后再发送。")
    if row.active_turn_id and row.active_until and row.active_until > now:
        raise HTTPException(409, "此对话仍有一轮正在处理，请等待结果。")
    if row.active_turn_id:
        previous = await session.get(ConversationTurn, row.active_turn_id)
        if previous and previous.status == "pending":
            previous.status, previous.finished_at = "interrupted", now
            previous.error = {
                "code": "lease_expired",
                "message": "上轮未能确认完成，已结束等待；结果未知。",
            }
    turn = ConversationTurn(
        id=uuid4(),
        conversation_id=cid,
        sequence=row.revision + 1,
        request_id=body.request_id,
        request_sha256=digest,
        question=body.question,
    )
    row.revision += 1
    row.active_turn_id = turn.id
    row.active_until = now + timedelta(seconds=get_settings().query_timeout_seconds + 60)
    row.updated_at = now
    if row.revision == 1:
        row.title = body.question[:80]
    session.add(turn)
    from knowflow.app.task_memory import claim as claim_task

    await session.flush()
    await claim_task(session, row, turn, body)
    await session.commit()  # Never keep the claim lock across model calls.
    return row, turn, True


async def recent_intents(session, conversation):
    rows = list(
        await session.scalars(
            select(ConversationTurn)
            .where(
                ConversationTurn.conversation_id == conversation.id,
                ConversationTurn.status.in_(["completed", "needs_clarification"]),
            )
            .order_by(ConversationTurn.sequence.desc())
            .limit(8)
        )
    )
    document_ids = {
        UUID(c["document_id"]) for turn in rows for c in (turn.response or {}).get("citations", [])
    }
    document_ids.update(
        UUID(key)
        for turn in rows
        for key in (turn.resolution or {}).get("referent_candidates", {}).get("source_versions", {})
    )
    docs = list(
        await session.scalars(
            select(Document).where(
                Document.knowledge_base_id == conversation.knowledge_base_id,
                Document.id.in_(document_ids),
                Document.status == "processed",
            )
        )
    )
    versions = {
        str(d.id): {"sha256": d.sha256, "updated_at": d.updated_at.isoformat()} for d in docs
    }
    result = []
    for turn in rows:
        from knowflow.app.memory_models import UserMemory

        # Do not revive deleted/expired/corrected aliases through an old resolved hint.
        from knowflow.app.memory_store import dependencies

        memory_dependencies = dependencies(turn.resolution or {})
        reasons = set()
        for dep in memory_dependencies:
            memory = await session.get(UserMemory, UUID(dep["id"]))
            reason = memory_invalidation(
                memory, dep, conversation.owner_id, conversation.knowledge_base_id
            )
            if reason:
                reasons.add(reason)
        if reasons:
            result.append(
                SimpleNamespace(
                    id=turn.id,
                    question=turn.question,
                    resolved_question=None,
                    status="needs_clarification",
                    response={"citations": []},
                    resolution={
                        "pending_intent": {
                            "question": "",
                            "supplements": [],
                            "requires_restatement": True,
                            "restatement_reasons": sorted(reasons),
                        },
                        "question": "此前引用的记忆已更正、删除或过期，请明确当前对象。",
                    },
                )
            )
            continue
        citations = [
            c
            for c in (turn.response or {}).get("citations", [])
            if turn.source_versions.get(c.get("document_id")) == versions.get(c.get("document_id"))
            and c.get("document_id") in versions
        ]
        saved_resolution = turn.resolution or {}
        offered = saved_resolution.get("referent_candidates")
        if offered:
            offered = {
                **offered,
                "options": [
                    r
                    for r in offered.get("options", [])
                    if offered.get("source_versions", {}).get(r.get("document_id"))
                    == versions.get(r.get("document_id"))
                    and r.get("document_id") in versions
                ],
            }
            saved_resolution = {**saved_resolution, "referent_candidates": offered}
        result.append(
            SimpleNamespace(
                id=turn.id,
                question=turn.question,
                resolved_question=turn.resolved_question,
                status=turn.status,
                resolution=saved_resolution,
                response={"citations": citations},
            )
        )
    # Upgrade an older pending restatement without editing the saved historical turn.
    # Only traverse the same uninterrupted restatement chain, never a completed topic.
    if result and result[0].status == "needs_clarification":
        pending = result[0].resolution.get("pending_intent") or {}
        if pending.get("requires_restatement") and not restatement_reasons(pending):
            for older in result[1:]:
                old_pending = older.resolution.get("pending_intent") or {}
                if older.status != "needs_clarification" or not old_pending.get(
                    "requires_restatement"
                ):
                    break
                inherited = restatement_reasons(old_pending)
                if inherited:
                    result[0].resolution = {
                        **result[0].resolution,
                        "pending_intent": {
                            **pending,
                            "restatement_reasons": inherited,
                        },
                    }
                    break
    return result


async def source_snapshot(session, kb_id):
    return {
        str(doc.id): {"sha256": doc.sha256, "updated_at": doc.updated_at.isoformat()}
        for doc in await session.scalars(
            select(Document).where(
                Document.knowledge_base_id == kb_id, Document.status == "processed"
            )
        )
    }


async def finish_turn(
    cid, turn_id, revision, owner_id, status, resolution, response, error, initial_versions=None
):
    async with SessionLocal() as session:
        conversation = await owned_conversation(session, cid, owner_id, lock=True)
        if conversation.active_turn_id != turn_id or conversation.revision != revision:
            raise HTTPException(409, "此轮已失去执行权，迟到结果未写入对话。")
        turn = await session.get(ConversationTurn, turn_id)
        versions = {}
        if response:
            ids = [UUID(c["document_id"]) for c in response.get("citations", [])]
            for doc in await session.scalars(
                select(Document).where(
                    Document.id.in_(ids),
                    Document.knowledge_base_id == conversation.knowledge_base_id,
                )
            ):
                version = {
                    "sha256": doc.sha256,
                    "updated_at": doc.updated_at.isoformat(),
                }
                # A concurrent re-index may change evidence during generation. Never label
                # the old response with the new document's version.
                if (initial_versions or {}).get(str(doc.id)) == version:
                    versions[str(doc.id)] = version
        turn.status, turn.resolution = status, resolution
        turn.resolved_question = (
            resolution.get("question") if resolution.get("status") != "clarify" else None
        )
        turn.response, turn.error, turn.source_versions = response, error, versions
        turn.finished_at = utc_now()
        from knowflow.app.memory_store import remember_candidate
        from knowflow.app.task_memory import finish as finish_task

        await finish_task(session, turn, status, resolution, response, versions)
        await remember_candidate(session, conversation, turn)
        # A validated multi-step plan is a natural task trigger; saving a checkpoint
        # performs no remote work and must commit atomically with the finished turn.
        plan = ((response or {}).get("query_plan") or {}).get("task_plan")
        if (
            status == "completed"
            and plan
            and plan.get("version") == "quoted-constraints-static-plan-v2-source-ids"
        ):
            from knowflow.app.memory_models import TaskAttempt
            from knowflow.app.task_memory import checkpoint

            resumed = await session.scalar(
                select(TaskAttempt.id).where(TaskAttempt.turn_id == turn.id)
            )
            if resumed is None:
                session.add(checkpoint(conversation, turn))
        conversation.active_turn_id, conversation.active_until = None, None
        conversation.updated_at = turn.finished_at
        if status == "completed":
            conversation.context_state = {
                "last_completed_turn_id": str(turn_id),
                "source_document_ids": list(versions),
                "note": "只保存引用身份，不作为本轮事实证据。",
            }
        await session.commit()
        return turn_view(turn)


@router.post("/conversations/{cid}/turns")
async def query_conversation(cid: UUID, body: TurnRequest, user: UserDep, session: SessionDep):
    conversation, turn, fresh = await claim_turn(cid, body, user.id, session)
    if not fresh:
        return turn_view(turn)
    owner_id, turn_id, sequence, kb_id = (
        user.id,
        turn.id,
        turn.sequence,
        conversation.knowledge_base_id,
    )
    resolution, response, error = {}, None, None
    initial_versions = {}
    status = "failed"
    budget = RoundBudget(get_settings())
    budget_token = CURRENT.set(budget)
    try:
        async with asyncio.timeout(budget.remaining()):
            history = await recent_intents(session, conversation)
            initial_versions = await source_snapshot(session, kb_id)
            current_source = await resolve_document_reference(session, kb_id, body.question)
            from knowflow.app.memory_store import memory_intents, recall

            memory_question = body.question
            if body.task_id:
                from knowflow.app.memory_models import TaskMemory

                saved_task = await session.get(TaskMemory, body.task_id)
                raw_inputs = [
                    r["question"]
                    for r in saved_task.state.get("user_inputs", [])
                    if not str(r.get("turn_id", "")).startswith("memory:")
                ]
                memory_question = "\n".join([saved_task.state["goal"], *raw_inputs, body.question])
            memories, memory_trace, style = await recall(
                session, owner_id, kb_id, memory_question, get_settings()
            )
            if any(c["key"] != "response_style" for c in memory_trace["conflicts"]):
                resolution = {
                    "status": "clarify",
                    "question": "记忆名称对应多个对象，请更正记忆或写出完整对象。",
                    "reason": "memory_alias_conflict",
                }
            elif body.task_id:
                from knowflow.app.task_memory import resume_context

                resolution = await resume_context(
                    session, body.task_id, body.question, initial_versions, memory_intents(memories)
                )
                if resolution.get("intent_context"):
                    clues = memory_intents(memories)
                    resolution["intent_context"]["referenced_user_inputs"].extend(
                        {"turn_id": str(row.id), "question": row.question} for row in clues
                    )
                    resolution["reference_turn_ids"].extend(str(row.id) for row in clues)
            else:
                from knowflow.app.conversation_context import VERSION as CONTEXT_VERSION

                named_resolution = named_document_resolution(
                    body.question, current_source, CONTEXT_VERSION
                )
                followup_resolution = (
                    prior_document_followup(
                        body.question,
                        await prior_document_ids(session, cid),
                        CONTEXT_VERSION,
                    )
                    if (
                        current_source.trace["status"] == "unscoped"
                        and not current_source.trace.get("unknown_title_token")
                        and not aggregate_list_question(body.question)
                        and (not history or history[0].status == "completed")
                    )
                    else None
                )
                if named_resolution is not None:
                    resolution = named_resolution
                elif followup_resolution is not None:
                    resolution = followup_resolution
                else:
                    kwargs = (
                        {"document_reference": current_source.trace}
                        if (current_source.trace["status"] != "unscoped")
                        else {}
                    )
                    resolution = await resolve_question(
                        body.question,
                        [*history, *memory_intents(memories)],
                        get_settings(),
                        **kwargs,
                    )
            if (
                resolution.get("status") == "clarify"
                and resolution.get("reason") != "memory_alias_conflict"
                and not body.task_id
                and not resolution.get("partial_lookup")
                and not resolution.get("candidate_lookup")
                and aggregate_list_question(body.question)
            ):
                # "列出/哪些/这些项目" asks across the knowledge base, not for one referent.
                resolution = {
                    **resolution,
                    "status": "standalone",
                    "question": body.question,
                    "routing_reason": "aggregate_list_search",
                    "request_kind": "fact_lookup",
                    "note": "跨文档列举类问题先检索，不按指代澄清。",
                }
            if (
                resolution.get("status") == "clarify"
                and resolution.get("reason") != "memory_alias_conflict"
                and "记忆" not in str(resolution.get("question", ""))
                and current_source.trace.get("unknown_title_token")
                and not body.task_id
                and re.search(
                    r"对象|哪一|哪份|哪个|名称|文件",
                    str(resolution.get("question", "")),
                )
            ):
                # The name is not in the file catalog: search the corpus and report
                # the result instead of asking the user to confirm an object first.
                resolution = {
                    **resolution,
                    "status": "standalone",
                    "question": body.question,
                    "routing_reason": "unknown_name_searched_first",
                    "request_kind": "fact_lookup",
                    "note": "名称不在文件清单中；先按原问题检索并如实报告，不直接要求澄清。",
                }
            resolution["current_document_reference"] = current_source.trace
            resolution["memory_context"] = memory_trace
            memory_trace["referenced_ids"] = [
                ref.split(":")[1]
                for ref in resolution.get("reference_turn_ids", [])
                if ref.startswith("memory:")
            ]
            if style:
                resolution["response_style"] = style
            if not body.task_id:
                from knowflow.app.referent_candidates import conditional_plan, find_options

                options = await find_options(
                    session, kb_id, body.question, history, resolution, get_settings()
                )
                if options:
                    resolution["referent_candidates"] = options
                    current_versions = await source_snapshot(session, kb_id)
                    stable = all(
                        current_versions.get(k) == v == initial_versions.get(k)
                        for k, v in options.get("source_versions", {}).items()
                    )
                    if not stable:
                        options.update(status="confirm_options", reason="source_version_changed")
                    if options["status"] in {"conditional_unique", "conditional_candidate"}:
                        plan = conditional_plan(body.question, options, history)
                        if plan:
                            resolution["candidate_lookup"] = plan
                            resolution["question"] = (
                                "本轮按下方当前材料中的候选作条件理解。如不是你指的对象，请更正名称。"
                            )
                    elif options.get("options"):
                        resolution["question"] = "已找到下方有来源的候选，请明确你指的是哪一个。"
            if resolution["status"] == "clarify":
                resolution["clarification_hints"] = await document_hints(session, kb_id)
            await session.rollback()
            if resolution["status"] == "clarify":
                status = "needs_clarification"
                response = QueryResponse(
                    answer=resolution["question"],
                    refused=True,
                    strategy=body.strategy,
                    citations=[],
                    clarification=resolution["clarification_hints"],
                    referent_candidates=resolution.get("referent_candidates"),
                ).model_dump(mode="json")
                if resolution.get("candidate_lookup") or resolution.get("partial_lookup"):
                    plan = resolution.get("candidate_lookup") or resolution["partial_lookup"]
                    query = QueryRequest(
                        **body.model_dump(exclude={"request_id", "expected_revision"})
                    )
                    query.question = plan["compiled_query"]
                    query._intent_context = plan["intent_context"]
                    query._request_kind = resolution.get("request_kind")
                    query._coverage = resolution.get("coverage")
                    query._context_document_ids = await prior_document_ids(session, cid)
                    query._allow_context_document = allows_prior_document(body.question, resolution)
                    if plan.get("source_document_ids"):
                        query._source_document_ids = tuple(
                            UUID(d) for d in plan["source_document_ids"]
                        )
                    query._response_style = style
                    execution = {"status": "failed", "full_request_complete": False}
                    result = None
                    try:
                        result = await query_knowledge_base(
                            kb_id, query, SimpleNamespace(id=owner_id), session
                        )
                        execution["status"] = (
                            "answered" if not result.refused and result.citations else "refused"
                        )
                    except Exception as exc:
                        execution["error_type"] = type(exc).__name__
                    resolution["partial_lookup_execution"] = execution
                    answered = execution["status"] == "answered"
                    english = not any("\u4e00" <= c <= "\u9fff" for c in body.question)
                    if result is not None and (answered or result.refused):
                        response = result.model_dump(mode="json")
                        prefix = "Part that could be checked:" if english else "可先核对的部分："
                        partial_text = result.answer
                    else:
                        prefix = "Independent part:" if english else "可独立查询的部分："
                        partial_text = (
                            "This part did not produce a verifiable answer in this attempt."
                            if english
                            else "这部分本轮未能返回可核对的答案，未自动重试。"
                        )
                    missing = "Still needs clarification:" if english else "仍需补充："
                    response.update(
                        answer=f"{prefix}\n\n{partial_text}\n\n{missing}\n\n{resolution['question']}",
                        refused=not answered,
                        clarification=resolution["clarification_hints"],
                        referent_candidates=resolution.get("referent_candidates"),
                        partial_answer={
                            "version": plan["version"],
                            **execution,
                            "question": plan["question"],
                            "current_scope": plan["current_scope"],
                            "blocked_scopes": plan["blocked_scopes"],
                            "validation_scope": "independent_part_only",
                        },
                    )
            else:
                query = QueryRequest(**body.model_dump(exclude={"request_id", "expected_revision"}))
                query.question = resolution["question"]
                query._intent_context = resolution.get("intent_context")
                query._request_kind = resolution.get("request_kind")
                query._coverage = resolution.get("coverage")
                query._context_document_ids = await prior_document_ids(session, cid)
                query._allow_context_document = allows_prior_document(body.question, resolution)
                if current_source.document_ids is not None:
                    query._source_document_ids = current_source.document_ids
                confirmed_options = (query._intent_context or {}).get("referent_options", [])
                if (
                    current_source.document_ids is None
                    and len(confirmed_options) == 1
                    and confirmed_options[0].get("document_id")
                ):
                    query._source_document_ids = (UUID(confirmed_options[0]["document_id"]),)
                query._response_style = style
                result = await query_knowledge_base(
                    kb_id, query, SimpleNamespace(id=owner_id), session
                )
                response, status = result.model_dump(mode="json"), "completed"
    except asyncio.CancelledError:
        await session.rollback()
        resolution["round_budget"] = budget.snapshot()
        await asyncio.shield(
            finish_turn(
                cid,
                turn_id,
                sequence,
                owner_id,
                "interrupted",
                resolution,
                None,
                {"code": "cancelled", "message": "请求中断，未确认回答完成。"},
            )
        )
        raise
    except Exception as exc:
        code = (
            "timeout"
            if isinstance(exc, TimeoutError)
            else f"HTTP_{exc.status_code}"
            if isinstance(exc, HTTPException)
            else type(exc).__name__
        )
        logger.warning(
            "turn failed: conversation=%s question=%r type=%s detail=%s",
            cid,
            body.question[:120],
            type(exc).__name__,
            str(getattr(exc, "detail", exc))[:400],
        )
        error = {
            "code": code,
            "message": "本轮未完成，未返回未经核验的答案；可查看记录后重新提问。",
        }
    finally:
        snapshot = budget.snapshot()
        resolution["round_budget"] = snapshot
        if response is not None:
            response["round_budget"] = snapshot
        CURRENT.reset(budget_token)
    await session.rollback()  # Release read transactions before fenced completion.
    return await finish_turn(
        cid, turn_id, sequence, owner_id, status, resolution, response, error, initial_versions
    )
