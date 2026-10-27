"""Owner-scoped read-only snapshots, excluding later turns and model judgments."""

from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from knowflow.app.api import UserDep
from knowflow.app.conversation_api import owned_conversation
from knowflow.app.conversation_review import MAX_TURNS, digest, make_bundle
from knowflow.app.models import ConversationTurn, utc_now
from knowflow.app.security import SessionDep
from knowflow.app.task_alignment_api import saved_input

router = APIRouter(prefix="/api/v1", tags=["conversation review export"])


@router.get("/conversations/{cid}/turns/{tid}/review-export")
async def export_review(cid: UUID, tid: UUID, user: UserDep, session: SessionDep):
    conversation = await owned_conversation(session, cid, user.id)
    target = await session.get(ConversationTurn, tid)
    if target is None or target.conversation_id != cid:
        raise HTTPException(404, "Turn not found")
    rows = list(
        await session.scalars(
            select(ConversationTurn)
            .where(
                ConversationTurn.conversation_id == cid,
                ConversationTurn.sequence <= target.sequence,
            )
            .order_by(ConversationTurn.sequence)
            .limit(MAX_TURNS + 1)
            .execution_options(populate_existing=True)
        )
    )
    if len(rows) > MAX_TURNS:
        raise HTTPException(413, "复核包最多100轮，未截断历史；可先选择更早的轮次。")
    if not rows or rows[-1].status == "pending":
        raise HTTPException(409, "所选轮次尚在执行，请结束后导出固定快照。")
    exported = []
    for turn in rows:
        inputs = await saved_input(session, conversation, turn)
        response = turn.response or {}
        citations = [
            {
                "id": f"{turn.id}:c{i}",
                **{
                    key: c.get(key)
                    for key in (
                        "chunk_id",
                        "document_id",
                        "filename",
                        "page_number",
                        "section",
                        "quote",
                        "source_spans",
                    )
                },
            }
            for i, c in enumerate(response.get("citations") or [], 1)
        ]
        row = {
            "id": str(turn.id),
            "sequence": turn.sequence,
            "status": turn.status,
            "question": turn.question,
            "answer": response.get("answer"),
            "refused_flag_not_semantic_label": response.get("refused"),
            "task_inputs": {
                key: inputs[key]
                for key in (
                    "requirement_sources",
                    "goal_basis",
                    "memory_identity_clues_not_requirements",
                    "resolved_retrieval_hint_not_authority",
                )
            },
            "saved_task_goal_source_turn_id": (
                (inputs.get("active_task") or {}).get("goal") or {}
            ).get("turn_id"),
            "citations": citations,
            "created_at": turn.created_at.isoformat(),
            "finished_at": turn.finished_at.isoformat() if turn.finished_at else None,
        }
        row["turn_sha256"] = digest(row)
        exported.append(row)
    snapshot = {
        "conversation_id": str(cid),
        "knowledge_base_id": str(conversation.knowledge_base_id),
        "target_turn_id": str(tid),
        "through_sequence": target.sequence,
        "history_policy": "all_saved_turns_through_target_no_future",
        "turns": exported,
    }
    try:
        bundle = make_bundle(snapshot, utc_now().isoformat())
    except ValueError as exc:
        raise HTTPException(413 if "exceeds" in str(exc) else 409, str(exc)) from exc
    from fastapi.responses import JSONResponse

    return JSONResponse(bundle, headers={"Cache-Control": "no-store"})
