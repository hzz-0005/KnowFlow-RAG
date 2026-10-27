"""Deleting one conversation: owner-only, turn cascade, lease guard."""

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from knowflow.app.db import SessionLocal
from knowflow.app.models import Conversation, utc_now
from test_conversations import create, workspace


@pytest.mark.asyncio
async def test_owner_deletes_conversation_and_it_leaves_the_list(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]

        deleted = await client.delete(f"/api/v1/conversations/{cid}", headers=auth[0])

        assert deleted.status_code == 204
        restored = await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])
        assert restored.status_code == 404
        items = (
            await client.get(
                f"/api/v1/knowledge-bases/{kb}/conversations", headers=auth[0]
            )
        ).json()["items"]
        assert all(row["id"] != cid for row in items)


@pytest.mark.asyncio
async def test_another_user_cannot_delete_the_conversation(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]

        denied = await client.delete(f"/api/v1/conversations/{cid}", headers=auth[1])

        assert denied.status_code == 404
        restored = await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])
        assert restored.status_code == 200


@pytest.mark.asyncio
async def test_a_conversation_with_an_active_turn_is_not_deleted(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            row = await session.get(Conversation, UUID(cid))
            row.active_turn_id = uuid4()
            row.active_until = utc_now() + timedelta(minutes=2)
            await session.commit()

        refused = await client.delete(f"/api/v1/conversations/{cid}", headers=auth[0])

        assert refused.status_code == 409
        restored = await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])
        assert restored.status_code == 200
