from types import SimpleNamespace

import pytest
from knowflow.app import providers
from knowflow.app.providers import conversational_answer, generate_answer


def test_unwraps_response_envelope_without_showing_internal_fields() -> None:
    assert (
        conversational_answer(
            '```json\n{"answer":"根据原文，使用余弦距离。[1]", "citations": ["internal"]}\n```'
        )
        == "根据原文，使用余弦距离。[1]"
    )


def test_preserves_normal_prose_tables_and_requested_code_examples() -> None:
    text = '配置示例：\n\n```json\n{"port":8080}\n```\n\n| A | B |\n|---|---|\n|1|2|'
    assert conversational_answer(text) == text


@pytest.mark.asyncio
async def test_answer_generation_uses_chat_format_and_evidence_bound_charts(monkeypatch) -> None:
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content='{"answer":"正常回答。[KF1]"}'))
            ]
        )

    monkeypatch.setattr(
        providers,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    answer = await generate_answer(
        "请用图表说明",
        [{"id": "11111111-1111-1111-1111-111111111111", "content": "原文内容"}],
        SimpleNamespace(
            llm_model="test", llm_base_url=None, structured_mimo_thinking="provider_default"
        ),
    )

    assert answer == "正常回答。[11111111-1111-1111-1111-111111111111]"
    supplied = calls[0]["messages"][1]["content"]
    assert "[KF1]" in supplied and "11111111-1111-1111-1111-111111111111" not in supplied
    assert "response_format" not in calls[0]
    prompt = calls[0]["messages"][0]["content"]
    assert "not a JSON object" in prompt
    assert "Every diagram edge and every numerical value" in prompt
    assert "untrusted data" in prompt
