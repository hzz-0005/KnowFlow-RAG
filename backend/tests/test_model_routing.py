from knowflow.app.config import Settings
from knowflow.app.providers import structured_request_options, task_route


def settings():
    return Settings(
        _env_file=None,
        llm_model="qwen3.8-flash",
        llm_complex_model="qwen3.8-max",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def test_relationship_extraction_keeps_complex_route_even_for_short_sentences():
    s = settings()
    simple = task_route("extract", s, "A uses B.")
    scoped = task_route("extract", s, "In staging A may use B. In production A must not use B.")
    assert (simple.model, simple.thinking) == ("qwen3.8-max", "enabled")
    assert simple.reason == "relation_extraction_task"
    assert (scoped.model, scoped.thinking) == ("qwen3.8-max", "enabled")
    assert task_route("answer", s, "Where is it?", 1).model == "qwen3.8-flash"
    assert task_route("answer", s, "Where is it?", 2).model == "qwen3.8-max"


def test_options_preserve_complex_reasoning_and_do_not_send_foreign_provider_parameters():
    s = settings()
    assert structured_request_options(s.llm_model, s.llm_base_url, s, thinking="disabled") == {
        "extra_body": {"enable_thinking": False}
    }
    assert structured_request_options(
        s.llm_complex_model, s.llm_base_url, s, thinking="enabled"
    ) == {"extra_body": {"enable_thinking": True}, "reasoning_effort": "low"}
    assert structured_request_options(
        "deepseek-flash", "https://api.deepseek.com", s, thinking="enabled"
    ) == {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "low"}
    assert (
        structured_request_options("qwen3.8-max", "https://example.com", s, thinking="enabled")
        == {}
    )


def test_independent_judge_does_not_use_chat_model_or_disable_thinking():
    s = settings()
    s.eval_judge_api_key = "fixture"
    s.eval_judge_base_url = "https://api.deepseek.com"
    s.eval_judge_model = "deepseek-flash"
    route = task_route("review", s)
    assert (route.model, route.thinking, route.reason) == (
        "deepseek-flash",
        "enabled",
        "independent_review",
    )
