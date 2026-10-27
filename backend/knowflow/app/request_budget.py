"""Shared estimates and provider output-field semantics, never advertised model limits."""

from urllib.parse import urlparse


def estimate_messages(messages, counter):
    return 3 + sum(counter.count(m["content"]) + counter.count(m["role"]) + 8 for m in messages)


def output_limit(model, base_url, configured="auto"):
    host = urlparse(base_url or "").hostname
    model = model or ""
    known_completion = (
        host
        in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
        and model.startswith(("qwen3.7", "qwen3.8"))
    ) or (host == "api.xiaomimimo.com" and model.startswith("mimo-v2.6"))
    parameter = (
        configured
        if configured != "auto"
        else ("max_completion_tokens" if known_completion else "max_tokens")
    )
    known_scope = (known_completion and parameter == "max_completion_tokens") or (
        host == "api.deepseek.com" and parameter == "max_tokens"
    )
    return parameter, "reasoning_and_answer" if known_scope else "unverified"
