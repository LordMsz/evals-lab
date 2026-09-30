"""Provider-agnostic LLM client.

LLM_BACKEND   = fake | anthropic | openai | ollama   (default: fake)
APP_MODEL     = model used by the application
JUDGE_BACKEND = backend for the LLM-as-judge (default: same as LLM_BACKEND)
JUDGE_MODEL   = model used by the judge (use a stronger/different model than APP_MODEL)

The `fake` backend is deterministic (unless FAKE_NOISE > 0) and lets you exercise every
harness, report and CI gate without spending tokens. It is NOT a quality signal.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

DEFAULT_MODELS = {
    # Override with APP_MODEL / JUDGE_MODEL - model names change often.
    "anthropic": ("claude-haiku-4-5", "claude-sonnet-4-5"),
    "openai": ("gpt-6-luna", "gpt-6-luna"),
    "ollama": ("qwen2.5:7b", "qwen2.5:14b"),
    "fake": ("fake-app", "fake-judge"),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    by_role: dict = field(default_factory=dict)


USAGE = Usage()


def _backend(role: str) -> str:
    app = os.getenv("LLM_BACKEND", "fake")
    return os.getenv("JUDGE_BACKEND", app) if role == "judge" else app


def _model(role: str, backend: str) -> str:
    env = os.getenv("JUDGE_MODEL" if role == "judge" else "APP_MODEL")
    return env or DEFAULT_MODELS[backend][1 if role == "judge" else 0]


def model_id(role: str = "app") -> str:
    b = _backend(role)
    return f"{b}:{_model(role, b)}"


def complete(system: str, user: str, *, role: str = "app", temperature: float = 0.0,
             max_tokens: int = 800) -> str:
    from app import telemetry
    backend = _backend(role)
    model = _model(role, backend)
    with telemetry.chat_span(backend, model, system, user) as span:
        text, in_tok, out_tok = _call(backend, model, role, system, user, temperature, max_tokens)
        span.set_attribute("gen_ai.usage.input_tokens", in_tok)
        span.set_attribute("gen_ai.usage.output_tokens", out_tok)
        span.set_attribute("eval_lab.role", role)
        if telemetry.CAPTURE_CONTENT:
            span.set_attribute("gen_ai.output.messages", json.dumps(
                [{"role": "assistant", "parts": [{"type": "text", "content": text}]}]))
    return text


def _call(backend, model, role, system, user, temperature, max_tokens):
    t0 = time.time()
    in_tok = out_tok = 0
    if backend == "fake":
        from app.fake_model import fake_complete
        text = fake_complete(system, user)
        in_tok, out_tok = (len(system) + len(user)) // 4, len(text) // 4
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(model=model, system=system, max_tokens=max_tokens,
                                   temperature=temperature,
                                   messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in r.content if b.type == "text")
        in_tok, out_tok = r.usage.input_tokens, r.usage.output_tokens
    elif backend in ("openai", "ollama"):
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                        api_key="ollama") if backend == "ollama" else OpenAI()
        if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):   # reasoning models: no temperature
            effort = os.getenv("OPENAI_REASONING_EFFORT", "low")
            if role == "judge":
                effort = os.getenv("JUDGE_REASONING_EFFORT", effort)
            kwargs = {"reasoning_effort": effort}
        else:
            kwargs = {"temperature": temperature}
        r = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system},
                                   {"role": "user", "content": user}], **kwargs)
        text = r.choices[0].message.content or ""
        if r.usage:
            in_tok, out_tok = r.usage.prompt_tokens, r.usage.completion_tokens
    else:
        raise ValueError(f"unknown backend {backend}")

    USAGE.calls += 1
    USAGE.input_tokens += in_tok
    USAGE.output_tokens += out_tok
    USAGE.seconds += time.time() - t0
    USAGE.by_role[role] = USAGE.by_role.get(role, 0) + 1
    return text, in_tok, out_tok


def parse_json(text: str) -> dict:
    """Lenient JSON extraction - models love wrapping JSON in prose or ``` fences."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))
