"""OpenTelemetry instrumentation using the GenAI semantic conventions (status: Development).

Vendor-neutral: the same spans can go to Langfuse, Phoenix, MLflow, Datadog, Grafana, etc.

    TRACE_FILE=reports/traces.jsonl python -m app.triage "..."          # local JSON lines
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006 python ...        # e.g. Phoenix / Langfuse
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true             # opt-in prompt capture

Online evals attach results to the trace as `gen_ai.evaluation.result` events - so the
same judge you run in CI can score sampled production traffic.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

CAPTURE_CONTENT = os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT") == "true"


class JsonlExporter(SpanExporter):
    def __init__(self, path):
        self.path = path

    def export(self, spans):
        with open(self.path, "a") as f:
            for s in spans:
                f.write(json.dumps({
                    "name": s.name, "trace_id": f"{s.context.trace_id:032x}",
                    "span_id": f"{s.context.span_id:016x}",
                    "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
                    "attributes": dict(s.attributes),
                    "events": [{"name": e.name, "attributes": dict(e.attributes)} for e in s.events],
                }) + "\n")
        return SpanExportResult.SUCCESS


def _init():
    endpoint, path = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), os.getenv("TRACE_FILE")
    if not (endpoint or path):
        return trace.get_tracer("eval-lab")  # no-op tracer
    provider = TracerProvider(resource=Resource.create({"service.name": "support-triage"}))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(
            endpoint=endpoint.rstrip("/") + "/v1/traces")))
    if path:
        provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(path)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("eval-lab")


tracer = _init()


@contextmanager
def agent_span(name: str):
    with tracer.start_as_current_span(f"invoke_agent {name}") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", name)
        yield span


@contextmanager
def chat_span(provider: str, model: str, system: str, user: str):
    with tracer.start_as_current_span(f"chat {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", provider)
        span.set_attribute("gen_ai.request.model", model)
        if CAPTURE_CONTENT:
            span.set_attribute("gen_ai.system_instructions", system)
            span.set_attribute("gen_ai.input.messages", json.dumps(
                [{"role": "user", "parts": [{"type": "text", "content": user}]}]))
        yield span


@contextmanager
def tool_span(name: str, args: dict):
    with tracer.start_as_current_span(f"execute_tool {name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        if CAPTURE_CONTENT:
            span.set_attribute("gen_ai.tool.call.arguments", json.dumps(args))
        yield span


@contextmanager
def evaluation_span(otel_ids: dict, name: str):
    """Async/online eval: open a child span of the (already finished) agent span, so the
    judge's own LLM call and its verdict land in the same trace as the request."""
    span_ctx = trace.SpanContext(int(otel_ids["trace_id"], 16), int(otel_ids["span_id"], 16),
                                 is_remote=True, trace_flags=trace.TraceFlags(1))
    parent = trace.set_span_in_context(trace.NonRecordingSpan(span_ctx))
    with tracer.start_as_current_span(f"evaluate {name}", context=parent) as span:
        span.set_attribute("gen_ai.operation.name", "evaluate")
        yield span


def record_evaluation(span, name: str, passed: bool, explanation: str = ""):
    """Emit the verdict as a gen_ai.evaluation.result event (OTel GenAI semconv)."""
    span.add_event("gen_ai.evaluation.result", {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.score.label": "pass" if passed else "fail",
        "gen_ai.evaluation.score.value": 1.0 if passed else 0.0,
        "gen_ai.evaluation.explanation": explanation[:500],
    })
