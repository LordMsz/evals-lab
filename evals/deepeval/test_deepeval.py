"""Approach C - DeepEval: pytest-native, metric objects, built-in RAG/agent metrics.

    deepeval test run evals/deepeval/test_deepeval.py     # nicer report
    pytest evals/deepeval/test_deepeval.py                # also works

Fake backend  -> our custom metrics + built-in ToolCorrectness (deterministic, no LLM).
Real backend  -> additionally DeepEval's research-backed LLM metrics (G-Eval, Faithfulness)
                 using DeepEval's own judge prompts.
"""
from __future__ import annotations

import os

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall

from app import llm
from app.triage import triage
from evals.checks import deterministic_checks, load_cases
from evals.judge import judge_grounded

CASES = load_cases()
BACKEND = os.getenv("LLM_BACKEND", "fake")


class LabJudgeModel(DeepEvalBaseLLM):
    """Route DeepEval's judge calls through our provider layer (any backend, incl. fake).
    Without this DeepEval defaults to OpenAI - even ToolCorrectness instantiates it."""

    def __init__(self):
        super().__init__(model=llm.model_id("judge"))

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None, **kw):
        text = llm.complete("You are an evaluation assistant. Reply in JSON when asked.",
                            prompt, role="judge")
        return schema.model_validate(llm.parse_json(text)) if schema else text

    async def a_generate(self, prompt: str, schema=None, **kw):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return llm.model_id("judge")


class SharedChecksMetric(BaseMetric):
    """Wraps the shared deterministic graders as a DeepEval metric."""

    def __init__(self, case: dict, result: dict, threshold: float = 1.0):
        self.case, self.result, self.threshold = case, result, threshold
        self.async_mode, self.include_reason = False, True

    def measure(self, test_case: LLMTestCase, *a, **kw) -> float:
        checks = deterministic_checks(self.case, self.result)
        self.score = sum(ok for _, ok, _ in checks) / len(checks)
        self.reason = "; ".join(f"{n}: {d}" for n, ok, d in checks if not ok) or "ok"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *a, **kw):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Shared deterministic checks"


class GroundedJudgeMetric(SharedChecksMetric):
    """Wraps our calibrated binary judge (evals/judge.py)."""

    def measure(self, test_case: LLMTestCase, *a, **kw) -> float:
        v = judge_grounded(self.case["ticket"], self.result, self.case.get("rubric"))
        self.score, self.reason, self.success = float(v["pass"]), v["reason"], v["pass"]
        return self.score

    @property
    def __name__(self):
        return "Grounded (custom judge)"


def _native_llm_metrics():
    """DeepEval's own LLM-graded metrics - only when a real model is configured."""
    if BACKEND == "fake":
        return []
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams
    judge = LabJudgeModel()  # or a string like "gpt-5" to use DeepEval's native OpenAI client
    return [
        FaithfulnessMetric(threshold=0.8, model=judge),
        GEval(name="No false promises", model=judge, threshold=0.7,
              evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT,
                                 SingleTurnParams.RETRIEVAL_CONTEXT],
              evaluation_steps=[
                  "List every promise or commitment made in the actual output.",
                  "Check each against the retrieval context.",
                  "Heavily penalize refunds, credits or delivery dates not in the context."]),
    ]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_triage(case):
    result = triage(case["ticket"])
    out, trace = result["output"], result["trace"]
    test_case = LLMTestCase(
        input=case["ticket"],
        actual_output=out.get("answer", ""),
        # the judge must see ALL evidence: retrieved docs AND tool results (lesson from real runs)
        retrieval_context=([c["text"] for c in trace["retrieved"]]
                           + [f"Tool {t['name']} returned: {t['result']}" for t in trace["tool_calls"]])
        or ["(nothing retrieved)"],
        tools_called=[ToolCall(name=t["name"], input_parameters=t["args"])
                      for t in trace["tool_calls"]],
        expected_tools=[ToolCall(name=case["expect"]["tool"])] if "tool" in case["expect"] else [],
        tags=case.get("tags", []),
    )
    metrics = [SharedChecksMetric(case, result), GroundedJudgeMetric(case, result),
               ToolCorrectnessMetric(threshold=1.0, model=LabJudgeModel()), *_native_llm_metrics()]
    assert_test(test_case, metrics, run_async=False)
