"""LangSmith pytest integration: ordinary pytest tests that also log to a LangSmith experiment.
    LANGSMITH_API_KEY=... pytest evals/platforms/langsmith_suite.py       # results -> LangSmith
    LANGSMITH_TEST_TRACKING=false pytest evals/platforms/langsmith_suite.py  # offline
The alternative API is langsmith.Client().evaluate(target, data="dataset", evaluators=[...]).
"""
import pytest
from langsmith import testing as t

from app.triage import triage
from evals.checks import deterministic_checks, load_cases
from evals.judge import judge_grounded

CASES = load_cases()


@pytest.mark.langsmith
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_triage(case):
    t.log_inputs({"ticket": case["ticket"]})
    t.log_reference_outputs(case["expect"])
    result = triage(case["ticket"])
    t.log_outputs(result["output"])
    checks = deterministic_checks(case, result)
    t.log_feedback(key="deterministic", score=float(all(ok for _, ok, _ in checks)))
    v = judge_grounded(case["ticket"], result, case.get("rubric"))
    t.log_feedback(key="grounded", score=float(v["pass"]))
    assert all(ok for _, ok, _ in checks) and v["pass"], v["reason"]
