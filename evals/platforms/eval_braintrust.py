"""Braintrust Eval(): data + task + scores.  Local, no account needed:
    braintrust eval evals/platforms/eval_braintrust.py --no-send-logs
With BRAINTRUST_API_KEY set (drop --no-send-logs) each run becomes an experiment in the UI,
diffed against the previous one; autoevals provides ready scorers (Factuality, ClosedQA, ...).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from braintrust import Eval  # noqa: E402

from app.triage import triage  # noqa: E402
from evals.checks import deterministic_checks, load_cases  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402

CASES = load_cases()
BY_TICKET = {c["ticket"]: c for c in CASES}
VERSION = os.getenv("APP_VERSION", "v1")


def deterministic(input, output, expected=None, **kw):
    checks = deterministic_checks(BY_TICKET[input], output)
    return {"name": "deterministic", "score": sum(ok for _, ok, _ in checks) / len(checks),
            "metadata": {"failed": [n for n, ok, _ in checks if not ok]}}


def grounded(input, output, **kw):
    v = judge_grounded(input, output, BY_TICKET[input].get("rubric"))
    return {"name": "grounded", "score": float(v["pass"]), "metadata": {"reason": v["reason"]}}


Eval(
    "support-triage",
    experiment_name=f"triage-{VERSION}",
    data=lambda: [{"input": c["ticket"], "expected": c["expect"], "tags": c.get("tags", [])}
                  for c in CASES],
    task=lambda input: triage(input, version=VERSION),
    scores=[deterministic, grounded],
    metadata={"prompt_version": VERSION},
)
