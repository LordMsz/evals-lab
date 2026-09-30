"""Approach A - 'evals are just tests': plain pytest, no eval framework.

Two layers, like a normal test pyramid:
  1. per-case tests  (deterministic graders + one LLM judge)  -> readable failures in CI
  2. suite gates     (aggregate pass-rates vs thresholds)     -> tolerate non-determinism

TRIALS=3 runs each case 3x; gates then report pass@k (any trial passed) and
pass^k (all trials passed - the consistency bar your users actually feel).

    pytest evals/test_plain_pytest.py -m "not llm_judge"   # fast PR tier
    TRIALS=3 pytest evals/test_plain_pytest.py             # nightly tier
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest

from app import llm
from app.triage import triage
from evals.checks import ROOT, deterministic_checks, load_cases, retrieval_recall
from evals.judge import judge_grounded

CASES = load_cases()
TRIALS = int(os.getenv("TRIALS", "1"))
GATE_DETERMINISTIC = float(os.getenv("GATE_DETERMINISTIC", "0.9"))
GATE_JUDGE = float(os.getenv("GATE_JUDGE", "0.85"))
RESULTS: dict = defaultdict(dict)


@lru_cache(maxsize=None)
def run(case_id: str, trial: int) -> dict:
    case = next(c for c in CASES if c["id"] == case_id)
    return triage(case["ticket"])


def _ids(cases):
    return [c["id"] for c in cases]


@pytest.mark.parametrize("trial", range(TRIALS))
@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_deterministic(case, trial):
    result = run(case["id"], trial)
    checks = deterministic_checks(case, result)
    RESULTS[case["id"]].setdefault("det", []).append(all(ok for _, ok, _ in checks))
    RESULTS[case["id"]]["recall"] = retrieval_recall(case, result)
    failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
    assert not failed, "\n".join(failed)


@pytest.mark.llm_judge
@pytest.mark.parametrize("trial", range(TRIALS))
@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_grounded_judge(case, trial):
    result = run(case["id"], trial)
    verdict = judge_grounded(case["ticket"], result, case.get("rubric"))
    RESULTS[case["id"]].setdefault("judge", []).append(verdict["pass"])
    assert verdict["pass"], f"{verdict['reason']}\nanswer: {result['output'].get('answer')}"


def _rates(key):
    rows = [r[key] for r in RESULTS.values() if key in r]
    if not rows:
        return None
    return {
        "cases": len(rows),
        "pass_rate": sum(sum(t) for t in rows) / sum(len(t) for t in rows),
        "pass@k": sum(any(t) for t in rows) / len(rows),
        "pass^k": sum(all(t) for t in rows) / len(rows),
    }


@pytest.mark.gate
def test_zz_suite_gates():
    """Runs last (zz). Fails the build when aggregate quality drops below thresholds."""
    det, jud = _rates("det"), _rates("judge")
    recall = [r["recall"] for r in RESULTS.values() if r.get("recall") is not None]
    summary = {
        "app_model": llm.model_id("app"), "judge_model": llm.model_id("judge"),
        "version": os.getenv("APP_VERSION", "v1"), "trials": TRIALS,
        "deterministic": det, "judge": jud,
        "retrieval_recall": sum(recall) / len(recall) if recall else None,
        "llm_calls": llm.USAGE.calls, "tokens_in": llm.USAGE.input_tokens,
        "tokens_out": llm.USAGE.output_tokens,
        "per_case": {k: v for k, v in RESULTS.items()},
    }
    out = Path(os.getenv("REPORT", ROOT / "reports" / "pytest_summary.json"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_case"}, indent=2))

    if det:
        assert det["pass^k"] >= GATE_DETERMINISTIC, f"deterministic pass^k {det['pass^k']:.2f}"
    if jud:
        assert jud["pass_rate"] >= GATE_JUDGE, f"judge pass rate {jud['pass_rate']:.2f}"
