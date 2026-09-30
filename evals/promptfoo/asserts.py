"""promptfoo Python assertions delegating to the shared graders in evals/."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from evals.checks import deterministic_checks  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402


def deterministic(output, context):
    case, result = json.loads(context["vars"]["case"]), json.loads(output)
    checks = deterministic_checks(case, result)
    return {
        "pass": all(ok for _, ok, _ in checks),
        "score": sum(ok for _, ok, _ in checks) / len(checks),
        "reason": "; ".join(f"{n}: {d}" for n, ok, d in checks if not ok) or "all checks passed",
        "componentResults": [{"pass": ok, "score": float(ok), "reason": f"{n}: {d}"}
                             for n, ok, d in checks],
    }


def grounded(output, context):
    case, result = json.loads(context["vars"]["case"]), json.loads(output)
    v = judge_grounded(case["ticket"], result, case.get("rubric"))
    return {"pass": v["pass"], "score": float(v["pass"]), "reason": v["reason"]}
