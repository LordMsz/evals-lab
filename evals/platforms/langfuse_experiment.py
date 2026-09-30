"""Langfuse experiment runner (SDK v3+/v4). Needs a Langfuse project (cloud or self-hosted):
    export LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... LANGFUSE_HOST=https://cloud.langfuse.com
    python evals/platforms/langfuse_experiment.py

What Langfuse adds over the plain harness: every item becomes a trace linked to the experiment,
scores land on those traces, experiments are compared in the UI, and the SAME judge can be
re-used as a managed server-side evaluator on production observations (configured in the UI).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from langfuse import Evaluation, get_client  # noqa: E402

from app.triage import triage  # noqa: E402
from evals.checks import deterministic_checks, load_cases  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402

langfuse = get_client()
CASES = {c["ticket"]: c for c in load_cases()}
data = [{"input": c["ticket"], "expected_output": c["expect"], "metadata": {"id": c["id"]}}
        for c in CASES.values()]
# In a real project the golden set lives in Langfuse: langfuse.get_dataset("support-triage")


def make_task(version):
    def task(*, item, **kwargs):
        ticket = item["input"] if isinstance(item, dict) else item.input
        return triage(ticket, version=version)
    return task


def deterministic(*, input, output, **kwargs):
    checks = deterministic_checks(CASES[input], output)
    failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
    return Evaluation(name="deterministic", value=0.0 if failed else 1.0,
                      comment="; ".join(failed) or "ok")


def grounded(*, input, output, **kwargs):
    v = judge_grounded(input, output, CASES[input].get("rubric"))
    return Evaluation(name="grounded", value=float(v["pass"]), comment=v["reason"])


def pass_rate(*, item_results, **kwargs):                     # run-level evaluator = CI gate
    vals = [e.value for r in item_results for e in r.evaluations if e.name == "grounded"]
    return Evaluation(name="grounded_pass_rate", value=sum(vals) / len(vals))


for version in ("v1", "v2"):
    result = langfuse.run_experiment(
        name="support-triage", run_name=f"triage-{version}", data=data,
        task=make_task(version), evaluators=[deterministic, grounded],
        run_evaluators=[pass_rate], metadata={"prompt_version": version})
    print(result.format())
langfuse.flush()
