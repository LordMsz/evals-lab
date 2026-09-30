"""Platform-style evals, hands-on: Arize Phoenix datasets + experiments (runs fully local).

    pip install arize-phoenix && phoenix serve            # http://localhost:6006
    python evals/platforms/phoenix_experiment.py           # uploads dataset, runs v1 and v2

Same graders as every other harness; Phoenix stores the dataset (versioned), each run as an
experiment, per-example scores, and lets you compare experiments side by side in the UI.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phoenix.client import Client  # noqa: E402

from app.triage import triage  # noqa: E402
from evals.checks import deterministic_checks, load_cases  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402

client = Client(base_url=os.getenv("PHOENIX_URL", "http://localhost:6006"))
cases = load_cases()
dataset = client.datasets.create_dataset(
    name=f"support-triage-golden-{os.getenv('RUN_TAG', 'v1')}",
    inputs=[{"ticket": c["ticket"]} for c in cases],
    outputs=[{"expect": c["expect"]} for c in cases],
    metadata=[{"id": c["id"], "rubric": c.get("rubric"), "tags": c.get("tags", [])} for c in cases],
    dataset_description="12 golden support tickets (eval-lab)",
)
BY_TICKET = {c["ticket"]: c for c in cases}


def make_task(version):
    def task(input):                       # Phoenix binds params by name: input/expected/metadata
        r = triage(input["ticket"], version=version)
        r["trace"].pop("otel", None)
        return r
    return task


def deterministic(input, output):
    checks = deterministic_checks(BY_TICKET[input["ticket"]], output)
    failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
    return {"score": 1.0 - len(failed) / len(checks), "label": "fail" if failed else "pass",
            "explanation": "; ".join(failed) or "all checks passed"}


def grounded(input, output):
    case = BY_TICKET[input["ticket"]]
    v = judge_grounded(case["ticket"], output, case.get("rubric"))
    return {"score": float(v["pass"]), "label": "pass" if v["pass"] else "fail",
            "explanation": v["reason"]}


for version in ("v1", "v2"):
    client.experiments.run_experiment(
        dataset=dataset, task=make_task(version), evaluators=[deterministic, grounded],
        experiment_name=f"triage-{version}", experiment_metadata={"prompt_version": version},
        print_summary=True)
print("Compare in UI:", client.experiments.get_dataset_experiments_url(dataset_id=dataset.id))
