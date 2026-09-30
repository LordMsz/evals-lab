"""MLflow 3 GenAI evaluation (Apache-2.0, fully self-hostable; managed on Databricks and SageMaker).

    pip install "mlflow>=3.16"
    python evals/platforms/mlflow_eval.py            # writes to ./mlruns.db (SQLite), no server
    mlflow ui --backend-store-uri sqlite:///mlruns.db  # http://localhost:5000 -> Experiments / Evaluations

Built-in judges (Correctness, Guidelines, RetrievalGroundedness, ...) need a judge model
(e.g. "openai:/gpt-5-mini"); here we plug in the lab's own graders as custom @scorer functions.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mlflow  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import scorer  # noqa: E402

from app.triage import triage  # noqa: E402
from evals.checks import deterministic_checks, load_cases  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns.db"))
mlflow.set_experiment("support-triage")
CASES = {c["id"]: c for c in load_cases()}
DATA = [{"inputs": {"case_id": cid, "ticket": c["ticket"]}, "expectations": c["expect"]}
        for cid, c in CASES.items()]


@scorer
def deterministic(inputs, outputs) -> Feedback:
    checks = deterministic_checks(CASES[inputs["case_id"]], outputs)
    failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
    return Feedback(value=not failed, rationale="; ".join(failed) or "all checks passed")


@scorer
def grounded(inputs, outputs) -> Feedback:
    case = CASES[inputs["case_id"]]
    v = judge_grounded(case["ticket"], outputs, case.get("rubric"))
    return Feedback(value=v["pass"], rationale=v["reason"])


for version in ("v1", "v2"):
    def predict_fn(case_id, ticket, _v=version):
        r = triage(ticket, version=_v)
        r["trace"].pop("otel", None)
        return r
    with mlflow.start_run(run_name=f"triage-{version}"):
        mlflow.log_param("prompt_version", version)
        res = mlflow.genai.evaluate(data=DATA, predict_fn=predict_fn,
                                    scorers=[deterministic, grounded])
        print(version, {k: round(v, 3) for k, v in res.metrics.items()})
