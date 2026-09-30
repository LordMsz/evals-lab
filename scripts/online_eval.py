"""Online evaluation sketch: score a SAMPLE of production traffic with the same judge,
write verdicts into the trace (gen_ai.evaluation.result), and alert on drift.

    TRACE_FILE=reports/traces.jsonl python scripts/online_eval.py --n 40 --sample 0.25
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006 python scripts/online_eval.py   # Phoenix
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import telemetry  # noqa: E402
from app.triage import triage  # noqa: E402
from evals.checks import load_cases  # noqa: E402
from evals.judge import judge_grounded  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40, help="simulated production requests")
ap.add_argument("--sample", type=float, default=0.25, help="fraction sent to the judge")
ap.add_argument("--alert-below", type=float, default=0.85)
args = ap.parse_args()

tickets = [c["ticket"] for c in load_cases()]   # stand-in for live traffic
rng = random.Random(1)
judged = passed = 0
for i in range(args.n):
    ticket = rng.choice(tickets)
    result = triage(ticket)
    if rng.random() < args.sample:                       # sampling keeps judge cost bounded
        with telemetry.evaluation_span(result["trace"]["otel"], "grounded") as span:
            v = judge_grounded(ticket, result)
            telemetry.record_evaluation(span, "grounded", v["pass"], v["reason"])
        judged += 1
        passed += v["pass"]
rate = passed / judged if judged else 1.0
print(json.dumps({"requests": args.n, "judged": judged, "grounded_pass_rate": round(rate, 3)}))
if rate < args.alert_below:
    print(f"ALERT: grounded pass rate {rate:.2f} < {args.alert_below}")   # -> pager/Slack
