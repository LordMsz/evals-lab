"""Inspect exits 0 even when accuracy collapses (it's a measurement tool, not a test runner).
Gate on the log instead:  python scripts/inspect_gate.py --min deterministic=0.9 grounded=0.85"""
import argparse
import sys

from inspect_ai.log import list_eval_logs, read_eval_log

ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", default="reports/inspect-logs")
ap.add_argument("--min", nargs="+", default=["deterministic=0.9", "grounded=0.85"])
args = ap.parse_args()

log = read_eval_log(list_eval_logs(args.log_dir)[0])   # newest first
scores = {s.name: s.metrics["accuracy"].value for s in log.results.scores
          if "accuracy" in s.metrics}
print(log.eval.task, log.eval.task_args, scores)
bad = [f"{k}={scores.get(k)} < {v}" for k, v in (m.split("=") for m in args.min)
       if scores.get(k, 0) < float(v)]
if bad:
    sys.exit("GATE FAILED: " + ", ".join(bad))
