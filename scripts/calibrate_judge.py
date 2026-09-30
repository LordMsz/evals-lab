"""Validate the LLM judge against human labels before trusting it in CI.

    python scripts/calibrate_judge.py                  # judge = JUDGE_BACKEND/JUDGE_MODEL
    python scripts/calibrate_judge.py --split test     # final check on held-out labels only

Reports:
  * confusion matrix, TPR/TNR (positive class = human PASS), Cohen's kappa
  * disagreements (read these! they tell you how to fix the judge prompt)
  * bias-corrected pass-rate estimate: if the judge says 80% pass on production traffic,
    what is the true pass rate given its known error rates? (Rogan-Gladen correction)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import llm  # noqa: E402
from evals.judge import judge_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def kappa(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / n**2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def corrected_pass_rate(observed, tpr, tnr):
    denom = tpr + tnr - 1
    return None if denom <= 0 else min(1.0, max(0.0, (observed + tnr - 1) / denom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "reports" / "judge_calibration.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "data" / "judge_labels.jsonl").read_text().splitlines()]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    tp = fn = fp = tn = 0
    disagreements = []
    for r in rows:
        v = judge_raw(r["ticket"], r["context"], r["tool_result"], r["answer"])
        judge_pass, human_pass = v["pass"], r["human"] == "PASS"
        tp += judge_pass and human_pass
        fn += (not judge_pass) and human_pass
        fp += judge_pass and not human_pass
        tn += (not judge_pass) and not human_pass
        if judge_pass != human_pass:
            disagreements.append({"id": r["id"], "human": r["human"],
                                  "judge": "PASS" if judge_pass else "FAIL",
                                  "note": r["note"], "judge_reason": v["reason"]})
    tpr = tp / (tp + fn) if tp + fn else 0
    tnr = tn / (tn + fp) if tn + fp else 0
    report = {
        "judge_model": llm.model_id("judge"), "split": args.split, "n": len(rows),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "accuracy": (tp + tn) / len(rows), "TPR": tpr, "TNR": tnr,
        "cohens_kappa": kappa(tp, fn, fp, tn),
        "example_correction": {
            "judge_says_pass_rate": 0.80,
            "corrected_true_pass_rate": corrected_pass_rate(0.80, tpr, tnr)},
        "disagreements": disagreements,
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"judge={report['judge_model']}  n={len(rows)}  acc={report['accuracy']:.2f}  "
          f"TPR={tpr:.2f}  TNR={tnr:.2f}  kappa={report['cohens_kappa']:.2f}")
    print(f"judge says 80% pass -> corrected estimate: "
          f"{report['example_correction']['corrected_true_pass_rate']}")
    for d in disagreements:
        print(f"  {d['id']} human={d['human']} judge={d['judge']}  ({d['note']})")
    # a judge that misses most failures must not gate releases
    sys.exit(0 if tnr >= 0.8 and tpr >= 0.8 else 1)


if __name__ == "__main__":
    main()
