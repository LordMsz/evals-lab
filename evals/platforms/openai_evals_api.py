"""DEPRECATED: OpenAI shuts down the Evals dashboard + API on 2026-11-30 (read-only from 2026-10-31);
OpenAI recommends promptfoo instead. Kept as a reference for the concepts.

OpenAI Evals API: evals are server-side objects (eval = schema + graders, run = data + model).

Two ways to use it:
  --mode hosted : OpenAI generates the samples itself from a prompt template + model
                  (tests a PROMPT/MODEL, not your app: no retrieval, no tools, no code)
  --mode byo    : bring your own outputs - run the app locally, upload item+sample JSONL,
                  let OpenAI's graders score them (tests your APP, grading hosted)

    OPENAI_API_KEY=... python evals/platforms/openai_evals_api.py --mode byo
    python evals/platforms/openai_evals_api.py --mode hosted --dry-run     # print payloads only
Results: dashboard (report_url) or client.evals.runs.output_items.list(...).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.triage import ANSWER_PROMPTS, triage  # noqa: E402
from evals.checks import load_cases  # noqa: E402

JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5")
APP_MODEL = os.getenv("OPENAI_APP_MODEL", "gpt-5-mini")

DATA_SOURCE_CONFIG = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "ticket": {"type": "string"},
            "context": {"type": "string"},
            "expected_categories": {"type": "array", "items": {"type": "string"}},
            "output": {"type": "string"},        # BYO mode: the app's answer lives in the item
        },
        "required": ["ticket", "context", "expected_categories"],
    },
    "include_sample_schema": True,           # hosted mode: graders read {{sample.output_text}}
}


def config(mode):
    return {**DATA_SOURCE_CONFIG, "include_sample_schema": mode == "hosted"}


def criteria(mode):
    """BYO grades {{item.output}} (produced by our app); hosted grades the model's sample."""
    ref = "{{item.output}}" if mode == "byo" else "{{sample.output_text}}"
    return json.loads(json.dumps(TESTING_CRITERIA).replace("{{sample.output_text}}", ref))

TESTING_CRITERIA = [
    {   # python grader: deterministic, runs in OpenAI's sandbox
        "type": "python", "name": "category_correct", "pass_threshold": 1.0,
        "source": (
            "import json, re\n"
            "def grade(sample, item) -> float:\n"
            "    text = item.get('output') or (sample or {}).get('output_text', '')\n"
            "    m = re.search(r'\\{.*\\}', text, re.S)\n"
            "    try:\n"
            "        return 1.0 if json.loads(m.group(0))['category'] in item['expected_categories'] else 0.0\n"
            "    except Exception:\n"
            "        return 0.0\n"),
    },
    {   # score_model grader: LLM judge returning a number
        "type": "score_model", "name": "grounded", "model": JUDGE_MODEL,
        "range": [0, 1], "pass_threshold": 1.0,
        "input": [
            {"role": "system", "content":
                "Score 1 if every factual claim in the answer (policies, numbers, dates, promises) is "
                "supported by the context, else 0. Escalating to a human is always acceptable."},
            {"role": "user", "content":
                "Context:\n{{item.context}}\n\nAssistant output:\n{{sample.output_text}}"},
        ],
    },
    {   # label_model grader: LLM judge choosing a label
        "type": "label_model", "name": "no_false_promise", "model": JUDGE_MODEL,
        "labels": ["safe", "false_promise"], "passing_labels": ["safe"],
        "input": [
            {"role": "developer", "content":
                "Label 'false_promise' if the reply promises a refund, credit or delivery date "
                "that the context does not state; otherwise 'safe'."},
            {"role": "user", "content": "Context: {{item.context}}\nReply: {{sample.output_text}}"},
        ],
    },
]


def items():
    out = []
    for c in load_cases():
        r = triage(c["ticket"])          # context = what our retriever found (for BYO + hosted)
        ctx = "\n".join(f'{d["id"]}: {d["text"]}' for d in r["trace"]["retrieved"]) or "(none)"
        out.append((c, r, {"ticket": c["ticket"], "context": ctx,
                           "expected_categories": c["expect"]["category"]}))
    return out


def run_payload(mode):
    rows = items()
    if mode == "byo":
        return {"type": "jsonl", "source": {"type": "file_content", "content": [
            {"item": {**item, "output": json.dumps(r["output"])}}
            for _, r, item in rows]}}
    return {
        "type": "responses", "model": APP_MODEL,
        "input_messages": {"type": "template", "template": [
            {"role": "developer", "content": ANSWER_PROMPTS[os.getenv("APP_VERSION", "v1")]},
            {"role": "user", "content": "<ticket>{{item.ticket}}</ticket>\n"
                                        "<context>{{item.context}}</context>\n"
                                        "<tool_result>none</tool_result>"}]},
        "source": {"type": "file_content", "content": [{"item": item} for _, _, item in rows]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["byo", "hosted"], default="byo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data_source = run_payload(a.mode)
    if a.dry_run or not os.getenv("OPENAI_API_KEY"):
        print(json.dumps({"eval": {"data_source_config": config(a.mode),
                                   "testing_criteria": criteria(a.mode)},
                          "run": {"data_source": data_source}}, indent=2)[:3000])
        print(f"\n[dry run] {len(data_source['source']['content'])} items; set OPENAI_API_KEY to run.")
        return
    from openai import OpenAI
    client = OpenAI()
    ev = client.evals.create(name=f"support-triage-{a.mode}", data_source_config=config(a.mode),
                             testing_criteria=criteria(a.mode))
    run = client.evals.runs.create(
        ev.id, name=f"triage-{os.getenv('APP_VERSION', 'v1')}-{a.mode}", data_source=data_source)
    print("report:", run.report_url)
    while run.status in ("queued", "in_progress"):
        time.sleep(5)
        run = client.evals.runs.retrieve(run.id, eval_id=ev.id)
    print(run.status, run.result_counts, run.per_testing_criteria_results)


if __name__ == "__main__":
    main()
