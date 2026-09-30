"""LLM-as-judge, built the way current practice recommends:

* binary PASS/FAIL, one criterion per judge (not a 1-10 "quality" score)
* the judge sees the evidence (context + tool results), not just the answer
* reasoning before verdict, strict JSON output, temperature 0
* a different (ideally stronger) model than the app  -> JUDGE_MODEL
* validated against human labels before you trust it -> scripts/calibrate_judge.py
"""
from __future__ import annotations

import json

from app import llm

GROUNDEDNESS_SYSTEM = """ROLE: judge
You are a strict QA reviewer for a customer-support assistant.
Criterion - GROUNDED: every factual claim in <answer> (policies, numbers, dates, promises,
order status) must be supported by <context> or <tool_result>. Saying "I don't know" or
escalating / handing off to a human (including "right away") is always acceptable.
Friendly filler without facts is acceptable. Faithful paraphrases and roundings of stated
facts are acceptable (e.g. "30 minutes" -> "half an hour", "5-7 business days" -> "about a week").
Check arithmetic derived from the evidence.
{extra}
FAIL if the answer invents or alters any fact, promises something not in the evidence,
or violates the extra criterion.

Examples:
- context "refunds within 30 days", answer "You can get a refund within 60 days" -> FAIL
- context "Basic 9 EUR/month", answer "Basic is 9 EUR per month" -> PASS
- answer "It will arrive tomorrow" with no delivery date in evidence -> FAIL

Think step by step, then reply with JSON only:
{{"reasoning": "<one or two sentences>", "verdict": "PASS" | "FAIL"}}"""


def judge_grounded(ticket: str, result: dict, extra_rubric: str | None = None) -> dict:
    trace = result["trace"]
    ctx = "\n".join(f'<doc id="{c["id"]}">{c["text"]}</doc>' for c in trace["retrieved"])
    tool = json.dumps(trace["tool_calls"][0]["result"]) if trace["tool_calls"] else "none"
    user = (f"<ticket>{ticket}</ticket>\n<context>\n{ctx}\n</context>\n"
            f"<tool_result>{tool}</tool_result>\n"
            f"<answer>{result['output'].get('answer', '')}</answer>")
    system = GROUNDEDNESS_SYSTEM.format(
        extra=f"Extra criterion: {extra_rubric}" if extra_rubric else "")
    raw = llm.complete(system, user, role="judge")
    try:
        v = llm.parse_json(raw)
        passed = str(v.get("verdict", "")).upper() == "PASS"
        return {"pass": passed, "reason": v.get("reasoning", ""), "raw": raw}
    except Exception as e:
        return {"pass": False, "reason": f"judge output unparsable: {e}", "raw": raw}


def judge_raw(ticket: str, context: str, tool_result: str, answer: str,
              extra_rubric: str | None = None) -> dict:
    """Same judge, fed directly with strings (used for calibration)."""
    fake_result = {"trace": {"retrieved": [{"id": "CTX", "text": context}],
                             "tool_calls": [{"result": tool_result}] if tool_result else []},
                   "output": {"answer": answer}}
    return judge_grounded(ticket, fake_result, extra_rubric)
