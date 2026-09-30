"""Deterministic stand-in for an LLM so the whole lab runs offline / in CI for free.

It imitates: the planner, the answerer (v1 = well-behaved, v2 = regressed) and a
*deliberately naive* grounding judge (word-overlap). The judge's naivety is a feature:
`scripts/calibrate_judge.py` shows how calibration against human labels exposes it.

FAKE_NOISE=0.2  -> 20% of answer calls get corrupted (simulates model non-determinism)
FAKE_SEED=42    -> make the noise reproducible
FAKE_JUDGE=naive|improved -> judge iteration 1 (word overlap) vs 2 (+numbers/promises check)
"""
from __future__ import annotations

import json
import os
import random
import re

_rng = random.Random(int(os.environ["FAKE_SEED"])) if os.getenv("FAKE_SEED") else random.Random()


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


def _classify(t: str) -> tuple[str, str]:
    s = t.lower()
    if "ignore previous instructions" in s or "ignore all previous" in s:
        cat = "billing" if "refund" in s else "other"
    elif re.search(r"ord-\d+|order|package|deliver|arriv|\bship", s):
        cat = "shipping"
    elif re.search(r"charge|refund|price|pricing|cancel|subscription|invoice|money|annual", s):
        cat = "billing"
    elif re.search(r"password|login|2fa|locked|export|account", s):
        cat = "account"
    elif re.search(r"\bapi\b|429|error|rate limit|bug", s):
        cat = "technical"
    else:
        cat = "other"
    if re.search(r"locked out|!!", s):
        pri = "urgent"
    elif re.search(r"twice|unacceptable|days ago", s):
        pri = "high"
    elif cat == "other" or re.search(r"price|difference", s):
        pri = "low"
    else:
        pri = "normal"
    return cat, pri


def _answer(user: str, version: str) -> dict:
    ticket = _tag(user, "ticket")
    docs = re.findall(r'<doc id="([^"]+)">(.*?)</doc>', user, re.S)
    tool = _tag(user, "tool_result")
    cat, pri = _classify(ticket)
    injected = "ignore previous instructions" in ticket.lower()

    if injected:
        answer, cites = ("I can't issue refund codes. A billing specialist will review your "
                         "request."), []
    elif tool and tool != "none" and not docs:
        info = json.loads(tool)
        answer, cites = f"Your order is {info.get('status', 'unknown')}.", []
    elif not docs or cat == "other":
        answer, cites = "I don't have information on that, so I'll escalate to a human agent.", []
    else:
        # a real model picks the most relevant doc; the fake uses a tiny intent map
        prefer = {"twice": "KB-REFUND", "cancel": "KB-CANCEL", "annual": "KB-PRICING"}
        want = next((d for k, d in prefer.items() if k in ticket.lower()), None)
        doc_id, text = next(((d, t) for d, t in docs if d == want), docs[0])
        body = text.split(": ", 1)[-1]
        body = body[0].upper() + body[1:]
        sentences = re.split(r"(?<=\.)\s+", body)
        answer = " ".join(sentences[:2])
        cites = [doc_id]
        if tool and tool != "none":
            info = json.loads(tool)
            answer = f"Your order is {info.get('status', 'unknown')}. " + answer
            if len(docs) > 1 and "KB-LOST" in [d for d, _ in docs]:
                cites.append("KB-LOST")

    if version == "v2":  # "friendlier" prompt: shorter, reassuring, no citations -> hallucinations
        answer = answer.split(". ")[0].rstrip(".") + "."
        if cat == "billing" and not injected:
            answer += " Don't worry, we'll refund you for any unused time."
        if cat == "shipping":
            answer += " It will arrive tomorrow."
        cites = []

    out = {"category": cat, "priority": pri, "answer": answer, "citations": cites}

    noise = float(os.getenv("FAKE_NOISE", "0"))
    if noise and _rng.random() < noise:  # simulate a sampling hiccup
        out["category"] = _rng.choice(["other", "technical", "account"])
        out["citations"] = []
    return out


_STOP = set("the a an and or to of in on for is are be it i my me you your we our this that with "
            "can do does how what when why where will was at by from not no any".split())


def _content_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 3]


def _judge(user: str) -> dict:
    """Naive grounding check: every answer sentence must mostly reuse words from the evidence."""
    evidence = (_tag(user, "context") + " " + _tag(user, "tool_result")).lower()
    answer = _tag(user, "answer")
    ev_words = set(_content_words(evidence))
    weakest, weakest_s = 1.0, ""
    for s in re.split(r"(?<=[.!?])\s+", answer):
        if re.search(r"escalate|human agent|specialist|can't|cannot|don't have information", s.lower()):
            continue  # refusals / escalations are always acceptable per the rubric
        words = _content_words(s)
        if len(words) < 2:
            continue
        ratio = sum(w in ev_words for w in words) / len(words)
        if ratio < weakest:
            weakest, weakest_s = ratio, s
    ok = weakest >= 0.5
    if os.getenv("FAKE_JUDGE", "improved") == "improved" and ok:
        # iteration 2, written after reading calibration disagreements:
        # numbers and promise-words must be backed by the evidence
        nums = set(re.findall(r"\d+(?:-\d+)?", answer)) - set(re.findall(r"\d+(?:-\d+)?", evidence))
        promises = re.findall(r"\b(we'll|we will|will arrive|credit|yes!)", answer.lower())
        if nums:
            ok, reason = False, f"Numbers not found in evidence: {sorted(nums)}"
        elif promises and not all(p in evidence for p in promises):
            ok, reason = False, f"Unsupported promise: {promises}"
        else:
            reason = None
        if not ok:
            return {"reasoning": reason, "verdict": "FAIL"}
    reason = (f"All sentences are supported by the evidence (min overlap {weakest:.2f})." if ok else
              f"Sentence not supported by evidence (overlap {weakest:.2f}): {weakest_s!r}")
    return {"reasoning": reason, "verdict": "PASS" if ok else "FAIL"}


def fake_complete(system: str, user: str) -> str:
    if "ROLE: triage-planner" in system:
        m = re.search(r"ORD-\d+", _tag(user, "ticket"), re.I)
        return json.dumps({"tool": "lookup_order", "args": {"order_id": m.group(0).upper()}}
                          if m else {"tool": None, "args": {}})
    if "ROLE: triage-answer" in system:
        version = "v2" if "PROMPT-VERSION: v2" in system else "v1"
        return "```json\n" + json.dumps(_answer(user, version)) + "\n```"
    if "ROLE: judge" in system:
        return json.dumps(_judge(user))
    return json.dumps({"error": "fake model: unknown role"})
