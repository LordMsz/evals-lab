"""Graders shared by every harness (pytest, promptfoo, DeepEval, Inspect).

Architectural point: keep *what good means* in your own code + dataset.
The harness (runner, reporting, UI) is swappable; your graders and golden data are the asset.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.triage import CATEGORIES, PRIORITIES

ROOT = Path(__file__).resolve().parents[1]


def load_cases(path: str | Path = ROOT / "data" / "cases.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def deterministic_checks(case: dict, result: dict) -> list[tuple[str, bool, str]]:
    """Cheap, fast, reproducible code-based graders. Returns (name, passed, detail)."""
    exp, out, trace = case["expect"], result["output"], result["trace"]
    answer = (out.get("answer") or "").lower()
    r: list[tuple[str, bool, str]] = []

    r.append(("schema", trace["parse_error"] is None and out.get("category") in CATEGORIES
              and out.get("priority") in PRIORITIES and isinstance(out.get("answer"), str),
              trace["parse_error"] or f"keys={sorted(out)}"))
    r.append(("category", out.get("category") in exp["category"],
              f"got {out.get('category')!r}, want one of {exp['category']}"))
    if "priority" in exp:
        r.append(("priority", out.get("priority") in exp["priority"],
                  f"got {out.get('priority')!r}, want one of {exp['priority']}"))
    if "tool" in exp:
        called = [t["name"] for t in trace["tool_calls"]]
        r.append(("tool_call", exp["tool"] in called, f"called={called}"))
    if "cite_any" in exp:
        cites = out.get("citations") or []
        r.append(("citation", any(c in cites for c in exp["cite_any"]),
                  f"citations={cites}, want any of {exp['cite_any']}"))
    if exp.get("no_citations"):
        r.append(("no_citation", not out.get("citations"), f"citations={out.get('citations')}"))
    for pat in exp.get("must_include", []):
        r.append((f"includes:{pat}", re.search(pat, answer, re.I) is not None, answer[:120]))
    for pat in exp.get("must_not", []):
        r.append((f"excludes:{pat}", re.search(re.escape(pat), answer, re.I) is None, answer[:120]))
    if len(answer.split(". ")) > 4:
        r.append(("length", False, "more than ~3 sentences"))
    return r


def retrieval_recall(case: dict, result: dict) -> float | None:
    """Classic IR metric - did retrieval surface a doc the answer should rely on?"""
    want = case["expect"].get("cite_any")
    if not want:
        return None
    got = [c["id"] for c in result["trace"]["retrieved"]]
    return 1.0 if any(w in got for w in want) else 0.0
