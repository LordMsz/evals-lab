"""Generate promptfoo test cases from the shared golden dataset (single source of truth)."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generate_tests(config=None):
    tests = []
    with open(os.path.join(ROOT, "data", "cases.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            tests.append({
                "description": case["id"],
                "vars": {"ticket": case["ticket"], "case": json.dumps(case)},
                "metadata": {"tags": ",".join(case.get("tags", []))},
            })
    return tests
