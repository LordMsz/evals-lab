"""promptfoo custom Python provider: wraps the *whole app* (retrieval + tool + 2 LLM calls),
not just a prompt. This is how you eval a workflow rather than a model."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.triage import triage  # noqa: E402


def call_api(prompt, options, context):
    version = (options.get("config") or {}).get("version", "v1")
    result = triage(prompt, version=version)
    return {"output": json.dumps(result), "metadata": {"version": version}}
