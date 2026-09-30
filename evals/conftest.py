import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def pytest_configure(config):
    config.addinivalue_line("markers", "llm_judge: uses an LLM grader (slower, costs tokens)")
    config.addinivalue_line("markers", "gate: aggregate threshold gate over the whole suite")
