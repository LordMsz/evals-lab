"""Approach D - Inspect AI (UK AISI): Task = Dataset + Solver + Scorer, rich logs & viewer.

    inspect eval evals/inspect/triage_task.py --model mockllm/model          # app uses LLM_BACKEND
    inspect eval evals/inspect/triage_task.py --model mockllm/model -T version=v2
    inspect eval evals/inspect/triage_task.py --model mockllm/model --epochs 5  # repeated trials
    inspect view                                                              # log viewer

Inspect is model-centric (benchmarks, agents in sandboxes). To eval *your app* you write a
custom solver that calls it - shown here. --model is required but unused by our solver.
"""
from __future__ import annotations

import json

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (CORRECT, INCORRECT, Score, Target, accuracy, scorer,
                               stderr)
from inspect_ai.solver import Generate, TaskState, solver

from app.triage import triage
from evals.checks import deterministic_checks, load_cases
from evals.judge import judge_grounded


def dataset() -> list[Sample]:
    return [Sample(id=c["id"], input=c["ticket"], target=c["expect"]["category"],
                   metadata={"case": c}) for c in load_cases()]


@solver
def triage_app(version: str = "v1"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = triage(state.input_text, version=version)
        state.metadata["result"] = result
        state.output.completion = json.dumps(result["output"])
        return state
    return solve


@scorer(metrics=[accuracy(), stderr()])
def deterministic():
    async def score(state: TaskState, target: Target) -> Score:
        checks = deterministic_checks(state.metadata["case"], state.metadata["result"])
        failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
        return Score(value=INCORRECT if failed else CORRECT,
                     answer=state.output.completion,
                     explanation="; ".join(failed) or "all checks passed")
    return score


@scorer(metrics=[accuracy(), stderr()])
def grounded():
    async def score(state: TaskState, target: Target) -> Score:
        case = state.metadata["case"]
        v = judge_grounded(case["ticket"], state.metadata["result"], case.get("rubric"))
        return Score(value=CORRECT if v["pass"] else INCORRECT, explanation=v["reason"])
    return score


@task
def support_triage(version: str = "v1", trials: int = 1):
    return Task(
        dataset=dataset(),
        solver=triage_app(version),
        scorer=[deterministic(), grounded()],
        # pass_at_{k}: any trial passed; "mean": average. Consistency = look at per-epoch scores.
        epochs=Epochs(trials, ["mean", f"pass_at_{trials}"] if trials > 1 else ["mean"]),
        metadata={"version": version},
    )
