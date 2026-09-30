# Usage: make all            (fake backend, free). promptfoo exits 100 because v2 fails by design;
#        the leading "-" keeps "make all" going.
#        make all LLM_BACKEND=anthropic ANTHROPIC_API_KEY=...
-include .env
export
export PYTHONPATH := $(CURDIR)
export PROMPTFOO_PYTHON := $(CURDIR)/.venv/bin/python
export PROMPTFOO_DISABLE_TELEMETRY := 1
export DEEPEVAL_TELEMETRY_OPT_OUT := YES
export INSPECT_LOG_DIR := $(CURDIR)/reports/inspect-logs
PY := .venv/bin/python
BIN := .venv/bin

setup:
	python3 -m venv .venv && $(BIN)/pip install -r requirements.txt && npm install

pytest:        ; $(BIN)/pytest evals/test_plain_pytest.py
pytest-fast:   ; $(BIN)/pytest evals/test_plain_pytest.py -m "not llm_judge and not gate"
pytest-trials: ; TRIALS=5 $(BIN)/pytest evals/test_plain_pytest.py -s
promptfoo:     ; -npx promptfoo eval -c evals/promptfoo/promptfooconfig.yaml -o reports/promptfoo_results.json
promptfoo-view:; npx promptfoo view
deepeval:      ; $(BIN)/deepeval test run evals/deepeval/test_deepeval.py
inspect:       ; $(BIN)/inspect eval evals/inspect/triage_task.py --model mockllm/model && $(PY) scripts/inspect_gate.py
inspect-view:  ; $(BIN)/inspect view
calibrate:     ; -$(PY) scripts/calibrate_judge.py
online:        ; TRACE_FILE=reports/traces.jsonl $(PY) scripts/online_eval.py
regression:    ; APP_VERSION=v2 $(BIN)/pytest evals/test_plain_pytest.py || true

all: pytest promptfoo deepeval inspect calibrate online
.PHONY: setup pytest pytest-fast pytest-trials promptfoo promptfoo-view deepeval inspect inspect-view calibrate online regression all

# --- platform & vendor evals (optional; see README) ---
phoenix-exp:   ; $(PY) evals/platforms/phoenix_experiment.py
langfuse-exp:  ; $(PY) evals/platforms/langfuse_experiment.py
mlflow-exp:    ; $(PY) evals/platforms/mlflow_eval.py
mlflow-ui:     ; $(BIN)/mlflow ui --backend-store-uri sqlite:///mlruns.db
braintrust:    ; $(BIN)/braintrust eval evals/platforms/eval_braintrust.py --no-send-logs
langsmith:     ; LANGSMITH_TEST_TRACKING=false $(BIN)/pytest evals/platforms/langsmith_suite.py
openai-evals:  ; $(PY) evals/platforms/openai_evals_api.py --mode byo
plugin-eval:   ; cd skill-evals/triage-plugin && claude plugin eval . --trust-plugin --runs 2 --model haiku --judge-model haiku --max-cost-usd 2 --no-publish

# --- self-hosted platforms in Docker (infra/compose.yaml; works from the dev container) ---
COMPOSE := docker compose -f infra/compose.yaml
OLLAMA_MODEL ?= qwen2.5:7b
up-phoenix:    ; $(COMPOSE) --profile phoenix up -d
up-langfuse:   ; $(COMPOSE) --profile langfuse up -d --wait
up-ollama:     ; $(COMPOSE) --profile ollama up -d && $(COMPOSE) exec ollama ollama pull $(OLLAMA_MODEL)
platforms-ps:  ; $(COMPOSE) --profile '*' ps
platforms-down:; $(COMPOSE) --profile '*' rm --stop --force
.PHONY: up-phoenix up-langfuse up-ollama platforms-ps platforms-down
