# eval-lab

One small LLM app, one golden dataset, evaluated by several tools. The graders
(`evals/checks.py`, `evals/judge.py`) and the data (`data/*.jsonl`) are shared. Each tool is a thin adapter around them.

Requirements: Linux/macOS/WSL, Python 3.11+, Node 20+ (for promptfoo only). No Docker needed.

---

## 1. Setup (once)

```bash
make setup                        # .venv + requirements.txt + npm install (promptfoo)
cp .env.example .env              # then put your key in .env (never commit it)
```

`.env` is loaded automatically by `make`. Without a key, everything runs on the offline **fake** backend (free, deterministic, used for testing the harness only).

| Variable | Values | Default |
|---|---|---|
| `LLM_BACKEND` | `fake` · `openai` · `anthropic` · `ollama` | `fake` |
| `APP_MODEL` / `JUDGE_MODEL` | any model id of that backend | see `app/llm.py` |
| `JUDGE_BACKEND` | judge on a different provider than the app | = `LLM_BACKEND` |
| `APP_VERSION` | `v1` (good prompt) · `v2` (regressed prompt) | `v1` |
| `TRIALS` | repeat each case N times | `1` |

## 2. Core harnesses (the main line)

| Command | What runs | Fails build on regression? |
|---|---|---|
| `make pytest` | plain pytest: per-case asserts + suite gates | yes (exit 1) |
| `make pytest-fast` | only deterministic checks, no LLM judge | yes |
| `make regression` | same suite against the broken `v2` prompt | shows failures |
| `make promptfoo` / `make promptfoo-view` | v1 vs v2 side by side; web UI on :15500 | yes (exit 100) |
| `make deepeval` | DeepEval metrics via pytest | yes (exit 1) |
| `make inspect` / `make inspect-view` | Inspect AI task + gate script; log viewer on :7575 | via `scripts/inspect_gate.py` |
| `make calibrate` | judge vs 24 human labels → TPR/TNR/kappa | exit 1 if judge untrustworthy |
| `make online` | simulated prod traffic, sampled judging, OTel traces | alert line only |
| `make all` | all of the above | |

Real model example:
```bash
make calibrate LLM_BACKEND=openai JUDGE_MODEL=gpt-6-luna JUDGE_REASONING_EFFORT=medium   # 1st: can the judge be trusted?
make pytest    LLM_BACKEND=openai TRIALS=3
```

## 3. Platforms, self-hosted locally (optional)

Use a separate venv per platform. Their dependency trees are large and can clash.

**Phoenix** (Arize, ELv2): single process, SQLite by default (Postgres for prod), no Docker.
```bash
python3 -m venv .venv-phoenix && .venv-phoenix/bin/pip install arize-phoenix opentelemetry-exporter-otlp-proto-http
.venv-phoenix/bin/phoenix serve &                                        # UI http://localhost:6006
PYTHONPATH=. .venv-phoenix/bin/python evals/platforms/phoenix_experiment.py   # dataset + v1/v2 experiments
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006 make online            # traces + online evals
```

**MLflow** (Apache-2.0): SQLite file, no server needed to run evals.
```bash
python3 -m venv .venv-mlflow && .venv-mlflow/bin/pip install "mlflow>=3.16"
PYTHONPATH=. .venv-mlflow/bin/python evals/platforms/mlflow_eval.py
.venv-mlflow/bin/mlflow ui --backend-store-uri sqlite:///mlruns.db       # UI http://localhost:5000
```

**Langfuse** (MIT core): needs Postgres, ClickHouse, Redis and S3/MinIO, so run it with Docker Compose locally or Helm/Terraform in AWS/Azure.
```bash
git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up   # UI http://localhost:3000
# create project + API keys in the UI, then:
LANGFUSE_HOST=http://localhost:3000 LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
  PYTHONPATH=. python evals/platforms/langfuse_experiment.py
```

SaaS-only or vendor tools (reference only, not needed for the main line):

| Script | Tool | Runs without account? |
|---|---|---|
| `evals/platforms/eval_braintrust.py` | Braintrust `Eval()` | yes, `--no-send-logs` |
| `evals/platforms/langsmith_suite.py` | LangSmith pytest plugin | yes, `LANGSMITH_TEST_TRACKING=false` |
| `evals/platforms/openai_evals_api.py` | OpenAI Evals API (**shuts down 30 Nov 2026**, use promptfoo) | `--dry-run` prints payloads; live needs `OPENAI_API_KEY` |
| `skill-evals/triage-plugin/` | Claude Code `claude plugin eval` (side note: agent-extension evals) | no, uses your Claude login |

## 4. Layout

```
app/            system under test: triage.py (RAG + tool + 2 LLM calls), llm.py (provider switch),
                fake_model.py (offline stand-in), telemetry.py (OTel GenAI spans)
data/           cases.jsonl (golden set), judge_labels.jsonl (human labels for calibration)
evals/          checks.py + judge.py (the graders), one adapter per harness, platforms/
scripts/        calibrate_judge.py, online_eval.py, inspect_gate.py
.github/        tiered CI example (PR smoke → golden set → nightly)
```

## 5. Troubleshooting
- `ModuleNotFoundError: app`: run from the repo root with `PYTHONPATH=.` (the Makefile sets it).
- promptfoo can't find Python: `export PROMPTFOO_PYTHON=$PWD/.venv/bin/python`.
- Inspect viewer crashes on locale `en-US@posix`: `export LANG=en_US.UTF-8`.
