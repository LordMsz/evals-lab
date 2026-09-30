# eval-lab

One small LLM app, one golden dataset, evaluated by several tools. The graders
(`evals/checks.py`, `evals/judge.py`) and the data (`data/*.jsonl`) are shared. Each tool is a thin adapter around them.

Requirements, pick one:
- **Dev container (recommended):** Docker + VS Code with the Dev Containers extension. Nothing else on the host.
- **Local:** Linux/macOS/WSL, Python 3.11+, Node 20+ (for promptfoo only).

---

## 1. Setup (once)

**Dev container:** open the folder in VS Code → *Reopen in Container*. The first build creates
`.venv` and `node_modules` (both live in Docker volumes, not in your checkout) and installs
Python 3.12, Node 22, the Docker CLI and the `claude` CLI. Then:

```bash
cp .env.example .env              # optional: only for real models; put your key in .env
```

**Local:**
```bash
make setup                        # .venv + requirements.txt + npm install (promptfoo)
cp .env.example .env              # then put your key in .env (never commit it)
```

Windows tip: a checkout on a Windows drive is bind-mounted into the container, which is slow for
file-heavy work. For speed, use *Dev Containers: Clone Repository in Container Volume* or keep the
repo in the WSL filesystem.

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

### In Docker (dev container or any shell with Docker)

`infra/compose.yaml` holds each platform as a compose profile. The dev container talks to the
host's Docker engine, so these run as sibling containers on the same network. Nothing starts until
you ask for it.

| Command | Starts | UI (host browser) | From the dev container |
|---|---|---|---|
| `make up-phoenix` | Phoenix (SQLite) | http://localhost:6006 | `http://phoenix:6006` |
| `make up-langfuse` | Langfuse web + worker, Postgres, ClickHouse, Redis, MinIO | http://localhost:3000 (`admin@eval-lab.test` / `eval-lab-local`) | `http://langfuse-web:3000` |
| `make up-ollama` | Ollama + pulls `OLLAMA_MODEL` (default `qwen2.5:7b`) | – | `http://ollama:11434` |
| `make platforms-ps` / `make platforms-down` | status / stop and remove (data volumes are kept) | | |

The dev container already has `PHOENIX_URL`, `LANGFUSE_HOST`/keys and `OLLAMA_BASE_URL` pointing at
these services. The Langfuse project and its API keys are created on first start. Then:

```bash
make up-phoenix  && .venv/bin/pip install arize-phoenix-client && make phoenix-exp
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix:6006 make online                       # traces → Phoenix

make up-langfuse && .venv/bin/pip install "langfuse>=3" && make langfuse-exp
OTEL_EXPORTER_OTLP_ENDPOINT=http://langfuse-web:3000/api/public/otel \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20$(printf pk-lf-local-dev:sk-lf-local-dev | base64)" \
  make online                                                                      # traces → Langfuse

make up-ollama   && make pytest LLM_BACKEND=ollama

.venv/bin/pip install "mlflow>=3.16" && make mlflow-exp && make mlflow-ui           # UI on :5000, forwarded
```

If the platform SDKs clash with `requirements.txt`, use a separate venv as shown below.

### Without Docker

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
.devcontainer/  dev container (Python + Node + Docker CLI), see §1
infra/          compose profiles for self-hosted platforms (Phoenix, Langfuse, Ollama), see §3
.github/        tiered CI example (PR smoke → golden set → nightly)
```

## 5. Troubleshooting
- `ModuleNotFoundError: app`: run from the repo root with `PYTHONPATH=.` (the Makefile sets it).
- promptfoo can't find Python: `export PROMPTFOO_PYTHON=$PWD/.venv/bin/python`.
- Inspect viewer crashes on locale `en-US@posix`: `export LANG=en_US.UTF-8`.
