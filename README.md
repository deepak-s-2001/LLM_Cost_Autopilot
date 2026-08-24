# LLM Cost Autopilot

A routing layer that sits in front of OpenAI, Anthropic, and a local Ollama model. It classifies each incoming request's complexity, routes it to the cheapest model that can actually handle it, and asynchronously verifies the response — escalating to a stronger model (and feeding the failure back into the classifier's training data) when the cheap model wasn't good enough. Time-sensitive prompts bypass tier routing entirely and go straight to a flagship model with live web search, since a classifier has no signal for "this needs current information."

Repo: https://github.com/deepak-s-2001/LLM_Cost_Autopilot

## What's not in this repo

Same spirit as any local research/dev app — the public repo ships code and the trained classifier artifact only, not runtime data. Not included:

- `.env` — your own API keys, never committed (`.env.example` is the template)
- `data/autopilot.db` — the local SQLite database (regenerates empty on first run)
- `data/logs/` — accumulated request logs and load-test reports
- `PROJECT.md`, `CASE_STUDY.md` — private working docs, gitignored intentionally

Everything needed to run the system from scratch — schema, migrations, the trained classifier, config — is in the repo.

## Architecture

```
                         ┌─────────────────┐
      client ──POST──▶   │   FastAPI (api)  │◀── also serves frontend/ (static chat UI) at "/"
                         │  /v1/completions │
                         └───────┬──────────┘
                                 │ 1. classify prompt → tier (or detect "needs current info" → web search)
                                 │ 2. resolve tier → model (config/routing.yaml)
                                 │ 3. send_request() → provider (OpenAI / Anthropic / Ollama)
                                 │ 4. log to SQLite, return response
                                 │ 5. enqueue verification job (BackgroundTasks)
                                 ▼
                         ┌──────────────────┐        ┌─────────────────────┐
                         │  data/autopilot  │◀──poll──│ verifier_worker.py  │
                         │      .db         │         │  (separate process) │
                         └──────────────────┘         └──────────┬──────────┘
                                 ▲                                │ on fail:
                                 │                                │ - escalate to
                       ┌─────────┴─────────┐                      │   stronger model
                       │  dashboard/app.py │                      │ - log training
                       │ (Streamlit, stats  │                      │   example
                       │  only — chat lives │                      ▼
                       │  in frontend/)     │         scripts/retrain.py
                       └───────────────────┘          (feedback loop)
```

Providers (`app/providers/`) are unified behind a single `send_request(prompt, model_config)` interface returning a standardized `Response` (text, tokens, cost, latency). The classifier (`app/classifier/`) is a logistic regression model over ten hand-engineered text features, trained on a 226-row labeled dataset plus any accumulated routing-failure examples. The chat frontend (`frontend/`) is plain HTML/CSS/JS with no build step, served directly by the FastAPI app.

## Data organization

```
app/classifier/artifacts/   trained classifier.joblib + feature_columns.json (committed — needed to run)
app/classifier/data/        labeled_prompts.csv (226-row training set, committed)
config/routing.yaml         tier → model map, hot-reloadable via PUT /v1/routing-config
data/
  autopilot.db               SQLite DB — requests, verification_jobs, escalations, training_examples (gitignored, regenerates empty)
  logs/                       accumulated request/load-test logs (gitignored)
  prompts/                    load-test prompt datasets (committed — no real data, synthetic prompts only)
frontend/                    static chat UI (index.html, app.js, style.css)
dashboard/                   Streamlit analytics dashboard
worker/                      async verification worker (separate process)
```

## Prerequisites

- Python 3.12
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (optional — only needed for the containerized path)
- [Ollama](https://ollama.com) — provides the free local tier
- An OpenAI API key and an Anthropic API key

## Setup

```bash
git clone https://github.com/deepak-s-2001/LLM_Cost_Autopilot.git
cd LLM_Cost_Autopilot

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # macOS/Linux

ollama pull llama3.1:8b

cp .env.example .env
# edit .env: fill in OPENAI_API_KEY and ANTHROPIC_API_KEY

python -m app.classifier.train      # trains and saves the classifier artifact (already committed, but re-run any time you add training data)
```

## Run locally

**Terminal 1 — API (also serves the chat UI):**

```bash
.venv\Scripts\uvicorn app.api.main:app --reload    # Windows
# .venv/bin/uvicorn app.api.main:app --reload      # macOS/Linux
```

Open http://localhost:8000 for the chat UI.

**Terminal 2 — verification worker (required for escalation/quality checks to run):**

```bash
.venv\Scripts\python worker\verifier_worker.py     # Windows
# .venv/bin/python worker/verifier_worker.py       # macOS/Linux
```

**Terminal 3 — analytics dashboard (optional):**

```bash
.venv\Scripts\streamlit run dashboard\app.py        # Windows
# .venv/bin/streamlit run dashboard/app.py          # macOS/Linux
```

Open http://localhost:8501 for cost/routing/quality charts.

**Alternative — Docker Compose (API + worker only):**

```bash
docker compose up --build
```

Ollama stays on the host; containers reach it via `host.docker.internal:11434` (already configured). Run the dashboard separately with the Terminal 3 command above — it isn't containerized.

## API

- `POST /v1/completions` — `{messages: [{role, content}], use_case?, max_tokens?}` (standard chat completion shape). The caller doesn't pick the model — the response's `routing` field reports which model was used, why, its cost, and the classifier's full tier-confidence breakdown.
- `GET /v1/completions/{id}` — poll for verification status (`pending` / `passed` / `escalated`); on `escalated`, includes the corrected answer.
- `GET /v1/models` — registry of available models and pricing.
- `GET /v1/stats` — cost savings summary, routing distribution, escalation rate over time.
- `PUT /v1/routing-config` — update the tier → model mapping without redeploying (validated against the model registry).

## Verification

```bash
pytest                                    # all provider/LLM calls mocked — zero API cost
python scripts/baseline_test.py --yes     # deliberate real-spend baseline (~$0.03)
python scripts/retrain.py                 # retrain classifier on accumulated failures
```

Smoke-test the running API:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Should return a tier-1 response routed to `llama3.1:8b` at $0 cost.

## Design notes

- **Async verification is genuinely async.** The worker polls `verification_jobs` and processes them after the original HTTP response has already returned — a completion response can't report a same-request escalation, so `escalated` is always `false` in the immediate response. Escalation results are delivered separately, via `GET /v1/completions/{id}` (the chat frontend polls this automatically and appends a corrected-answer card if one arrives).
- **Time-sensitive prompts bypass tier routing.** A cheap/local model's training cutoff can't be fixed by routing cleverness alone — prompts detected as needing current information (recency language, or a year at/after the cheapest model's approximate cutoff) go straight to the flagship model with live web search instead.
- **Classifier is intentionally simple (v1).** Logistic regression over engineered features, not a fine-tuned model — noticeably weaker on summarization-vs-reasoning boundary cases. The escalation feedback loop (`scripts/retrain.py`) is the mechanism designed to improve it over time, though retraining is currently a manual step, not an automated schedule.
