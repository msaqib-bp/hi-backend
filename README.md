# AI Smart Civic Services — Backend API

FastAPI service that turns an unstructured citizen complaint into structured, actionable
information: category, priority, responsible department, and a one-line dispatch summary
— then tracks it through resolution and reports the statistics.

Built for the OpenBook AI Hackathon. Frontend lives in a separate repository.

---

## Contents

- [The problem](#the-problem)
- [What the AI actually does](#what-the-ai-actually-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Statistics engine](#statistics-engine)
- [Testing](#testing)
- [Deployment](#deployment)
- [Limitations](#limitations)

---

## The problem

Citizens report broken streetlights, overflowing bins, potholes, water leaks and blocked
drains through fragmented channels. The result:

- **Citizens** get no acknowledgement and no way to check progress.
- **Service teams** receive an unsorted pile of free text — no way to tell what is urgent,
  which department owns it, or which five reports describe the same burst pipe.
- **Managers** have no data on what breaks most, where, or how slowly it gets fixed.

This service is the middle layer: it reads the complaint, structures it, routes it, and
measures the whole pipeline.

---

## What the AI actually does

### Input → processing → output

| | |
|---|---|
| **Input** | The citizen's free-text complaint (and optionally the location string) |
| **Processing** | Normalise → TF-IDF word (1–2gram) + character (3–5gram) vectors → two linear classifiers; the category score is then blended with a curated civic lexicon; the priority model also receives 9 engineered severity features |
| **Output** | `category` (7 classes) and `priority` (4 classes), each with a calibrated confidence and runner-up candidates; a one-line dispatch summary; the routed department; keywords; the engine and model version that produced it |

Every prediction is stored on the complaint as `ai_output`, so the UI can show *why* a
decision was made and an auditor can still read it after the model is retrained.

### The engines

| | `MLAnalyzer` (default) | LLM provider (optional) |
|---|---|---|
| Technology | scikit-learn, local | Claude **or** DeepSeek |
| Cost | **Free** | Pay per token |
| Needs a key | No | Yes — either key works |
| Handles | Category, priority, summary, duplicates | Better summaries, civic assistant |
| If it fails | Falls back to keyword rules | Falls back to the ML result |

**The ML engine is the default on purpose.** No LLM API has a meaningful free tier, so
making one mandatory would mean the demo dies the moment a key expires or a quota runs
out. The classifiers carry every mandatory capability on their own; the LLM is an
enhancement layer on the one field it is genuinely better at — writing the dispatch
instruction.

### Choosing an LLM provider

Two are supported, and the app works identically with either — or with neither.

```bash
DEEPSEEK_API_KEY=sk-...      # DeepSeek (OpenAI-compatible)
# or
ANTHROPIC_API_KEY=sk-ant-... # Claude
```

`LLM_PROVIDER` defaults to `auto`: it picks whichever key is present, preferring
Anthropic when both are, because its **strict tool schema** is validated server-side and
cannot return an invalid label. DeepSeek uses **JSON mode** with the schema and an example
in the prompt, and the response is validated on the way back — a hallucinated category
raises rather than being coerced, so the pipeline falls back to the ML result instead of
filing a complaint under an invented label.

Set `LLM_PROVIDER=deepseek` (or `anthropic`, or `none`) to force one. An explicit choice
whose key is missing reports "not configured" rather than silently running on the other
vendor's bill.

The `OpenAICompatibleAnalyzer` is not DeepSeek-specific — point
`OPENAI_COMPATIBLE_BASE_URL` / `_MODEL` / `_API_KEY` at Groq, Together, OpenRouter or a
local vLLM/Ollama server and it works unchanged.

**Verify whichever you set:**

```bash
python -m app.check_llm
```

It reports which provider was selected and why, then makes three real calls (triage,
summary, assistant) and prints the results — so a bad key or an exhausted balance surfaces
there rather than silently degrading every complaint to the extractive summary.

The pipeline degrades in three tiers and **never fails a submission**:

```
MLAnalyzer ──(optional summary upgrade)──▶ LLMAnalyzer
     │ unavailable/raises                       │ unavailable/raises
     ▼                                          ▼
RuleAnalyzer  ─────────────────────────  keep the ML result
```

### Accuracy — the honest numbers

Full report: [`reports/ai_evaluation.md`](reports/ai_evaluation.md) (regenerated on every
training run).

| Metric | Random split | **Unseen phrasings** |
|---|---|---|
| Category accuracy | 1.000 | **0.821** |
| Priority accuracy | 0.962 | **0.583** |
| Priority within one band | — | **~0.95** |

**Quote the right column.** The random split is inflated: all samples are generated from a
shared pool of ~78 issue phrasings, so the same phrasing appears in both halves and the
classifier only has to memorise it. The second column withholds entire phrasings and
impact clauses from training, so it measures what actually matters — recognising a way of
describing a problem the model has never read.

The lexicon blend is what makes that second column work:

| System | Category accuracy on unseen phrasings |
|---|---|
| Classifier alone | 0.525 |
| **Classifier + lexicon blend** | **0.821** |

Civic vocabulary is a small, stable, closed domain. A model trained on limited phrasings
has no representation for "nallah" or "culvert" if training never contained them — a
curated word list does. The blend weight (α = 0.60) was chosen by sweeping both evaluation
regimes: `python -m app.ml.tune_blend`.

---

## Architecture

```
Citizen UI ─┐
            ├──▶ FastAPI ──▶ ComplaintManager ──▶ AIPipeline ──▶ Database
Admin UI ───┘    (HTTP)      (business logic)     (ML / LLM)     (Postgres)
                     │                                               │
                     └──────▶ StatisticsService ─────────────────────┘
                              (analytics + interpretation)
```

Layers, and what each is responsible for:

| Layer | Classes | Responsibility |
|---|---|---|
| API | `app/api/v1/*` | HTTP only — validation, auth, status codes. No business logic. |
| Services | `ComplaintManager` | Intake, routing, lifecycle transitions, audit trail |
| | `StatisticsService` | Descriptive stats, quartiles, distributions, trends |
| | `NotificationManager` | Status-change messages |
| AI | `AIAnalyzer` (ABC) | The contract: `analyze(text, location) -> AIResult` |
| | `MLAnalyzer` / `LLMAnalyzer` (Claude) / `OpenAICompatibleAnalyzer` (DeepSeek) / `RuleAnalyzer` | The implementations |
| | `llm_shared.py` | Prompts, schema and result-building shared by both LLM providers, so they cannot drift apart |
| | `AIPipeline` | Composition + fallback (itself an `AIAnalyzer`) |
| | `DuplicateDetector` | Cosine similarity over recent open complaints |
| Data | SQLAlchemy models, `DatabaseManager` | Persistence and sessions |

The AI is **inside** the workflow: a complaint cannot be created without passing through
`AIPipeline`. Services take their dependencies through the constructor, so tests inject a
stub analyzer and assert on behaviour without touching a model file.

### Project layout

```
app/
├── api/v1/           # Route handlers
├── core/             # Config, logging, security, exceptions
├── db/               # Engine, session, seeding
├── ml/               # Dataset generator, training, lexicon, preprocessing
│   └── artifacts/    # Trained .joblib models (committed — see below)
├── models/           # SQLAlchemy entities
├── schemas/          # Pydantic request/response contracts
└── services/         # Business logic
    └── ai/           # The analyzer classes
alembic/              # Migrations
reports/              # Generated AI evaluation report
tests/                # 85 tests
```

---

## Quick start

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m app.ml.train        # ~70s — trains the models and writes the evaluation report
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API.

Default admin: `admin@civic.gov` / `admin123` (change via `.env`).
On first boot the app seeds 7 departments and ~180 demo complaints so the dashboard has
data immediately. It skips seeding if any complaint already exists.

Configuration is documented in [`.env.example`](.env.example) — every value has a working
default, so an empty `.env` runs fine.

### Retraining

```bash
python -m app.ml.train --samples 9000 --seed 7   # deterministic given the seed
python -m app.ml.tune_blend                       # re-sweep the lexicon blend weight
```

---

## API reference

Base path `/api/v1`. Interactive docs at `/docs`.

### Public — no authentication

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/complaints` | Submit a complaint. Runs AI inline and returns the triage. |
| `GET` | `/complaints/track/{reference_code}` | Track by `CIV-XXXXXX` code (case-insensitive) |
| `GET` | `/analytics/overview` | KPIs + distributions + interpretation |
| `GET` | `/analytics/distribution/{dimension}` | `category` \| `priority` \| `status` \| `location` \| `department` |
| `GET` | `/analytics/resolution-time` | Full descriptive stats, quartiles, IQR, outliers |
| `GET` | `/analytics/trends?days=30` | Daily volume with 7-day moving average |
| `GET` | `/analytics/departments` | Per-department performance |
| `GET` | `/departments` | List service departments |
| `GET` | `/ai/status` | Which engine is live, model version, measured accuracy |
| `POST` | `/ai/assistant` | Natural-language question over live data |
| `GET` | `/health` | Health check (database + AI) |

### Admin — requires `Authorization: Bearer <token>`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/login` | Exchange credentials for a JWT |
| `GET` | `/auth/me` | Current user |
| `GET` | `/complaints` | List with filters + pagination |
| `GET` | `/complaints/{id}` | Full detail |
| `PATCH` | `/complaints/{id}` | Status, assignment, AI overrides |
| `GET` | `/complaints/{id}/duplicates` | Likely duplicates |
| `POST` | `/complaints/{id}/duplicate-of/{original_id}` | Close as duplicate |
| `POST` | `/ai/complaints/{id}/reanalyze` | Re-run the AI (preserves human overrides) |

**Filters on `GET /complaints`:** `status`, `category`, `priority`, `department_id`,
`location`, `q` (free-text), `date_from`, `date_to`, `overdue_only`, `page`, `page_size`,
`sort`.

### Example

```bash
curl -X POST http://localhost:8000/api/v1/complaints \
  -H 'Content-Type: application/json' \
  -d '{"description":"A burst water pipeline is flooding the main road near the hospital and ambulances cannot pass.","location":"Hospital Road, Ward 4"}'
```

```jsonc
{
  "complaint": {
    "reference_code": "CIV-5U2PKC",
    "category": "water",
    "priority": "critical",
    "status": "assigned",
    "assigned_department": { "name": "Water Supply" },
    "ai_summary": "Critical priority water supply issue at Hospital Road, Ward 4: A burst water pipeline is flooding the main road…",
    "ai_output": {
      "category_confidence": 0.74,
      "priority_confidence": 0.99,
      "category_alternatives": [{ "label": "drainage", "confidence": 0.19 }],
      "engine": "ml",
      "model_version": "1.0.0",
      "processing_ms": 11.4
    }
  },
  "possible_duplicates": [],
  "message": "Complaint CIV-5U2PKC received and routed to Water Supply."
}
```

### Errors

One envelope everywhere, so the frontend handles failures uniformly:

```json
{ "error": { "type": "invalid_transition", "message": "…", "detail": {}, "request_id": "…" } }
```

`validation_error` (422) · `not_found` (404) · `authentication_error` (401) ·
`permission_denied` (403) · `invalid_transition` (409) · `database_error` (503)

---

## Statistics engine

`StatisticsService` implements the Statistics benchmark. Every endpoint returns an
`interpretation` string generated from the numbers — the requirement is to *explain* the
statistics, not just display them.

**Computed:** mean, median, mode, min, max, range, variance, standard deviation ·
Q1/Q2/Q3, IQR, Tukey fences, named outliers · frequency distributions across five
dimensions · daily trends with a 7-day moving average · per-department performance ·
SLA breach rate · AI override rate.

Real output:

> Across 160 resolved complaints the average resolution time is 41.7 hours and the median
> is 20.0 hours. **The mean sits 2.1× above the median, so a minority of very slow cases is
> dragging the average up** — most complaints are handled faster than the average suggests.
> […] **16 complaints (10.0%) fall outside the Tukey fences, taking longer than 95.6 hours.**
> These are statistical outliers rather than slow-but-normal cases […]

> **Prioritisation is working as intended** — Critical complaints are resolved in a median
> of 4.6 hours versus 109.6 hours for Low priority.

Two deliberate choices:

- **Guards on sample size.** Quartiles below n=4 are an artefact of interpolation, so the
  service declines to report them rather than returning a quotable number. Variance of a
  single observation is `null`, not `0`.
- **Mode on binned values.** Raw mode on continuous data is meaningless (every value is
  unique). Values are binned to the nearest hour first, and the interpretation says so.

---

## Testing

```bash
pytest              # 85 tests, ~20s
pytest -k ai        # AI layer only
ruff check .
```

Coverage is aimed at the things that actually break:

- **Statistics** — every value checked against hand-computed expectations, including the
  n<4 quartile guard and the empty-dataset path.
- **AI fallback** — both engines failing must still produce a usable result. Real-model
  tests run against the trained artifacts and skip cleanly when they are absent.
- **API** — the full citizen journey, the admin lifecycle, illegal status transitions,
  auth boundaries, and that account enumeration is not possible via the login error.

---

## Deployment

### Render (blueprint)

> ⚠️ **Use "New → Blueprint", not "New → Web Service".** A plain Web Service ignores
> `render.yaml` entirely and falls back to Render's default start command
> (`gunicorn your_application.wsgi`), which fails with
> `ModuleNotFoundError: No module named 'your_application'`. If you see that, or the build
> log shows Python 3.14 instead of the pinned 3.12.7, or `alembic upgrade head` never ran —
> the blueprint was not applied. Delete the service and recreate it as a Blueprint.

1. Push this directory to its own GitHub repository (`render.yaml` must be at the repo
   root — it is).
2. Render → **New → Blueprint** → select the repo. [`render.yaml`](render.yaml) provisions
   a free Postgres instance and the web service, and runs `alembic upgrade head` on build.
3. Set the two `sync: false` variables in the dashboard:
   - `ADMIN_PASSWORD` — change it before sharing the URL.
   - `CORS_ORIGINS` — your Vercel URL. **Without this the frontend gets CORS errors on
     every request**, which looks like the API being down.
4. Optionally set **one** of `DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY` to enable
   LLM-written summaries and the natural-language assistant. Leave both blank and the app
   runs entirely on the local models at zero cost — nothing breaks.

Notes on the free tier:

- **Model artifacts are committed to the repo** and loaded at boot. Training during build
  would exceed the free tier's build memory and add minutes to every deploy for an
  identical result.
- **One gunicorn worker.** Each worker loads its own copy of the models; two would risk an
  OOM restart on a 512 MB instance.
- **Cold starts.** Free services sleep after ~15 minutes idle and take ~50s to wake. Hit
  `/health` before a demo.
- **PostgreSQL, not SQLite** — Render's filesystem is ephemeral, so a SQLite file would be
  wiped on every restart, deleting anything submitted during judging.

A [`Dockerfile`](Dockerfile) is included for Railway / Fly.io / Cloud Run.

### External PostgreSQL (Neon, Supabase, …)

Any hosted Postgres works, and it is the fastest way to give an already-running Render
service a real database — no Blueprint, no redeploy, no service to recreate:

1. Create a project at [neon.tech](https://neon.tech) (free tier; unlike Render's free
   database it does not expire after 30 days).
2. Copy the connection string.
3. Render → your service → **Environment** → set `DATABASE_URL` to it → **Save**.

The service restarts, `alembic upgrade head` builds the schema, and the demo data seeds
on first boot. **Paste the string exactly as the provider prints it** — including
`?sslmode=require&channel_binding=require`. Editing it is what breaks it.

Why the app can accept it verbatim: SQLAlchemy's asyncpg dialect forwards every query
parameter to `asyncpg.connect()` as a keyword argument, and asyncpg accepts none of
libpq's spellings — an unedited Neon URL would raise `TypeError: connect() got an
unexpected keyword argument 'sslmode'` on the first connection, i.e. *inside the
migration step*, hundreds of lines after the config parsed cleanly.
[`normalise_database_url`](app/core/config.py) therefore rewrites the scheme, translates
`sslmode` to asyncpg's `ssl` (translated, never dropped — dropping it would silently
downgrade the connection to plaintext), and strips the libpq-only parameters. Both the
app and Alembic read the same `settings.DATABASE_URL`, so one rule covers both.

`tests/test_database_url.py` asserts the resulting kwargs against `asyncpg.connect`'s
live signature, so the check stays honest across driver upgrades.

**Hardening the connection (optional).** Neon hands you `sslmode=require`, which asyncpg
implements with `verify_mode = CERT_NONE` — encrypted, but the certificate is never
checked, so an attacker able to intercept the route could present any certificate at all.
Changing that one parameter to `sslmode=verify-full` closes the gap and costs nothing;
Neon's certificate is issued by Let's Encrypt and validates against the system trust
store. [`app/db/engine_options.py`](app/db/engine_options.py) supplies that trust store,
because asyncpg otherwise resolves `verify-ca`/`verify-full` against
`~/.postgresql/root.crt` and raises when the file is absent — which on a container it
always is, leaving the *strongest* setting as the only one that cannot boot.

That module exists rather than the logic living in `DatabaseManager` because **Alembic
builds its own engine**. Sharing `settings.DATABASE_URL` is not sufficient: an
`SSLContext` is an object, so it cannot travel inside a connection string, and while the
app connected happily under `verify-full`, `alembic upgrade head` failed — during deploy,
before the app ever started. Both paths now call `connect_args_for()`.

Verified against Neon: `require`, `verify-ca` and `verify-full` all connect *and*
migrate, and the verifying context rejects hostname mismatches, self-signed and expired
certificates.

**Pooled vs direct endpoint.** Neon and Supabase both offer two, and show the pooled one
first. Either works: [`DatabaseManager`](app/db/session.py) detects the `-pooler` host and
disables asyncpg's prepared-statement cache, which PgBouncer's transaction mode would
otherwise break with `prepared statement "__asyncpg_stmt_1__" does not exist` — an error
that appears only intermittently, under load. The direct endpoint is marginally faster
here, since the app keeps its own small connection pool.

### The `your_application` shim

Render's placeholder start command, `gunicorn your_application.wsgi`, is what a service
gets when it was created as a plain Web Service rather than from `render.yaml`. It fails
with a `ModuleNotFoundError` that says nothing about the real problem.

Two files make that command work anyway:

- **`gunicorn.conf.py`** — auto-loaded by gunicorn from the working directory. Sets
  `worker_class` to the uvicorn ASGI worker and binds `0.0.0.0:$PORT`, neither of which
  the placeholder command supplies. (Explicit `-k`/`-b` flags still win, so the documented
  start commands are unaffected.)
- **`your_application/wsgi.py`** — re-exports the real app as `application`.

This is not a degraded path: same server, same ASGI worker, same application object, full
lifespan support. Only the module path differs, and startup logs a warning naming the
proper fix. **It does not remove the reason to use a Blueprint** — a manually-created
service still has no PostgreSQL, so it runs on an ephemeral SQLite file and loses every
complaint on restart.

### Deploy troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'your_application'` | Created as a Web Service, so `render.yaml` was ignored | Should no longer occur — `your_application/wsgi.py` resolves the placeholder command (see below). Still worth fixing the start command properly. |
| Log shows `started_via_compatibility_shim` | Running on Render's placeholder start command | Works, but set the start command or redeploy as a Blueprint — otherwise there is no PostgreSQL and data is lost on every restart |
| Every complaint disappears after a restart | No `DATABASE_URL`, so the app fell back to SQLite on an ephemeral disk | Point `DATABASE_URL` at a Neon (or Render) Postgres — see [External PostgreSQL](#external-postgresql-neon-supabase-) |
| `TypeError: connect() got an unexpected keyword argument 'sslmode'` | A Postgres URL reached asyncpg without normalisation | Should no longer occur. If it does, the URL was hand-edited into an unusual shape — paste the provider's string verbatim |
| `prepared statement "__asyncpg_stmt_1__" does not exist`, intermittently | Transaction pooler with statement caching on | Should no longer occur on `-pooler` hosts; otherwise switch to the direct endpoint |
| App crashes at startup with a `SECRET_KEY` validation error | `ENVIRONMENT=production` with a short or default signing key | Let Render generate it (`generateValue: true`), or set 32+ random bytes |
| `/health` reports `"ml_loaded": false` | Model artifacts missing from the repo | They are committed under `app/ml/artifacts/`; check they were not gitignored |
| Frontend loads but every request fails, browser console says CORS | `CORS_ORIGINS` does not include the site's origin | Set it on Render to the origin — scheme and host only. A trailing slash is tolerated; a path is not. Startup logs `cors_origins=[…]`, and warns `cors_origins_not_configured` when production still allows only localhost |
| Requests fail in the browser but the same call works from `curl` | Same as above — a CORS rejection is invisible server-side | The request really does run and return 200; only the browser discards the response for want of `access-control-allow-origin`. Check the startup log line, not the request log |
| First request after idle takes ~50s | Free instance cold start | Hit `/health` before demoing |

---

## Limitations

Stated plainly, because the spec requires explaining what the AI cannot do:

1. **Training data is generated, not observed.** No public labelled corpus of municipal
   complaints exists. Even the unseen-phrasing score measures generalisation within one
   generator's idea of how citizens write; real accuracy will be lower.
2. **English-centric.** Some code-mixed samples are included, but complaints written
   mainly in Hindi, Marathi or Tamil will classify poorly.
3. **Priority reads tone, not truth.** The model infers urgency from how alarmed the
   writer sounds. A calm report of a genuinely dangerous problem gets under-prioritised.
4. **No fact verification.** The system triages *claims*. It cannot tell whether a
   reported leak exists, and will confidently classify a fabricated complaint.
5. **Duplicate detection is lexical.** It catches reworded matches, not two descriptions
   of the same incident that share no vocabulary. Nothing is auto-merged; candidates are
   surfaced for a human to decide.
6. **Notifications are logged, not sent.** The `NotificationManager` seam is real and
   correct, but wiring a live email/SMS provider was out of scope.
7. **LLM output is not measured.** The accuracy figures above are for the local
   classifiers, which have a held-out test set. When an LLM writes the summary there is no
   equivalent benchmark — it is judged only by reading it. That is another reason the
   measurable classifiers keep ownership of category and priority.

The dashboard tracks the **admin override rate** — how often a human corrects the model on
real complaints. After launch, that number is the honest accuracy figure, not the held-out
test score.

### Python version

`.python-version` pins 3.12.7 — the version the test suite and the committed model
artifacts were built against. Render reads this file for native Python services whether or
not `render.yaml` is applied, so it holds even if the service was created manually.
`render.yaml` sets the same value via `PYTHON_VERSION` for belt and braces.

The pin is not cosmetic: the `.joblib` artifacts are unpickled at boot, and keeping the
runtime identical to the one that produced them removes a whole class of "works locally,
fails on deploy" problems.
