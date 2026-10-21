# Mirsad (مرصاد)

**Mirsad** (Arabic: the watchtower that misses nothing) is a smart, OpenAI-compatible proxy that sits between an AI agent and the model
provider. It **routes** each request to a cheap or a powerful model based on
complexity, **blocks** agents stuck in failing tool loops, and **logs** every
call to a live monitoring dashboard that shows cost savings, errors and latency.

Zero budget: models run on Groq's free tier; costs on the dashboard are
**simulated** with public OpenAI prices (GPT-4o-mini vs GPT-4o) applied to the
real token counts.

```
Web UI (chat) -> LangChain agent -> Mirsad proxy -> [loop guard -> router -> LiteLLM] -> Groq
                                        |
                                        +-> SQLite -> web dashboard (+ alerts)
```

## Routing modes (`MIRSAD_ROUTING_MODE` in `.env`)

| Mode | Who decides small vs big answering model |
|---|---|
| `rules` | the keyword rules in `proxy/router.py` (original behaviour) |
| `shadow` | rules decide; the local classifier's recommendation is only recorded |
| `local_llm` | a **local Qwen model via Ollama classifies** the user turn; Mirsad then picks `SMALL_MODEL` or `BIG_MODEL`. Qwen never answers. Falls back to rules (reason recorded) if Ollama is down, slow, or returns invalid output |

`local_llm` needs Ollama running with the classifier model downloaded:
`ollama pull qwen3:4b-instruct-2507-q4_K_M` (about 2.5 GB). The classifier prompt is the frozen "v1"
(`proxy/classifier.py`, `PROMPT_ID` is stored in every row). Its only evaluation so far is a **preliminary
one on an AI-authored candidate set**, not an independent human-validated benchmark.
One classification is made per user turn (the agent sends `X-Turn-ID`) and reused by every tool step.

## API-key failover (live mode only)

Set `GROQ_API_KEY` (primary) and optionally `GROQ_API_KEY_BACKUP_1`, `GROQ_API_KEY_BACKUP_2`, … in `.env`
(see `.env.example`; placeholders only). Environment changes need a **proxy restart**.

| Provider response | What Mirsad does |
|---|---|
| 401 invalid/revoked key | disables that key for the running process and retries the **same model with the same messages** on the next key |
| 429 rate limit | **never rotates keys** (limits are per organization); waits once for `Retry-After` if ≤ `PROVIDER_RETRY_AFTER_MAX_S`, otherwise returns HTTP 503 `provider_rate_limited` with `Retry-After` |
| 403, 400/404/422, 5xx, timeouts | no failover; clear error (another key would not fix it) |
| all keys rejected / none configured | HTTP 502/503 with a clear message. It never switches to mock mode |

LiteLLM/SDK retries are switched off; `PROVIDER_MAX_ATTEMPTS` (default 3) is the only retry budget.
Each provider attempt is stored in the `provider_attempts` table with a safe alias (`primary`, `backup_1`),
outcome, status and duration. Key values are never logged, stored or returned (`/health` lists aliases only).
Loop-guard blocks (HTTP 429 `loop_blocked`) are unrelated to credential failover.

## Quick start (about 10 minutes)

**1. Requirements:** Python 3.11+ and Git.

**2. Install**
```bash
git clone <your-repo-url> mirsad
cd mirsad
python -m venv venv
# macOS/Linux:  source venv/bin/activate
# Windows:      venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure**
```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Open `.env` and paste your free key from https://console.groq.com/keys.
Then open the Groq console model list and check that `SMALL_MODEL` and
`BIG_MODEL` exist and support tool calling (Groq changes its list often).

**4. Run (two terminals, venv activated in both)**
```bash
# Terminal 1 - the proxy
uvicorn proxy.main:app --port 8000

# Terminal 2 - chat + dashboard (web UI)
uvicorn ui.web_server:app --port 8501
```
Open http://localhost:8501. Start from one of the three scenario cards, then open
**Dashboard** in the sidebar. The language (العربية / English) and theme (light / dark / auto)
switches are in the top bar and in the user menu at the bottom of the sidebar. Recent chats are
kept in the browser (localStorage).

The previous Streamlit page still works (`streamlit run ui/app.py`) but is no longer maintained.

**No internet / Groq down?** Set `MIRSAD_MOCK=1` in `.env` and restart the
proxy. Everything works with simulated replies (never present mock numbers as
real results).

## Demo script (3-minute pitch)

1. Dashboard: click **Reset data**.
2. Chat: **Simple task** -> answered by the small model; the caption under the
   answer shows its cost vs GPT-4o.
3. Chat: **Complex task** -> routed to the big model.
4. Chat: **Loop scenario** -> the `fetch_report` tool always fails, the agent
   retries, Mirsad blocks it on the 4th attempt.
5. Dashboard: red alert banner, *Loops blocked = 1*, cost savings %, latency
   by model, and the request log showing *why* each route was chosen.

## Evaluation (free, no API calls)

```bash
python -m eval.run_eval      # writes eval/results/summary.md
pytest -q                    # 68 tests, all offline
```

| Metric | Result | How |
|---|---|---|
| Routing accuracy (held-out set) | 92.5% (37/40) | 40 hand-labelled prompts never used for tuning |
| Routing accuracy (dev set) | 96.7% (58/60) | 60 prompts used while designing the rules |
| Loop mitigation rate | 100% (10/10) | synthetic loops replayed through the real LoopGuard |
| Loop false-positive rate | 0% (0/6) | healthy retry patterns that must not be blocked |
| Cost savings % | from your own runs | `sum(saved) / sum(baseline)` from the SQLite log |

Before the presentation, **each team member should add ~10 prompts of their
own** to `eval/test_prompts.csv` (label them *before* running the script) so
the held-out set is not written by the same author as the rules.

## Project structure

```
config.py            all thresholds, model names and prices in one place
proxy/main.py        FastAPI endpoint: the 7-step pipeline
proxy/router.py      rule-based complexity score -> simple / complex
proxy/routing.py     routing modes (rules / shadow / local_llm), one decision per user turn
proxy/classifier.py  local Qwen classifier via Ollama (classifies only, never answers)
proxy/provider.py    Groq call through LiteLLM with API-key failover
proxy/credentials.py primary + backup Groq keys (aliases only in logs)
proxy/loop_guard.py  repeated-failure detection per session
proxy/cost.py        token pricing, baseline, savings formula
proxy/db.py          SQLite schema and writes
proxy/mock.py        offline fake model (tool calls + loops)
agent/agent.py       LangChain create_agent pointed at the proxy
agent/tools.py       list_files, read_file, calculator, fetch_report (always fails)
monitoring/metrics.py  dashboard metrics + alert rules
ui/web_server.py     FastAPI app for the web UI: /api/chat, /api/metrics
ui/web/index.html    chat + dashboard page (AR/EN, light/dark), libraries bundled in ui/web/vendor
ui/app.py            old Streamlit page (kept for reference)
eval/                dev/test prompt sets and run_eval.py
tests/               pytest suite
docs/diagrams.md     Development flow + Monitoring flow (Mermaid)
data/files/          sample files the agent can read
```

## Key design decisions

| Decision | Why | Trade-off |
|---|---|---|
| Rule-based router | Free, instant, every decision is explainable | Misses complex prompts without trigger words (see eval errors) |
| Route by latest **user** message | All steps of one task use the same model, stable for tool calling | A simple follow-up step can't drop to the small model |
| OpenAI-compatible API | Any agent integrates by changing `base_url` only | No streaming support yet |
| Loop guard on tool-call fingerprints | Catches real retry loops; healthy retries are not blocked | Different-input loops are not caught (agent step limit covers them) |
| SQLite + 3 s polling | No extra services; dashboard feels live | Single machine only |
| LiteLLM | One code path for any provider; switch with config | Extra dependency, pinned for supply-chain safety |
| Simulated OpenAI pricing | Groq free tier costs 0, so savings need a realistic price scale | Numbers are estimates, labelled as such |

## Limitations

- Costs are simulated; token counts are real (live mode) or estimated (mock mode).
- Router uses keywords, so it is language-specific (English) and can be fooled.
- Loop guard state is in memory: restarting the proxy resets counters.
- Groq free tier: ~30 requests/min and limited tokens/min per key; heavy use returns
  errors that appear on the dashboard as the error rate.
- LiteLLM is pinned to 1.106.0 (versions 1.82.7 and 1.82.8 were compromised on
  PyPI in March 2026).
