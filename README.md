# Mirsad (مرصاد)

A proxy that sits between an AI agent and its model provider and does three jobs:

1. **Routes** every task to a small (cheap) or big (strong) model.
2. **Stops** agents that keep repeating a failing tool call.
3. **Records** every call so a dashboard can show cost, errors and speed.

It costs nothing to run: answers come from Groq's free tier, routing decisions come from a small model running locally
with Ollama, and the dashboard's costs are **simulated** (real token counts priced as GPT-4o-mini vs GPT-4o).

## How it works

```
Chat page -> LangChain agent -> Mirsad proxy -> Groq (small or big model)
                                   |  \-> local Qwen classifier (Ollama)
                                   \-> SQLite -> Streamlit dashboard
```

For each request the proxy does this, in order:

1. **Loop guard**: the same tool call failed 3 times with the same input? Reject it (HTTP 429). No model is called.
2. **Router**: a local Qwen model reads the user's message and answers `simple` or `complex`. Mirsad then picks the small or big
   Groq model. Qwen never answers the user. This happens once per user message; later tool steps reuse the decision.
3. **Answer**: the chosen Groq model replies (or a fake reply in mock mode).
4. **Log**: tokens, time and priced cost are saved to SQLite.

One task can produce several requests, because the agent calls the model again after each tool result. The dashboard counts
requests (model calls), not tasks; calls of one task share a `turn_id`.

**If something fails:** Ollama down or slow -> Mirsad uses the old keyword rules and records why.
Groq rejects an API key (401) -> it tries the next backup key. Groq rate limit (429) -> it does **not** switch keys
(limits are per account), it waits briefly or returns a clear error. It never silently switches to mock answers.

## Run it

You need Python 3.11+, [Ollama](https://ollama.com) installed and running, and (for real answers) a free
[Groq key](https://console.groq.com/keys).

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M            # exact name; plain qwen3:4b will not work
cp .env.example .env                                  # Windows: copy .env.example .env
```

Edit `.env`:

| Want | Set |
|---|---|
| Try without a key | `MIRSAD_MOCK=1` (fake answers, routing still real) |
| Real answers | `MIRSAD_MOCK=0` and `GROQ_API_KEY=...` |

Start two terminals (venv active in both):

```bash
uvicorn proxy.main:app --port 8000     # 1: the proxy
streamlit run ui/app.py                # 2: chat + dashboard at http://localhost:8501
```

Check http://localhost:8000/health: you want `classifier_status: ready`. After **any** `.env` change, restart the proxy.

| Problem | Fix |
|---|---|
| `classifier_status` says `model not downloaded` | run the `ollama pull` line above |
| `classifier_status` says `unreachable` | start Ollama |
| `No module named 'mcp'` | `pip install mcp`, restart the proxy |

## Demo (3 minutes)

1. Dashboard: **Reset data**.
2. Chat: **Simple task** -> should go to the small model.
3. Chat: **Complex task** -> should go to the big model.
4. Chat: **Loop scenario** -> `fetch_report` always fails; Mirsad blocks the agent on the 4th attempt.
5. Dashboard: red alert, loops blocked, hypothetical savings, latency per model, and a log with the reason for every route.

The line under each answer shows who classified, who answered, live or mock, where the token counts came from, and the
priced cost versus an all-GPT-4o baseline. The two costs are equal when the task went to the big model.

## What is where

| File | Job |
|---|---|
| `config.py` | every setting, threshold and price |
| `proxy/main.py` | the request pipeline |
| `proxy/routing.py`, `proxy/classifier.py` | one routing decision per user message; the local Qwen classifier |
| `proxy/router.py` | old keyword rules (a mode, and the fallback) |
| `proxy/loop_guard.py` | repeated-failure detection |
| `proxy/provider.py`, `proxy/credentials.py` | Groq call and API-key failover |
| `proxy/cost.py`, `proxy/db.py`, `proxy/mock.py` | pricing, SQLite, fake model |
| `agent/` | the LangChain agent and its tools (`list_files`, `read_file`, `calculator`, `fetch_report` which always fails) |
| `monitoring/metrics.py`, `ui/app.py` | dashboard numbers, alerts and pages |
| `eval/`, `tests/` | offline evaluation and 65 tests |
| `data/files/` | sample files the agent reads |
| `docs/diagrams.md` | flow diagrams |

`MIRSAD_ROUTING_MODE` picks who routes: `rules` (keywords), `shadow` (rules decide, Qwen is only recorded),
`local_llm` (Qwen decides, rules are the fallback).

## Testing and evaluation

```bash
pytest -q                  # 65 tests, offline
python -m eval.run_eval    # evaluates the keyword rules and the loop guard
```

| Metric | Result |
|---|---|
| Routing accuracy, **keyword rules**, held-out set | 92.5% (37/40) |
| Routing accuracy, **keyword rules**, dev set | 96.7% (58/60) |
| Loops blocked / false alarms | 10/10 and 0/6 |

The Qwen classifier is **not** covered by these numbers. Its only evaluation is preliminary, on examples written by an AI,
so we claim no accuracy figure for it. Cost savings come from your own runs of the agent.

## Limits

- Costs are simulated; they are not Groq invoices.
- The classifier can be wrong, and it adds about 1-3 s per user message.
- The loop guard and routing cache live in memory and reset when the proxy restarts.
- The loop guard only catches repeats with identical input.
- No streaming; one machine only (SQLite).
- Groq's free tier allows roughly 30 requests per minute for the whole account.
- LiteLLM is pinned to 1.106.0 (1.82.7 and 1.82.8 were compromised on PyPI in March 2026).
