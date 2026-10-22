# Mirsad (مرصاد)

A proxy that sits between an AI agent and its model provider and does three jobs:

1. **Routes** every task to a small (cheap) or big (strong) model.
2. **Stops** agents that keep repeating a failing tool call.
3. **Records** every call so a dashboard can show cost, errors and speed.

It costs nothing to run: answers come from Groq's free tier, routing decisions come from a small model running locally
with Ollama, and the dashboard's costs are **estimates** (provider-reported token counts priced at Groq's public list prices; the free tier is not billed).

## How it works

```
Browser (signed in) -> web server (port 8501) -> LangChain agent -> Mirsad proxy (port 8000, internal) -> Groq
        |                      |                                          |  \-> local Qwen classifier (Ollama)
        |                      \-> SQLite: users, conversations, usage      \-> SQLite request log (owner = verified user)
        \-> Supabase Auth (email + password)
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
uvicorn proxy.main:app --port 8000           # 1: the proxy (internal: only the web server may call it)
uvicorn ui.web_server:app --port 8501        # 2: sign-in, chat and dashboard at http://localhost:8501
```

Sign-in needs the Supabase and internal-secret settings in `.env`; see **Accounts and permissions** below. Without them the
web server answers with a clear setup error (it never falls back to anonymous access). The old Streamlit page is retired
because it could not sign users in. After **any** `.env` change, restart both servers. Signed in as an administrator,
the status pill and dashboard show `classifier_status`: you want `ready`.

| Problem | Fix |
|---|---|
| `classifier_status` says `model not downloaded` | run the `ollama pull` line above |
| `classifier_status` says `unreachable` | start Ollama |
| `No module named 'mcp'` | `pip install mcp`, restart the proxy |

## Accounts and permissions

Users sign up and sign in with email + password through **Supabase Auth**. Mirsad stores no passwords. Application data
(conversations, requests, roles) stays in SQLite.

**How it works**

- The browser holds one opaque, `HttpOnly`, `SameSite=Lax` cookie. Supabase tokens stay on the server, encrypted at rest;
  identity always comes from that server-side session, never from anything the browser sends.
- Authenticated changes need a CSRF token and a same-origin request. Sessions end after 12 h idle / 7 days, and are
  refreshed or re-checked with Supabase automatically.
- Every conversation, message and request record is owned by the verified user. Ordinary users see only the chat: replies
  show no route, model, token or cost details (admins see those in the dashboard). Monitoring, feedback analytics, the user
  list and reset are admin-only, enforced on the server. Old records with no owner are admin-only.
- The proxy on port 8000 accepts only the web server (shared `MIRSAD_INTERNAL_TOKEN`), so a browser cannot call it directly
  or forge an owner. Keep it bound to `127.0.0.1` (the default) and never expose port 8000.
- New accounts are always `user`. Admins are assigned only on the server (below).

**`.env` values** (see `.env.example`)

| Variable | Meaning |
|---|---|
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY` | from Supabase: Project Settings > API (the publishable key is safe to use; no service-role key is needed) |
| `MIRSAD_INTERNAL_TOKEN` | long random secret shared by the web server and the proxy |
| `MIRSAD_SESSION_SECRET` | long random secret that encrypts stored session tokens |
| `MIRSAD_PUBLIC_URL` | the address users open, e.g. `http://localhost:8501`; use `https://...` in production (cookies become `Secure`) |

Generate a secret with `python -c "import secrets; print(secrets.token_urlsafe(40))"`.

**Supabase dashboard settings** (Authentication)

1. *URL Configuration* > **Site URL**: `http://localhost:8501` (your real HTTPS address in production).
2. *URL Configuration* > **Redirect URLs**: add `http://localhost:8501/auth/callback` (and the production equivalent). Use exact paths, not wildcards.
3. *Sign In / Providers* > **Email**: enabled, **Confirm email** on (recommended), minimum password length 8 or more.
4. *Emails*: the default templates work as they are. Optional hardening: change the link to
   `{{ .SiteURL }}/auth/callback#token_hash={{ .TokenHash }}&type=signup` (use `type=recovery` for the reset template); the page supports both shapes.
5. Email delivery: Supabase's built-in sender only mails **members of your Supabase organization** and allows about 2 messages per hour.
   For real users, configure a custom SMTP server.

Email-link tokens arrive in the URL *fragment* (`#...`), which browsers never send to servers, so they cannot appear in server logs. The page removes them
from the address bar immediately and posts them to the server, which verifies them with Supabase.

**First administrator**

1. Sign up, confirm the email, and sign in once (this creates your local user row).
2. On the server machine run: `python -m scripts.manage_admin grant you@example.com`
3. Reload the page. Other commands: `list`, `revoke EMAIL` (the last admin cannot be revoked). Nothing in the browser can grant this role.

## Demo (3 minutes)

1. Dashboard (admin): **Reset data**.
2. Chat: **Simple task** -> should go to the small model.
3. Chat: **Complex task** -> should go to the big model.
4. Chat: **Loop scenario** -> `fetch_report` always fails; Mirsad blocks the agent on the 4th attempt.
5. Dashboard (admins, default **Live**): user turns, model calls and provider attempts are counted separately; tokens and
   estimated cost use provider-reported usage only; latency is split into local classification, model call, provider
   attempt and end-to-end turn. The help section on the dashboard defines every number, time basis and limitation.

Cost figures are **estimated inference cost at list price**, not an invoice. The comparison re-prices the *same recorded
tokens* at the configured BIG model's rates; it is hypothetical, can be negative, and does not prove real savings.
Prices live in `config.MODEL_PRICES` (source and retrieval date next to it): update them there when Groq changes a rate.

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
| `monitoring/metrics.py`, `monitoring/feedback.py` | dashboard numbers and alerts; admin feedback analytics |
| `ui/web_server.py`, `ui/web/index.html` | the web app: sign-in, chat with answer feedback, admin dashboard and feedback monitoring |
| `auth/` | Supabase client, server-side sessions, roles, per-user conversation storage, answer feedback |
| `scripts/manage_admin.py` | grant or revoke the admin role (server-side only) |
| `ui/app.py` | retired Streamlit page (refuses to run) |
| `eval/`, `tests/` | offline evaluation and the tests |
| `data/files/` | sample files the agent reads |
| `docs/diagrams.md` | flow diagrams |

`MIRSAD_ROUTING_MODE` picks who routes: `rules` (keywords), `shadow` (rules decide, Qwen is only recorded),
`local_llm` (Qwen decides, rules are the fallback).

## Answer feedback

Each completed answer has **Copy**, **Helpful** and **Not helpful** (with optional reason and comment). One current rating is
kept per user and answer (a database constraint), ratings can be changed or cleared, and they are restored when a
conversation is reopened. Feedback is stored in its own table and is never sent to the model.

Admins see it under **Dashboard > User feedback**: Helpful % = Helpful / answers currently rated (unrated answers never
count as Helpful), coverage = rated / completed answers, reasons, breakdowns by routing category and answering model (with
sample sizes) and a list with a detail view. It uses the dashboard's time window and its own record filter, which defaults
to **Live** so mock and legacy records do not mix into quality numbers. Ratings are opinions, not proof of accuracy.

A **File used: name** label appears under an answer only when a successful `read_file` tool result in that run is exactly the
content of that file (it says "partly read" when the 4000-character cap cut it). It is never inferred from the question or
from the model's text, and never shows a path.

## Testing and evaluation

```bash
pytest -q                  # offline
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
- Sessions, rate limits, the loop guard and routing cache live in memory or per process: one server process only.
- Deleting a conversation removes its messages but keeps the usage records (they are cost history).
- Chats that were saved in a browser by the old page (localStorage) are not imported and not shown.
- The classifier can be wrong, and it adds about 1-3 s per user message.
- The loop guard and routing cache live in memory and reset when the proxy restarts.
- The loop guard only catches repeats with identical input.
- No streaming; one machine only (SQLite).
- Groq's free tier allows roughly 30 requests per minute for the whole account.
- LiteLLM is pinned to 1.106.0 (1.82.7 and 1.82.8 were compromised on PyPI in March 2026).
