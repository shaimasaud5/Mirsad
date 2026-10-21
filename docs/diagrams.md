# Technical diagrams

Both diagrams describe the code in this repository. GitHub renders Mermaid
automatically; you can also paste them into https://mermaid.live to export PNG/SVG.

## 1. Development flow

```mermaid
flowchart LR
    A[Issue / task on GitHub] --> B[Feature branch]
    B --> C[Code in VS Code<br/>AI-assisted, reviewed by a teammate]
    C --> D[Run locally in MOCK mode<br/>MIRSAD_MOCK=1, no API cost]
    D --> E{pytest<br/>18 unit + proxy tests}
    E -- fail --> C
    E -- pass --> F[Offline eval<br/>python -m eval.run_eval]
    F --> G{Routing accuracy and<br/>loop metrics OK?}
    G -- no, adjust rules on DEV set only --> C
    G -- yes --> H[Live test with Groq<br/>MIRSAD_MOCK=0]
    H --> I[Pull request + review]
    I --> J[Merge to main]
    J --> K[Run demo:<br/>uvicorn proxy + streamlit]
    K --> L[Watch dashboard,<br/>log issues] --> A
```

## 2. Monitoring flow (one agent request)

```mermaid
sequenceDiagram
    autonumber
    participant U as User (Chat page)
    participant A as LangChain agent
    participant T as Local tools
    participant P as Mirsad proxy (FastAPI)
    participant G as LoopGuard
    participant R as Router
    participant L as LiteLLM to Groq
    participant D as SQLite (requests table)
    participant M as Dashboard (Streamlit)

    U->>A: task
    loop until final answer (max 8 model calls)
        A->>P: POST /v1/chat/completions + X-Session-ID
        P->>G: check(session, messages)
        alt same tool call failed >= 3 times
            G-->>P: blocked
            P->>D: INSERT status=blocked, est_saved_cost
            P-->>A: 429 Loop blocked
            A-->>U: stopped, new session
        else ok
            P->>R: classify(latest user message)
            R-->>P: simple or complex (+ score, reason)
            P->>L: completion(small or big model)
            alt provider error / rate limit
                L-->>P: exception
                P->>D: INSERT status=error
                P-->>A: 502
            else success
                L-->>P: reply + token usage
                P->>P: latency, actual cost, GPT-4o baseline cost
                P->>D: INSERT status=success
                P-->>A: OpenAI-format reply
                opt reply asks for a tool
                    A->>T: run tool locally
                    T-->>A: result or "ERROR: ..."
                end
            end
        end
    end
    A-->>U: final answer + cost summary
    loop every 2 seconds
        M->>D: SELECT * FROM requests
        M->>M: savings %, error rate, loops, latency p95
        M->>M: alert rules (last 5 min)
        M-->>U: KPIs, charts, red banner if triggered
    end
```

### Where each metric is calculated

| Metric | Type (brief) | Source column(s) | Code |
|---|---|---|---|
| Requests, error rate | Application | `status` | `monitoring/metrics.py::summarize` |
| Loops blocked | Application | `status = blocked` | same |
| Avg / p95 latency per model | Performance | `latency_ms`, `model` | `ui/app.py` latency chart |
| Route mix, routing accuracy | AI quality | `route`; offline eval set | `eval/run_eval.py` |
| Tokens, actual vs baseline cost, savings % | Cost | `input_tokens`, `output_tokens`, `actual_cost`, `baseline_cost` | `proxy/cost.py` |
| Estimated loop savings | Cost | `est_saved_cost` | `proxy/main.py` (blocked branch) |
| Red alert | Alerting | last 5 min of rows | `monitoring/metrics.py::alerts` |
