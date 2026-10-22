"""
Bounded live evaluation of ANSWERING models through the real agent + proxy (the local classifier is not involved).

    python -m eval.answering_models --live MODEL_A MODEL_B [--report path.json]

Safety / budget (all enforced in code, not by convention):
  * isolated temp database and a throw-away internal token: no user data is read or written;
  * at most MAX_RUNS agent runs in total and MAX_CALLS_PER_RUN provider calls per run (retries and credential failover
    count as calls); the provider is never called once a cap is reached;
  * the same cases, prompts and output budget (config.MAX_TOKENS) for every model.
Pass criteria are defined below, BEFORE any model is run. Without --live nothing is called.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

MAX_RUNS = 12
MAX_CALLS_PER_RUN = 3
KNOWN_TOOLS = {"list_files": set(), "read_file": {"name"}, "calculator": {"expression"}, "fetch_report": {"report_name"}}   # name -> required argument names (None = not checked)

NOT_FOUND = re.compile(r"not found|does not exist|doesn't exist|no such file|couldn't (find|read)|could not (find|read)|cannot (find|read)|"
                       r"can't (find|read)|unable to|isn't available|not available|no file|not exist|غير موجود|لا يوجد|لم أجد|لم أستطع|تعذر", re.I)
FILE_FACTS = ("jeddah", "riyadh", "pro plan", "api latency", "account manager", "q2", "marketing", "enterprise")


def _tool_calls(steps):
    return [s for s in steps if s.get("type") == "tool_call"]


def tools_valid(steps) -> list[str]:
    """Every tool call must name a known tool with the required string arguments."""
    problems = []
    for s in _tool_calls(steps):
        need = KNOWN_TOOLS.get(s.get("tool"), "unknown")
        args = s.get("args") or {}
        if need == "unknown":
            problems.append(f"unknown tool {s.get('tool')!r}")
        elif need and not all(isinstance(args.get(a), str) and args.get(a) for a in need):
            problems.append(f"invalid arguments for {s.get('tool')}: {args}")
    return problems


def _base(status, steps):
    p = []
    if status != "ok":
        p.append(f"status {status}")
    return p + tools_valid(steps)


def check_en_fact(answer, steps, status="ok"):
    p = _base(status, steps)
    if "paris" not in answer.lower():
        p.append("answer does not name Paris")
    return p


def check_ar_fact(answer, steps, status="ok"):
    p = _base(status, steps)
    if "الرياض" not in answer:
        p.append("answer does not name الرياض")
    return p


def check_translate(answer, steps, status="ok"):
    p = _base(status, steps)
    if not re.search(r"صباح\s*الخير", answer):
        p.append("translation lacks صباح الخير")
    return p


def check_format(answer, steps, status="ok"):
    p = _base(status, steps)
    lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^[-*•]\s+\S", ln)]
    if len(bullets) != 3 or len(lines) != 3:
        p.append(f"expected exactly 3 bullet lines, got {len(lines)} lines / {len(bullets)} bullets")
    for w in ("apples", "oranges", "bananas"):
        if w not in answer.lower():
            p.append(f"missing {w}")
    return p


def check_file_read(answer, steps, status="ok"):
    from agent.tools import verify_file_read
    p = _base(status, steps)
    reads = [s for s in _tool_calls(steps) if s["tool"] == "read_file"]
    if not any((s.get("args") or {}).get("name") == "team_notes.txt" for s in reads):
        p.append("never called read_file('team_notes.txt')")
    results = [s for s in steps if s.get("type") == "tool_result" and s.get("tool") == "read_file"]
    if not any(verify_file_read("team_notes.txt", r.get("result")) for r in results):
        p.append("no verified successful read of team_notes.txt")
    hits = [f for f in FILE_FACTS if f in answer.lower()]
    if len(hits) < 2:
        p.append(f"answer reflects fewer than 2 facts from the file ({hits})")
    return p


def check_missing_file(answer, steps, status="ok"):
    p = _base(status, steps)
    if not any(s["tool"] == "read_file" for s in _tool_calls(steps)):
        p.append("never tried read_file")
    if not NOT_FOUND.search(answer.replace("’", "'")):       # curly apostrophes are normal model output
        p.append("answer does not say the file is missing / unreadable")
    stripped = answer.replace("budget_2031.csv", "").replace("2031", "")
    if re.search(r"\d", stripped):
        p.append("answer contains numbers that are not in the prompt: possible fabricated file content")
    return p


CASES = [
    {"id": "en_fact", "prompt": "What is the capital of France? Answer in one short sentence.", "check": check_en_fact},
    {"id": "ar_fact", "prompt": "ما هي عاصمة المملكة العربية السعودية؟ أجب بجملة قصيرة.", "check": check_ar_fact},
    {"id": "translate", "prompt": "Translate to Arabic: Good morning, how are you?", "check": check_translate},
    {"id": "format", "prompt": "Format these as a markdown bullet list, nothing else: apples, oranges, bananas", "check": check_format},
    {"id": "file_read", "prompt": "Read team_notes.txt and tell me in one sentence what it says.", "check": check_file_read},
    {"id": "missing_file", "prompt": "Read the file budget_2031.csv and summarize it.", "check": check_missing_file},
]


# ------------------------------------------------------------------ live runner
def run_live(models: list[str], report_path: str | None = None) -> dict:
    total_runs = len(models) * len(CASES)
    if total_runs > MAX_RUNS:
        raise SystemExit(f"{total_runs} agent runs requested; the authorized maximum is {MAX_RUNS}")
    tmp = tempfile.mkdtemp(prefix="mirsad-eval-")
    os.environ.update({"MIRSAD_DB": str(Path(tmp) / "eval.db"), "MIRSAD_MOCK": "0", "MIRSAD_ROUTING_MODE": "rules",
                       "MIRSAD_INTERNAL_TOKEN": "eval-" + uuid.uuid4().hex})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import uvicorn

    import config
    from agent.agent import run_agent
    from proxy import cost, provider
    from proxy.main import app as proxy_app
    from monitoring import metrics

    port = 8700 + uuid.uuid4().int % 90
    config.PROXY_URL = f"http://127.0.0.1:{port}/v1"
    server = uvicorn.Server(uvicorn.Config(proxy_app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    real, state = provider._completion, {"run_calls": 0, "calls": []}

    def capped(**kw):
        if state["run_calls"] >= MAX_CALLS_PER_RUN:
            raise RuntimeError("eval cap: model-call limit for this run reached")      # never reaches the provider
        state["run_calls"] += 1
        resp = real(**kw)
        u = getattr(resp, "usage", None)
        det = getattr(u, "completion_tokens_details", None)
        state["calls"].append({"model": kw.get("model"), "finish_reason": resp.choices[0].finish_reason,
                               "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None),
                               "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None})
        return resp
    provider._completion = capped
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "max_tokens": config.MAX_TOKENS, "runs": []}
    try:
        for model in models:
            config.SMALL_MODEL = config.BIG_MODEL = model        # eval-only, in memory: every route uses the model under test
            for case in CASES:
                state["run_calls"], state["calls"] = 0, []
                t0 = time.perf_counter()
                res = run_agent(case["prompt"], session_id="eval_" + uuid.uuid4().hex[:6], user_id="eval-user", conversation_id=None)
                problems = case["check"](res["answer"], res["steps"], res["status"])
                out["runs"].append({"model": model, "case": case["id"], "passed": not problems, "problems": problems,
                                    "status": res["status"], "answer": res["answer"][:400], "turn_id": res["turn_id"],
                                    "seconds": round(time.perf_counter() - t0, 2), "provider_calls": list(state["calls"]),
                                    "tool_calls": [{"tool": s["tool"], "args": s["args"]} for s in _tool_calls(res["steps"])]})
    finally:
        provider._completion = real
        server.should_exit = True

    df = metrics.annotate(metrics.load())
    att = metrics.load_attempts(limit=None)
    for r in out["runs"]:
        g = df[df["turn_id"] == r["turn_id"]] if not df.empty else df
        r["recorded_model_calls"] = int(g["is_model_call"].sum()) if len(g) else 0
        r["recorded_attempts"] = int(g["provider_attempts"].fillna(0).sum()) if len(g) else 0
        r["recorded_blocked"] = int((g["status"] == "blocked").sum()) if len(g) else 0
        r["input_tokens"] = int(g["in_tok"].sum()) if len(g) else 0
        r["output_tokens"] = int(g["out_tok"].sum()) if len(g) else 0
        r["usage_complete"] = bool((g["usage_kind"] == "provider").all()) if len(g) else False
        costs = [cost.call_cost(m, i, o) for m, i, o in zip(g["model"], g["in_tok"], g["out_tok"])] if len(g) else []
        r["est_cost_usd"] = sum(c for c in costs if c is not None) if costs and all(c is not None for c in costs) else None
    out["totals"] = {m: {"runs": sum(r["model"] == m for r in out["runs"]), "passed": sum(r["passed"] for r in out["runs"] if r["model"] == m),
                         "provider_calls": sum(len(r["provider_calls"]) for r in out["runs"] if r["model"] == m),
                         "attempts": sum(r["recorded_attempts"] for r in out["runs"] if r["model"] == m),
                         "input_tokens": sum(r["input_tokens"] for r in out["runs"] if r["model"] == m),
                         "output_tokens": sum(r["output_tokens"] for r in out["runs"] if r["model"] == m),
                         "est_cost_usd": sum(r["est_cost_usd"] or 0 for r in out["runs"] if r["model"] == m)} for m in models}
    out["attempt_outcomes"] = att["outcome"].value_counts().to_dict() if not att.empty else {}
    if report_path:
        Path(report_path).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="really call the provider (bounded)")
    ap.add_argument("--report")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    if not a.live:
        print(f"{len(CASES)} cases defined; nothing called. Re-run with --live MODEL [MODEL...] (max {MAX_RUNS} runs).")
    else:
        rep = run_live(a.models, a.report)
        for r in rep["runs"]:
            print(("PASS " if r["passed"] else "FAIL ") + f"{r['model']:34s} {r['case']:13s} calls={len(r['provider_calls'])} {r['problems']}")
        print(json.dumps(rep["totals"], indent=2))
