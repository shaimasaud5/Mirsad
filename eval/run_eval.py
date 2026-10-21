"""
Offline evaluation. Uses NO Groq calls, so it is free and repeatable.

1. Routing Accuracy
   Hand-labelled simple/complex prompts scored by the same router the proxy uses.
   * dev_prompts.csv  (60): used while designing the rules
   * test_prompts.csv (40): held out, written before the final rules were scored,
                            never used for tuning -> this is the number to report

2. Loop Mitigation
   Synthetic agent histories are replayed through the same LoopGuard:
   * 10 true loops   (identical failing call repeated 5-10 times)  -> should be blocked
   * 6 negative controls (recovers, changes input, different tools,
     different sessions)                                           -> must NOT be blocked
   We report the loop mitigation rate, false-positive rate and model calls saved.

3. Cost Savings
   Read from the live request log (SQLite) if you have run the agent.

Run:  python -m eval.run_eval
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring import metrics            # noqa: E402
from proxy.loop_guard import LoopGuard     # noqa: E402
from proxy.router import classify          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


# ------------------------------------------------------------ 1. routing
def eval_routing(filename):
    rows = list(csv.DictReader(open(HERE / filename, encoding="utf-8")))
    results, cm = [], {(a, p): 0 for a in ("simple", "complex") for p in ("simple", "complex")}
    for r in rows:
        d = classify(r["prompt"])
        cm[(r["label"], d.route)] += 1
        results.append({**r, "predicted": d.route, "score": d.score,
                        "reason": d.reason_text, "correct": r["label"] == d.route})
    correct = sum(x["correct"] for x in results)
    acc = correct / len(results)

    def recall(c):
        tot = cm[(c, "simple")] + cm[(c, "complex")]
        return cm[(c, c)] / tot if tot else 0

    def precision(c):
        tot = cm[("simple", c)] + cm[("complex", c)]
        return cm[(c, c)] / tot if tot else 0

    OUT.mkdir(exist_ok=True)
    with open(OUT / filename.replace("_prompts", "_results"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader(); w.writerows(results)
    return {"n": len(results), "correct": correct, "accuracy": acc, "confusion": cm,
            "precision": {c: precision(c) for c in ("simple", "complex")},
            "recall": {c: recall(c) for c in ("simple", "complex")},
            "errors": [x for x in results if not x["correct"]]}


# --------------------------------------------------------------- 2. loops
def _call(cid, name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def _result(cid, ok):
    return {"role": "tool", "tool_call_id": cid,
            "content": "42 rows" if ok else "ERROR: service timeout, retry"}


def replay(guard, session, attempts):
    """attempts: list of (tool, args, ok). The agent sends the full history to the
    proxy after every tool result, exactly like a real agent. Returns how many
    model calls were allowed before a block (or len(attempts) if never blocked)."""
    msgs = [{"role": "user", "content": "do the task"}]
    if guard.check(session, msgs).blocked:
        return 0, True
    for i, (tool, args, ok) in enumerate(attempts, start=1):
        cid = f"{session}_{i}"
        msgs += [_call(cid, tool, args), _result(cid, ok)]
        if guard.check(session, msgs).blocked:
            return i, True
    return len(attempts), False


def eval_loops():
    guard = LoopGuard()
    loops = []
    tools = ["fetch_report", "read_file", "query_db", "send_email", "http_get"]
    for i in range(10):
        k = 5 + (i % 6)                       # agent would retry 5..10 times
        tool = tools[i % len(tools)]
        attempts = [(tool, {"target": f"x{i}"}, False)] * k
        allowed, blocked = replay(guard, f"loop{i}", attempts)
        loops.append({"scenario": f"{tool} x{k}", "retries": k, "allowed": allowed,
                      "blocked": blocked, "calls_saved": k - allowed if blocked else 0})

    controls = {
        "fails twice then succeeds": [("read_file", {"n": "a"}, False)] * 2 + [("read_file", {"n": "a"}, True)],
        "changes input each retry": [("read_file", {"n": f"f{j}"}, False) for j in range(5)],
        "different tools fail once": [(t, {}, False) for t in tools],
        "succeeds then one failure": [("query_db", {}, True)] * 3 + [("query_db", {}, False)],
        "fail, succeed, fail, succeed": [("http_get", {}, j % 2 == 1) for j in range(6)],
    }
    ctrl = []
    for name, attempts in controls.items():
        _, blocked = replay(guard, "ctrl_" + name, attempts)
        ctrl.append({"scenario": name, "blocked": blocked})
    # same failure spread over two sessions: 2 + 2 failures must not combine into a block
    _, b1 = replay(guard, "ctrl_sessA", [("read_file", {"n": "z"}, False)] * 2)
    _, b2 = replay(guard, "ctrl_sessB", [("read_file", {"n": "z"}, False)] * 2)
    ctrl.append({"scenario": "2 failures in each of 2 sessions", "blocked": b1 or b2})

    stopped = sum(l["blocked"] for l in loops)
    total_retries = sum(l["retries"] for l in loops)
    saved = sum(l["calls_saved"] for l in loops)
    fp = sum(c["blocked"] for c in ctrl)
    return {"loops": loops, "controls": ctrl,
            "mitigation_rate": stopped / len(loops),
            "false_positive_rate": fp / len(ctrl),
            "calls_saved": saved, "calls_without_guard": total_retries,
            "calls_saved_pct": saved / total_retries}


# ---------------------------------------------------------------- report
def main():
    test = eval_routing("test_prompts.csv")
    dev = eval_routing("dev_prompts.csv")
    l = eval_loops()
    s = metrics.summarize(metrics.load())

    lines = ["# Mirsad - Evaluation results", "",
             "## 1. Routing accuracy (offline, hand-labelled prompts)", "",
             f"**Held-out test set: {test['accuracy']:.1%}** ({test['correct']}/{test['n']})  ",
             f"Dev set (used to design the rules): {dev['accuracy']:.1%} ({dev['correct']}/{dev['n']})", "",
             "- A complex prompt sent to the small model hurts answer quality; a simple prompt "
             "sent to the big model only wastes money. So complex recall matters most.", ""]
    for title, r in (("Held-out test set", test), ("Dev set", dev)):
        cm = r["confusion"]
        lines += [f"### {title}", "",
                  "| actual \\ predicted | simple | complex |", "|---|---|---|",
                  f"| simple | {cm[('simple','simple')]} | {cm[('simple','complex')]} |",
                  f"| complex | {cm[('complex','simple')]} | {cm[('complex','complex')]} |", "",
                  f"- Precision: simple {r['precision']['simple']:.0%}, complex {r['precision']['complex']:.0%}",
                  f"- Recall: simple {r['recall']['simple']:.0%}, complex {r['recall']['complex']:.0%}",
                  "", "Misclassified:", ""]
        lines += [f"- #{e['id']} labelled **{e['label']}**, routed **{e['predicted']}** "
                  f"(score {e['score']}: {e['reason']}): {e['prompt'][:110]}"
                  for e in r["errors"]] or ["- none"]
        lines.append("")
    lines += [
        "", "## 2. Loop mitigation (offline replay through the real LoopGuard)", "",
        f"**Loop mitigation rate: {l['mitigation_rate']:.0%}** "
        f"({sum(x['blocked'] for x in l['loops'])}/{len(l['loops'])} loops stopped)", "",
        f"- Model calls without guard: {l['calls_without_guard']}, calls prevented: "
        f"{l['calls_saved']} ({l['calls_saved_pct']:.0%})",
        f"- **False-positive rate: {l['false_positive_rate']:.0%}** on {len(l['controls'])} "
        "negative controls (healthy retries that must not be blocked)", "",
        "| control scenario | blocked? |", "|---|---|",
    ]
    lines += [f"| {c['scenario']} | {'YES (bad)' if c['blocked'] else 'no (good)'} |" for c in l["controls"]]
    lines += ["", "## 3. Cost savings (from the live request log)", ""]
    if s.get("total"):
        lines += [f"- Requests logged: {s['total']} (success {s['success']}, errors {s['errors']}, "
                  f"blocked {s['blocked']})",
                  f"- Actual cost ${s['actual_cost']:.6f} vs baseline ${s['baseline_cost']:.6f} "
                  f"-> **savings {s['savings_pct']:.1f}%**",
                  f"- Share routed to the small model: {s['simple_share']:.0%}",
                  f"- Estimated loop savings: ${s['est_loop_savings']:.6f}",
                  "- Costs are simulated with public OpenAI prices (GPT-4o-mini vs GPT-4o) "
                  "applied to real token counts."]
    else:
        lines += ["- No requests logged yet. Run the agent, then re-run this script."]

    OUT.mkdir(exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved results in {OUT}")


if __name__ == "__main__":
    main()
