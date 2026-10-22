"""Answer feedback, admin feedback monitoring, verified file labels and the removal of the user-facing usage views. Offline."""
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import config
from agent import agent as agent_mod
from agent import tools
from auth import chatstore
from monitoring import feedback as fb_stats
from proxy import db
from test_auth import PW, browser, chat, env, make_admin, post, sign_in  # noqa: F401  (env is a pytest fixture)
from ui import web_server as ws


def put(c, csrf, mid, **body):
    return c.put(f"/api/messages/{mid}/feedback", headers={"X-CSRF-Token": csrf}, json=body)


def drop(c, csrf, mid):
    return c.delete(f"/api/messages/{mid}/feedback", headers={"X-CSRF-Token": csrf})


def answer(c, csrf, text="hello", conv=None, mode="live", extra_calls=0):
    """Sends one chat message; returns the assistant message id. `mode` rewrites the recorded execution mode and
    `extra_calls` adds more model calls to the same turn (a tool loop)."""
    r = chat(c, csrf, text, conv).json()
    mid = r["message_id"]
    turn = ws.conn.execute("SELECT turn_id FROM messages WHERE id=?", (mid,)).fetchone()[0]
    ws.conn.execute("UPDATE requests SET execution_mode=? WHERE turn_id=?", (mode, turn))
    ws.conn.commit()
    base = ws.conn.execute("SELECT * FROM requests WHERE turn_id=? LIMIT 1", (turn,)).fetchone()
    cols = [d[0] for d in ws.conn.execute("SELECT * FROM requests LIMIT 0").description]
    for i in range(extra_calls):
        row = dict(zip(cols, base)); row.pop("id"); row["request_id"] = f"req_extra{mid}_{i}"; row["ts"] += i + 1
        db.insert(ws.conn, row)
    return mid


def count(table="message_feedback", where="1=1", args=()):
    return ws.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args).fetchone()[0]


@pytest.fixture()
def two(env):
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    b, csrf_b, uid_b = sign_in(env, "b@example.com")
    adm, csrf_adm, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    return SimpleNamespace(a=a, ca=csrf_a, ua=uid_a, b=b, cb=csrf_b, ub=uid_b, adm=adm, cadm=csrf_adm)


# ===================================================================== creating, updating, clearing, restoring
def test_rating_lifecycle_never_inflates_counts_and_is_restored_on_reopen(two):
    c, cs = two.a, two.ca
    r = chat(c, cs, "What is 2+2?").json()
    mid, conv = r["message_id"], r["conversation_id"]
    assert r["feedback"] is None
    assert put(c, cs, mid, rating="helpful").json() == {"feedback": {"rating": "helpful", "reason": None, "comment": None}}
    assert put(c, cs, mid, rating="not_helpful", reason="incomplete", comment="  missing the units  ").status_code == 200
    assert put(c, cs, mid, rating="not_helpful", reason="incorrect", comment="wrong total").status_code == 200
    assert count() == 1                                                                   # one current rating, however often it changes
    msgs = c.get(f"/api/conversations/{conv}").json()["messages"]
    assert msgs[1]["feedback"] == {"rating": "not_helpful", "reason": "incorrect", "comment": "wrong total"}
    assert msgs[0].get("feedback") is None and msgs[1]["id"] == mid                       # user messages carry none
    assert put(c, cs, mid, rating="not_helpful").status_code == 200                       # submission without a reason is allowed
    assert c.get(f"/api/conversations/{conv}").json()["messages"][1]["feedback"] == {"rating": "not_helpful", "reason": None, "comment": None}
    assert drop(c, cs, mid).json() == {"feedback": None} and count() == 0
    assert drop(c, cs, mid).status_code == 200                                            # clearing twice is harmless
    assert c.get(f"/api/conversations/{conv}").json()["messages"][1]["feedback"] is None


def test_feedback_input_is_validated_on_the_server(two):
    c, cs = two.a, two.ca
    mid = chat(c, cs, "q").json()["message_id"]
    bad = [dict(rating="great"), dict(rating="helpful", reason="incorrect"), dict(rating="not_helpful", reason="rude"),
           dict(rating="not_helpful", comment="x" * (config_max() + 1)), dict(rating="helpful", user_id="someone"),
           dict(rating="helpful", message_id=5), dict(reason="other"), {}]
    for body in bad:
        assert put(c, cs, mid, **body).status_code == 422, body
    assert count() == 0
    assert put(c, cs, mid, rating="not_helpful", reason="other", comment="y" * config_max()).status_code == 200
    ws.conn.execute("DELETE FROM message_feedback"); ws.conn.commit()
    assert put(c, cs, mid, rating="helpful", comment="ok\x00\x07 fine").json()["feedback"]["comment"] == "ok fine"   # control characters dropped
    assert c.put("/api/messages/abc/feedback", headers={"X-CSRF-Token": cs}, json={"rating": "helpful"}).status_code == 422


def config_max():
    from auth.feedback import MAX_COMMENT
    return MAX_COMMENT


def test_only_completed_assistant_answers_can_be_rated(two, monkeypatch):
    c, cs = two.a, two.ca
    r = chat(c, cs, "q").json()
    user_msg = ws.conn.execute("SELECT id FROM messages WHERE role='user'").fetchone()[0]
    assert put(c, cs, user_msg, rating="helpful").status_code == 404                      # your own question is not an answer
    monkeypatch.setattr(ws, "run_agent", lambda *a, **k: {"answer": "boom", "steps": [], "status": "error", "session_id": "s", "turn_id": "t"})
    failed = chat(c, cs, "again", r["conversation_id"]).json()["message_id"]
    assert put(c, cs, failed, rating="helpful").status_code == 404                        # failed attempts are not rated
    assert put(c, cs, 999999, rating="helpful").status_code == 404 and count() == 0


def test_feedback_mutations_need_csrf_and_a_session(two):
    c, cs = two.a, two.ca
    mid = chat(c, cs, "q").json()["message_id"]
    assert c.put(f"/api/messages/{mid}/feedback", json={"rating": "helpful"}).status_code == 403
    assert c.put(f"/api/messages/{mid}/feedback", json={"rating": "helpful"}, headers={"X-CSRF-Token": "x"}).status_code == 403
    assert c.put(f"/api/messages/{mid}/feedback", json={"rating": "helpful"},
                 headers={"X-CSRF-Token": cs, "Origin": "https://evil.example"}).status_code == 403
    assert browser().put(f"/api/messages/{mid}/feedback", json={"rating": "helpful"}).status_code == 401
    assert count() == 0


# ===================================================================== isolation between users
def test_users_cannot_rate_read_change_or_clear_each_others_answers(two):
    mid = chat(two.a, two.ca, "A's question").json()["message_id"]
    assert put(two.a, two.ca, mid, rating="helpful", comment="A's note").status_code == 200
    for attempt in (lambda: put(two.b, two.cb, mid, rating="not_helpful", reason="incorrect", comment="B was here"),
                    lambda: drop(two.b, two.cb, mid)):
        r = attempt()
        assert r.status_code == 404 and r.json() == put(two.b, two.cb, 424242, rating="helpful").json()   # no existence leak
    row = ws.conn.execute("SELECT user_id, rating, comment FROM message_feedback").fetchall()
    assert row == [(two.ua, "helpful", "A's note")] and "B was here" not in str(row)
    assert count() == 1
    # B rates nothing of A's, and B's own answer is independent of A's rating
    own = chat(two.b, two.cb, "B's question").json()["message_id"]
    assert put(two.b, two.cb, own, rating="not_helpful", reason="other").status_code == 200
    assert count() == 2
    assert two.a.get("/api/conversations").json()["conversations"] != two.b.get("/api/conversations").json()["conversations"]
    assert "A's note" not in json.dumps(two.b.get("/api/conversations").json())


def test_database_itself_enforces_one_rating_and_valid_values(two):
    mid = chat(two.a, two.ca, "q").json()["message_id"]
    put(two.a, two.ca, mid, rating="helpful")
    ins = "INSERT INTO message_feedback (user_id, conversation_id, message_id, rating, reason) VALUES (?,?,?,?,?)"
    for args in ((two.ua, "c", mid, "helpful", None),            # duplicate for the same user and answer
                 (two.ub, "c", mid + 100, "great", None),         # unknown rating
                 (two.ub, "c", mid + 100, "not_helpful", "rude"), # unknown reason
                 (two.ub, "c", mid + 100, "helpful", "other")):   # a reason only belongs with Not helpful
        with pytest.raises(sqlite3.IntegrityError):
            ws.conn.execute(ins, args)
    ws.conn.rollback()
    assert count() == 1


def test_feedback_never_becomes_model_history(two):
    c, cs = two.a, two.ca
    r = chat(c, cs, "What is Flutter?").json()
    put(c, cs, r["message_id"], rating="not_helpful", reason="incorrect", comment="SECRET-COMMENT-TEXT")
    hist = chatstore.history_for_agent(ws.conn, r["conversation_id"], two.ua)
    assert [m["role"] for m in hist] == ["user", "assistant"] and "SECRET-COMMENT-TEXT" not in json.dumps(hist)
    chat(c, cs, "and Dart?", r["conversation_id"])
    assert "SECRET-COMMENT-TEXT" not in json.dumps(env_calls(c))


def env_calls(_c):
    return [dict(zip(("role", "content"), r)) for r in ws.conn.execute("SELECT role, content FROM messages")]


def test_deleting_a_conversation_removes_its_feedback_only(two):
    a1 = chat(two.a, two.ca, "one").json()
    a2 = chat(two.a, two.ca, "two").json()
    put(two.a, two.ca, a1["message_id"], rating="helpful"); put(two.a, two.ca, a2["message_id"], rating="helpful")
    assert two.a.delete(f"/api/conversations/{a1['conversation_id']}", headers={"X-CSRF-Token": two.ca}).status_code == 200
    assert count() == 1 and count(where="message_id=?", args=(a2["message_id"],)) == 1


# ===================================================================== admin monitoring
def seed(two):
    """5 live answers (2 helpful, 1 incorrect, 1 not-specified, 1 unrated), 1 mock rated helpful, 1 old live rated."""
    ids = [answer(two.a, two.ca, f"live {i}") for i in range(3)] + [answer(two.b, two.cb, f"live b{i}") for i in range(2)]
    put(two.a, two.ca, ids[0], rating="helpful")
    put(two.a, two.ca, ids[1], rating="not_helpful", reason="incorrect", comment="total is wrong")
    put(two.b, two.cb, ids[3], rating="helpful")
    put(two.b, two.cb, ids[4], rating="not_helpful")
    mock = answer(two.a, two.ca, "mock one", mode="mock")
    put(two.a, two.ca, mock, rating="helpful")
    old = answer(two.b, two.cb, "old one")
    put(two.b, two.cb, old, rating="not_helpful", reason="misunderstood")
    ws.conn.execute("UPDATE messages SET ts=ts-7200 WHERE id=?", (old,)); ws.conn.commit()
    return SimpleNamespace(ids=ids, mock=mock, old=old)


def test_summary_math_defaults_to_live_and_never_counts_unrated_as_helpful(two):
    s = seed(two)
    r = two.adm.get("/api/admin/feedback").json()                         # default kind = live
    sm = r["summary"]
    assert (sm["eligible"], sm["rated"], sm["helpful"], sm["not_helpful"], sm["unrated"]) == (6, 5, 2, 3, 1)
    assert sm["helpful_pct"] == 40.0 and round(sm["coverage_pct"], 2) == round(5 / 6 * 100, 2)
    assert r["reasons"] == {"incorrect": 1, "incomplete": 0, "misunderstood": 1, "other": 0, "not_specified": 1}
    assert r["by_execution"] == {"live": {"eligible": 6, "rated": 5}, "mock": {"eligible": 1, "rated": 1}}
    allr = two.adm.get("/api/admin/feedback?kind=all").json()["summary"]
    assert (allr["eligible"], allr["rated"], allr["helpful"]) == (7, 6, 3)
    mockr = two.adm.get("/api/admin/feedback?kind=mock").json()["summary"]
    assert (mockr["eligible"], mockr["rated"], mockr["helpful_pct"]) == (1, 1, 100.0)
    assert two.adm.get("/api/admin/feedback?kind=legacy").json()["summary"]["eligible"] == 0
    assert all(i["message_id"] != s.mock for i in r["items"])


def test_summary_and_list_share_the_same_filters(two):
    s = seed(two)
    recent = two.adm.get("/api/admin/feedback?minutes=60").json()         # the 2-hour-old answer drops out of BOTH
    assert recent["summary"]["rated"] == 4 and recent["items_total"] == 4
    assert s.old not in {i["message_id"] for i in recent["items"]}
    assert recent["reasons"]["misunderstood"] == 0
    neg = two.adm.get("/api/admin/feedback?rating=not_helpful").json()
    assert {i["rating"] for i in neg["items"]} == {"not_helpful"} and neg["summary"]["rated"] == 5   # summary unaffected by list filters
    only = two.adm.get("/api/admin/feedback?reason=not_specified").json()["items"]
    assert [i["reason"] for i in only] == [None]
    inc = two.adm.get("/api/admin/feedback?reason=incorrect").json()["items"]
    assert len(inc) == 1 and inc[0]["comment"] == "total is wrong" and inc[0]["question"] == "live 1"
    for bad in ("kind=weird", "rating=meh", "reason=rude"):
        assert two.adm.get(f"/api/admin/feedback?{bad}").status_code == 422


def test_changing_or_clearing_ratings_is_reflected_without_double_counting(two):
    mid = answer(two.a, two.ca, "q")
    put(two.a, two.ca, mid, rating="helpful")
    put(two.a, two.ca, mid, rating="not_helpful", reason="other")
    s = two.adm.get("/api/admin/feedback").json()
    assert (s["summary"]["rated"], s["summary"]["helpful"], s["summary"]["not_helpful"], s["reasons"]["other"]) == (1, 0, 1, 1)
    drop(two.a, two.ca, mid)
    s = two.adm.get("/api/admin/feedback").json()["summary"]
    assert (s["eligible"], s["rated"], s["unrated"], s["helpful_pct"]) == (1, 0, 1, None)


def test_one_rating_per_answer_even_when_the_answer_used_several_model_calls(two):
    mid = answer(two.a, two.ca, "analyse sales.csv", extra_calls=2)
    plain = answer(two.a, two.ca, "short one")
    put(two.a, two.ca, mid, rating="not_helpful", reason="incomplete")
    put(two.a, two.ca, mid, rating="not_helpful", reason="incomplete")
    r = two.adm.get("/api/admin/feedback").json()
    assert r["summary"]["rated"] == 1 and r["summary"]["eligible"] == 2 and r["items_total"] == 1
    item = r["items"][0]
    assert item["calls"] == 3 and item["message_id"] == mid and item["route"] in ("simple", "complex")
    assert sum(g["rated"] for g in r["by_route"]) == 1 and sum(g["eligible"] for g in r["by_route"]) == 2   # not tripled
    d = two.adm.get(f"/api/admin/feedback/{mid}").json()
    assert len(d["calls"]) == 3 and {c["route"] for c in d["calls"]} == {item["route"]}
    assert plain != mid


def test_breakdowns_report_sample_sizes_and_flag_small_samples(two):
    seed(two)
    r = two.adm.get("/api/admin/feedback").json()
    simple = next(g for g in r["by_route"] if g["key"] == "simple")
    assert simple["rated"] == 5 and simple["eligible"] == 6 and simple["low_sample"] is True and simple["helpful_pct"] == 40.0
    assert r["by_model"] and all("rated" in g and "eligible" in g for g in r["by_model"])
    assert r["definitions"]["low_sample_below"] == fb_stats.LOW_SAMPLE


def test_admin_detail_shows_question_answer_feedback_and_calls(two):
    mid = answer(two.a, two.ca, "How much revenue?")
    put(two.a, two.ca, mid, rating="not_helpful", reason="incorrect", comment="nope")
    d = two.adm.get(f"/api/admin/feedback/{mid}").json()
    assert d["question"] == "How much revenue?" and d["answer"] == "answer to: How much revenue?"
    assert d["feedback"]["comment"] == "nope" and d["user"] == "a@example.com" and d["calls"][0]["route_source"]
    assert two.adm.get("/api/admin/feedback/987654").status_code == 404
    assert two.a.get(f"/api/admin/feedback/{mid}").status_code == 403
    assert two.b.get(f"/api/admin/feedback/{mid}").status_code == 403


def test_ordinary_users_cannot_reach_any_admin_feedback_or_monitoring_endpoint(two):
    mid = answer(two.a, two.ca, "q")
    put(two.a, two.ca, mid, rating="helpful")
    for client in (two.a, two.b):
        for path in ("/api/admin/feedback", f"/api/admin/feedback/{mid}", "/api/admin/feedback?kind=all", "/api/metrics", "/api/admin/users"):
            r = client.get(path)
            assert r.status_code == 403 and r.json()["error"] == "forbidden", path
        assert client.post("/api/admin/reset", headers={"X-CSRF-Token": two.ca if client is two.a else two.cb}).status_code == 403
        assert client.get("/api/usage").status_code == 404
    assert count() == 1 and count("requests") >= 1


def test_admin_reset_clears_telemetry_but_keeps_user_feedback(two, monkeypatch):
    mid = answer(two.a, two.ca, "q")
    put(two.a, two.ca, mid, rating="helpful")
    monkeypatch.setattr(ws.httpx, "post", lambda url, **kw: ws_proxy_reset(kw["headers"]))
    assert post(two.adm, two.cadm, "/api/admin/reset").json()["status"] == "reset"
    assert count("requests") == 0 and count() == 1 and count("messages") == 2
    s = two.adm.get("/api/admin/feedback?kind=all").json()["summary"]
    assert s["rated"] == 1                                                  # kept, now without telemetry (shown as legacy/unknown)


def ws_proxy_reset(headers):
    from test_auth import proxy
    return proxy.post("/admin/reset", headers=headers)


# ===================================================================== technical details removed from the user experience
def test_page_has_no_usage_view_or_technical_summary_and_has_feedback_controls():
    html = (config.ROOT / "ui" / "web" / "index.html").read_text(encoding="utf-8")
    for gone in ("nav-usage", "/api/usage", "usageTitle", "function badges", "badges(m)", "stepsHtml(", "gpt4o:", "failover:", "classifier:"):
        assert gone not in html, gone
    for present in ("/feedback", "copyAnswer", 'data-act="${act}"', "aria-pressed", "fileUsed", "partialRead", "/api/admin/feedback",
                    "r_incorrect", "r_incomplete", "r_misunderstood", "r_other", "aria-live"):
        assert present in html, present
    ar = html[html.index("  ar: {"):]
    for key in ("copyAnswer", "rateHelpful", "rateNotHelpful", "r_incorrect", "fileUsed", "err_busy", "fbTitle", "fbNote"):   # Arabic labels exist too
        assert key + ":" in ar, key


# ===================================================================== verified file labels
@pytest.fixture()
def files_dir(tmp_path, monkeypatch):
    (tmp_path / "sales.csv").write_text("region,revenue\nRiyadh,100\n", encoding="utf-8")
    (tmp_path / "big.txt").write_text("x" * (tools.READ_LIMIT + 50), encoding="utf-8")
    monkeypatch.setattr(tools, "FILES", tmp_path)
    (tmp_path.parent / "outside.txt").write_text("secret", encoding="utf-8")
    return tmp_path


def test_verify_file_read_only_accepts_content_that_really_came_from_the_file(files_dir):
    ok = tools.read_file.invoke({"name": "sales.csv"})
    assert tools.verify_file_read("sales.csv", ok) == {"name": "sales.csv", "partial": False}
    big = tools.read_file.invoke({"name": "big.txt"})
    assert tools.verify_file_read("big.txt", big) == {"name": "big.txt", "partial": True}
    assert tools.verify_file_read("sales.csv", ok, "error") is None                        # the tool reported failure
    assert tools.verify_file_read("sales.csv", "ERROR: file 'sales.csv' not found.") is None
    assert tools.verify_file_read("sales.csv", "Total revenue is 100") is None             # model-style text, not the file
    assert tools.verify_file_read("sales.csv", ok + " ") is None
    assert tools.verify_file_read("nope.csv", ok) is None
    assert tools.verify_file_read("../outside.txt", "secret") is None                       # outside the allowed folder
    assert tools.verify_file_read(str(files_dir.parent / "outside.txt"), "secret") is None  # absolute path outside the folder
    inside = tools.verify_file_read(str(files_dir / "sales.csv"), ok)                       # absolute path INSIDE is a real read,
    assert inside == {"name": "sales.csv", "partial": False}                                # but only the basename is ever reported
    assert tools.verify_file_read(None, ok) is None and tools.verify_file_read("sales.csv", None) is None
    assert str(files_dir) not in json.dumps(tools.verify_file_read("sales.csv", ok))        # basename only


def fake_stream(monkeypatch, updates):
    monkeypatch.setattr(agent_mod, "build_agent", lambda *a, **k: SimpleNamespace(stream=lambda *a, **k: iter(updates)))


def turn(call_name, args, result, call_id="call_1", tool="read_file", status="success", answer="done"):
    return [{"model": {"messages": [AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": call_id}])]}},
            {"tools": {"messages": [ToolMessage(content=result, name=tool, tool_call_id=call_id, status=status)]}},
            {"model": {"messages": [AIMessage(content=answer)]}}]


def test_agent_reports_files_only_when_a_tool_result_proves_the_read(files_dir, monkeypatch):
    good = tools.read_file.invoke({"name": "sales.csv"})
    fake_stream(monkeypatch, turn("read_file", {"name": "sales.csv"}, good))
    r = agent_mod.run_agent("Read sales.csv", user_id="u1")
    assert r["files"] == [{"name": "sales.csv", "partial": False}] and r["status"] == "ok"

    fake_stream(monkeypatch, turn("read_file", {"name": "big.txt"}, tools.read_file.invoke({"name": "big.txt"})))
    assert agent_mod.run_agent("Read big.txt", user_id="u1")["files"] == [{"name": "big.txt", "partial": True}]

    fake_stream(monkeypatch, turn("read_file", {"name": "sales.csv"}, "ERROR: file 'sales.csv' not found."))
    assert agent_mod.run_agent("Read sales.csv", user_id="u1")["files"] == []              # failed read

    fake_stream(monkeypatch, turn("read_file", {"name": "sales.csv"}, good, status="error"))
    assert agent_mod.run_agent("Read sales.csv", user_id="u1")["files"] == []

    fake_stream(monkeypatch, [{"model": {"messages": [AIMessage(content="I read sales.csv and the total is 100.")]}}])
    assert agent_mod.run_agent("Read sales.csv", user_id="u1")["files"] == []              # claimed in text, never read

    fake_stream(monkeypatch, turn("list_files", {}, "sales.csv\nbig.txt", tool="list_files"))
    assert agent_mod.run_agent("What files exist?", user_id="u1")["files"] == []           # listing is not reading

    forged = turn("read_file", {"name": "sales.csv"}, good, call_id="call_1")
    forged[1]["tools"]["messages"][0].tool_call_id = "call_other"                          # result not tied to that request
    fake_stream(monkeypatch, forged)
    assert agent_mod.run_agent("Read sales.csv", user_id="u1")["files"] == []

    fake_stream(monkeypatch, turn("read_file", {"name": "../outside.txt"}, "secret"))
    assert agent_mod.run_agent("Read it", user_id="u1")["files"] == []


def test_file_label_is_stored_with_the_answer_and_restored(two, monkeypatch):
    def agent_with(files):
        return lambda task, session_id=None, history=None, *, user_id=None, conversation_id=None: {
            "answer": "Total is 100", "steps": [{"type": "tool_result", "tool": "read_file", "result": "/srv/secret/path"}],
            "files": files, "status": "ok", "session_id": session_id, "turn_id": "turn_f"}
    monkeypatch.setattr(ws, "run_agent", agent_with([{"name": "sales.csv", "partial": True}]))
    r = chat(two.a, two.ca, "Read sales.csv").json()
    assert r["files"] == [{"name": "sales.csv", "partial": True}] and "steps" not in r
    msgs = two.a.get(f"/api/conversations/{r['conversation_id']}").json()["messages"]
    assert msgs[1]["files"] == [{"name": "sales.csv", "partial": True}] and "/srv/secret" not in json.dumps(msgs)
    monkeypatch.setattr(ws, "run_agent", agent_with([]))
    r2 = chat(two.a, two.ca, "Read sales.csv please", r["conversation_id"]).json()
    assert r2["files"] == []                                                               # no proof, no label


# ===================================================================== migration
def test_feedback_migration_backs_up_and_preserves_existing_conversations(tmp_path):
    path = tmp_path / "pre.db"
    old = db.connect(path)
    old.execute("INSERT INTO requests (request_id, ts, status, user_id) VALUES ('r1', 1.0, 'success', 'u1')")
    old.execute("INSERT INTO conversations (id, user_id, title, session_id, created_at, updated_at) VALUES ('c1','u1','t','s',1,1)")
    old.execute("INSERT INTO messages (conversation_id, user_id, role, content, status, ts) VALUES ('c1','u1','assistant','hi','ok',1)")
    old.execute("DROP TABLE message_feedback")
    old.execute("ALTER TABLE messages DROP COLUMN files_json")                              # the shape before this release
    old.commit(); old.close()
    new = db.connect(path)
    assert new.execute("SELECT content, files_json FROM messages").fetchall() == [("hi", None)]
    assert new.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    assert new.execute("SELECT COUNT(*) FROM message_feedback").fetchone()[0] == 0
    backup = sqlite3.connect(tmp_path / "pre.db-backup-pre-feedback")
    assert "files_json" not in [r[1] for r in backup.execute("PRAGMA table_info(messages)")]   # untouched pre-migration copy
    assert backup.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    db.connect(path)                                                                         # idempotent
    assert chatstore.messages(new, "c1", "u1")[0]["files"] == []                             # old rows read fine
