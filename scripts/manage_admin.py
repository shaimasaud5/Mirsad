"""
Trusted, server-side way to manage administrators. Run it on the machine that hosts Mirsad's database.

    python -m scripts.manage_admin list
    python -m scripts.manage_admin grant you@example.com
    python -m scripts.manage_admin revoke you@example.com

Why this is safe: it edits the local SQLite file directly, so it needs filesystem access to the server. Nothing a
browser sends (signup metadata, cookies, localStorage, API requests) can reach it, and it needs no Supabase
service-role key. The person must have signed up, confirmed their email and signed in once (that creates their row).
Roles take effect on their next request; no restart is needed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from auth import users  # noqa: E402
from proxy import db  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manage Mirsad administrators (local database only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show users and roles")
    for name in ("grant", "revoke"):
        p = sub.add_parser(name, help=f"{name} the admin role")
        p.add_argument("email")
    args = ap.parse_args(argv)

    conn = db.connect()
    print(f"database: {config.DB_PATH}")
    if args.cmd == "list":
        rows = users.list_all(conn)
        if not rows:
            print("(no users yet: sign up and sign in once in the web UI first)")
        for u in rows:
            print(f"{u['role']:<6} {'DISABLED ' if u['disabled'] else ''}{u['email']}  requests={u['requests']}")
        return 0

    user = users.find_by_email(conn, args.email)
    if user is None:
        print(f"No user with email {args.email!r}. They must sign up, confirm their email and sign in once first.")
        return 1
    role = "admin" if args.cmd == "grant" else "user"
    if args.cmd == "revoke":
        admins = [u for u in users.list_all(conn) if u["role"] == "admin" and not u["disabled"]]
        if len(admins) <= 1 and user["role"] == "admin":
            print("Refusing to remove the last administrator.")
            return 1
    users.set_role(conn, user["user_id"], role)
    print(f"{user['email']} is now: {role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
