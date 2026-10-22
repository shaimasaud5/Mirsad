"""
Tools the agent can use. They run locally on your machine (not through the proxy).

Rule: a tool NEVER raises. On failure it returns a string starting with "ERROR".
That string travels back to the model inside the conversation, which is how the
proxy's loop guard can see repeated failures.
"""
import ast
import operator

from langchain.tools import tool

import config

FILES = config.AGENT_FILES_DIR
READ_LIMIT = 4000


@tool
def list_files() -> str:
    """List the files the assistant is allowed to read."""
    names = sorted(p.name for p in FILES.glob("*") if p.is_file())
    return "\n".join(names) if names else "ERROR: no files available"


@tool
def read_file(name: str) -> str:
    """Read a text/CSV file by name (e.g. 'sales.csv') and return its content."""
    path = (FILES / name).resolve()
    if FILES.resolve() not in path.parents:           # block ../../ tricks
        return f"ERROR: access to '{name}' is not allowed"
    if not path.is_file():
        return f"ERROR: file '{name}' not found. Use list_files to see available files."
    return path.read_text(encoding="utf-8")[:READ_LIMIT]    # cap size to protect token limits


def verify_file_read(name, result, status=None) -> dict | None:
    """Trusted provenance for the "File used" label. `result` is the content a read_file TOOL MESSAGE returned in this run.
    Returns {"name": basename, "partial": bool} only when that content is exactly what the file in the allowed folder
    yields; anything else (error result, unknown file, changed file, model-written text) returns None. Never a path."""
    if status == "error" or not isinstance(name, str) or not isinstance(result, str) or result.startswith("ERROR"):
        return None
    try:
        path = (FILES / name).resolve()
        if FILES.resolve() not in path.parents or not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    if result != text[:READ_LIMIT]:
        return None
    return {"name": path.name, "partial": len(text) > READ_LIMIT}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '1200 + 850 * 3'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"ERROR: could not calculate '{expression}': {exc}"


@tool
def fetch_report(report_name: str) -> str:
    """Fetch a live business report (e.g. 'q3_revenue') from the company reporting service."""
    # FAULT INJECTION: this simulates a broken external service that always times out.
    # It is how we reproduce a real agent failure loop on demand (chaos testing).
    return ("ERROR: reporting service timeout (HTTP 503) while fetching "
            f"'{report_name}'. This is usually temporary; retry the same request.")


ALL_TOOLS = [list_files, read_file, calculator, fetch_report]
