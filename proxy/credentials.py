"""
Groq API-key pool: one primary key plus optional ordered backups, read from environment variables.

  GROQ_API_KEY            primary (existing setting, unchanged)  -> alias "primary"
  GROQ_API_KEY_BACKUP_1   first backup                            -> alias "backup_1"
  GROQ_API_KEY_BACKUP_2   second backup                           -> alias "backup_2"   (up to _9)

Only the ALIAS is ever logged, stored or displayed. Key values live in memory inside this object and
are excluded from repr(). A credential that Groq rejects with 401 is disabled for the life of the
process (restart the proxy after fixing the keys).
"""
import os
import re
import threading
from dataclasses import dataclass, field

MAX_BACKUPS = 9
_PLACEHOLDER = re.compile(r"replace|your[-_ ]|changeme|example|xxx|<.*>", re.IGNORECASE)

_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9_\-]{6,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(r"(?i)(authorization|api[-_ ]?key)(['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]{6,}"),
]


def redact(text, secrets=()) -> str:
    """Removes configured key values and anything shaped like a credential from a message."""
    out = str(text)
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "[REDACTED]")
    for pat in _SECRET_PATTERNS:
        out = pat.sub(lambda m: (m.group(1) + m.group(2) + "[REDACTED]") if m.re.groups >= 2 else "[REDACTED]", out)
    return out


@dataclass
class Credential:
    alias: str
    key: str = field(repr=False)


class CredentialPool:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self._creds: list[Credential] = []
        self._disabled: dict[str, str] = {}
        self._lock = threading.Lock()
        seen = set()

        def add(alias, value):
            value = (value or "").strip()
            if not value or _PLACEHOLDER.search(value) or value in seen:
                return
            seen.add(value)
            self._creds.append(Credential(alias, value))

        add("primary", env.get("GROQ_API_KEY"))
        for n in range(1, MAX_BACKUPS + 1):
            add(f"backup_{n}", env.get(f"GROQ_API_KEY_BACKUP_{n}"))

    @property
    def configured(self) -> int:
        return len(self._creds)

    def active(self) -> list[Credential]:
        with self._lock:
            return [c for c in self._creds if c.alias not in self._disabled]

    def disable(self, alias: str, reason: str) -> None:
        with self._lock:
            self._disabled.setdefault(alias, reason)

    def aliases(self) -> list[str]:
        return [c.alias for c in self._creds]

    def secrets(self) -> list[str]:
        return [c.key for c in self._creds]

    def status(self) -> list[dict]:
        """Safe to expose: aliases and state only, never key material."""
        with self._lock:
            return [{"alias": c.alias, "state": "disabled" if c.alias in self._disabled else "active",
                     "reason": self._disabled.get(c.alias)} for c in self._creds]
