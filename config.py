"""
Central configuration. Every number the system uses lives here, so the team
can point at one file when the committee asks "where does this value come from?".
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---- Models ---------------------------------------------------------------
SMALL_MODEL = os.getenv("SMALL_MODEL", "groq/qwen/qwen3.8-27b")
BIG_MODEL = os.getenv("BIG_MODEL", "groq/openai/gpt-oss-120b")
MOCK_MODE = os.getenv("MIRSAD_MOCK", "0") == "1"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

# ---- Routing mode ---------------------------------------------------------
#   rules     : keyword rules decide (original behaviour)
#   shadow    : rules decide; the local LLM's recommendation is only recorded
#   local_llm : the local LLM's classification decides; rules are the fallback
ROUTING_MODE = os.getenv("MIRSAD_ROUTING_MODE", "rules").strip().lower()
if ROUTING_MODE not in ("rules", "shadow", "local_llm"):
    raise ValueError(f"MIRSAD_ROUTING_MODE must be rules, shadow or local_llm (got {ROUTING_MODE!r})")

# ---- Local classifier (Ollama). It classifies only; it never answers ------------
# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and each new connection can lose ~2 s
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
CLASSIFIER_TIMEOUT_S = float(os.getenv("CLASSIFIER_TIMEOUT_S", "20"))   # bounded wait per classification
CLASSIFIER_COOLDOWN_S = float(os.getenv("CLASSIFIER_COOLDOWN_S", "30")) # skip Ollama this long after an outage
CLASSIFIER_KEEP_ALIVE = os.getenv("CLASSIFIER_KEEP_ALIVE", "30m")       # keep the model loaded between turns
CLASSIFIER_NUM_CTX = int(os.getenv("CLASSIFIER_NUM_CTX", "8192"))
CLASSIFIER_MAX_CHARS = 6000        # latest user request is cut here (and flagged) - same as evaluated v1
CONTEXT_MAX_MESSAGES = 4           # earlier user/assistant messages given to the classifier as context
CONTEXT_CHARS_PER_MESSAGE = 400    # each context message is cut here (and flagged)
TURN_CACHE_MAX = 2000              # classifications remembered (one per user turn)

# ---- Provider calls / API-key failover (keys themselves: GROQ_API_KEY, GROQ_API_KEY_BACKUP_1..9) ----
PROVIDER_MAX_ATTEMPTS = int(os.getenv("PROVIDER_MAX_ATTEMPTS", "3"))          # total provider calls per request
PROVIDER_RETRY_AFTER_MAX_S = float(os.getenv("PROVIDER_RETRY_AFTER_MAX_S", "10"))  # longest Retry-After we will wait
PROVIDER_TIMEOUT_S = float(os.getenv("PROVIDER_TIMEOUT_S", "60"))

# ---- Authentication (Supabase Auth; the app data stays in SQLite) ------------------
def _env_clean(name: str, default: str = "") -> str:
    """Tolerates `NAME= 'value'` style lines (stray spaces/quotes) without ever echoing the value."""
    return os.getenv(name, default).strip().strip("'\"").strip()


SUPABASE_URL = _env_clean("SUPABASE_URL").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = _env_clean("SUPABASE_PUBLISHABLE_KEY")
PUBLIC_URL = _env_clean("MIRSAD_PUBLIC_URL", "http://localhost:8501").rstrip("/")   # where users open the web UI
# Shared secret between the web server and the proxy (the browser never sees it). Without it the proxy refuses
# every /v1 and /admin call: there is no anonymous mode.
INTERNAL_TOKEN = _env_clean("MIRSAD_INTERNAL_TOKEN")
# Encrypts the Supabase refresh/access tokens that are stored server-side. Not a Supabase key.
SESSION_SECRET = _env_clean("MIRSAD_SESSION_SECRET")
SESSION_IDLE_HOURS = float(os.getenv("MIRSAD_SESSION_IDLE_HOURS", "12"))
SESSION_MAX_DAYS = float(os.getenv("MIRSAD_SESSION_MAX_DAYS", "7"))
AUTH_REVALIDATE_S = float(os.getenv("MIRSAD_AUTH_REVALIDATE_S", "300"))   # re-check the user with Supabase this often
CHAT_RATE_PER_MIN = int(os.getenv("MIRSAD_CHAT_RATE_PER_MIN", "20"))      # per user (each chat can spend model quota)

# ---- Thresholds -----------------------------------------------------------
LOOP_THRESHOLD = int(os.getenv("LOOP_THRESHOLD", "3"))
ROUTER_THRESHOLD = int(os.getenv("ROUTER_THRESHOLD", "3"))

# ---- Pricing: ESTIMATED inference cost (not an invoice) -------------------------
# Public Groq on-demand list prices for the answering models, USD per 1M tokens, separate input/output rates.
# Groq's free tier is not billed, so these numbers describe what the recorded usage WOULD cost at list price.
# Keys are the exact configured model ids (SMALL_MODEL / BIG_MODEL). A model missing here is "price unavailable":
# it is excluded from cost totals, never priced as zero. Bump PRICING_VERSION whenever a rate changes, so
# records priced under an older table stay identifiable.
PRICING_VERSION = "groq-list-2026-10-22"
PRICING_CURRENCY = "USD"
PRICING_UNIT = "per 1M tokens"
PRICING_RETRIEVED = "2026-10-22"
PRICING_SOURCE_URL = "https://console.groq.com/docs/models"
MODEL_PRICES = {
    "groq/openai/gpt-oss-120b": {"input": 0.15, "output": 0.60,
                                 "source": "https://console.groq.com/docs/model/openai/gpt-oss-120b"},
    "groq/qwen/qwen3.8-27b": {"input": 0.80, "output": 4.00,
                              "source": "https://console.groq.com/docs/model/qwen/qwen3.8-27b"},
}
# The comparison baseline is hypothetical: the SAME recorded tokens priced at BIG_MODEL's rates.

# ---- Storage / URLs -------------------------------------------------------
DB_PATH = Path(os.getenv("MIRSAD_DB", ROOT / "data" / "mirsad.db"))
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8000/v1")
AGENT_FILES_DIR = ROOT / "data" / "files"

# ---- Dashboard alert rules -----------------------------------------------
ALERT_WINDOW_MIN = 5          # look at the last 5 minutes
ALERT_ERROR_RATE = 0.20       # red banner if > 20% of requests failed
