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

# ---- Thresholds -----------------------------------------------------------
LOOP_THRESHOLD = int(os.getenv("LOOP_THRESHOLD", "3"))
ROUTER_THRESHOLD = int(os.getenv("ROUTER_THRESHOLD", "3"))

# ---- Pricing (USD per 1M tokens) -----------------------------------------
# Groq's free tier costs us nothing, so we price every call as if the company
# were paying OpenAI. The small model is priced as GPT-4o-mini, the big model
# (and the "no router" baseline) as GPT-4o.
# VERIFY against https://openai.com/api/pricing before submitting.
PRICES = {
    "simple":  {"name": "gpt-4o-mini", "input": 0.15, "output": 0.60},
    "complex": {"name": "gpt-4o",      "input": 2.50, "output": 10.00},
}
BASELINE_TIER = "complex"  # "what if everything went to the expensive model"

# Assumed output size for a request we blocked (it never reached a model)
BLOCKED_OUTPUT_TOKENS_ESTIMATE = 150

# ---- Storage / URLs -------------------------------------------------------
DB_PATH = Path(os.getenv("MIRSAD_DB", ROOT / "data" / "mirsad.db"))
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8000/v1")
AGENT_FILES_DIR = ROOT / "data" / "files"

# ---- Dashboard alert rules -----------------------------------------------
ALERT_WINDOW_MIN = 5          # look at the last 5 minutes
ALERT_ERROR_RATE = 0.20       # red banner if > 20% of requests failed
