import os
import sys
import tempfile
from pathlib import Path

# Tests always run offline (mock model) against a throwaway database.
os.environ["MIRSAD_MOCK"] = "1"
os.environ["MIRSAD_DB"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["LOOP_THRESHOLD"] = "3"
os.environ["ROUTER_THRESHOLD"] = "3"
os.environ["MIRSAD_ROUTING_MODE"] = "rules"   # tests never need Ollama; routing-mode tests inject a stub
# Authentication settings used by the tests (fake values; Supabase itself is always replaced by an in-memory fake)
os.environ["MIRSAD_INTERNAL_TOKEN"] = "test-internal-token-0123456789abcdef"
os.environ["MIRSAD_SESSION_SECRET"] = "test-session-secret-0123456789abcdef"
os.environ["SUPABASE_URL"] = "https://example-project.supabase.co"
os.environ["SUPABASE_PUBLISHABLE_KEY"] = "sb_publishable_TESTONLY_not_a_real_key"
os.environ["MIRSAD_PUBLIC_URL"] = "http://localhost:8501"
os.environ["MIRSAD_AUTH_REVALIDATE_S"] = "300"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
