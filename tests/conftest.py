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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
