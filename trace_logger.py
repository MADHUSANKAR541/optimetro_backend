import json
from pathlib import Path

TRACE_FILE = Path(__file__).parent / "sample_data" / "explainability_trace.json"

def log_trace(trace: dict):
    try:
        existing = json.loads(TRACE_FILE.read_text())
    except FileNotFoundError:
        existing = []

    existing.append(trace)
    TRACE_FILE.write_text(json.dumps(existing, indent=2))
