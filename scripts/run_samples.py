"""
Usage: python scripts/run_samples.py [out.json]

Sends the 5 front-page sample lines (real held-out BGL eval lines) through
the real POST /diagnose -- FastAPI app in-process, real Mongo, Qdrant and
Ollama -- and prints classification, route, per-agent timings, and the root
cause and remediation text. Used to record the demo results in
docs/EVALUATION.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

SAMPLES = [
    ("normal", "1118710081 2005.06.13 R03-M1-N9-C:J04-U01 2005-06-13-17.48.01.577343 R03-M1-N9-C:J04-U01 RAS KERNEL FATAL instruction address: 0x00004ed8"),
    ("memory_cpu_fault", "1118538627 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-18.10.27.692687 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt"),
    ("filesystem_mount_failure", "1125552602 2005.08.31 R23-M1-N8-I:J18-U01 2005-08-31-22.30.02.453491 R23-M1-N8-I:J18-U01 RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1"),
    ("io_connection_lost", "1124721545 2005.08.22 R73-M1-N4-I:J18-U01 2005-08-22-07.39.05.715470 R73-M1-N4-I:J18-U01 RAS APP FATAL ciod: Error reading message prefix after LOGIN_MESSAGE on CioStream socket to 172.16.96.116:53370: Link has been severed"),
    ("power_control_failure", "1120673475 2005.07.06 NULL 2005-07-06-11.11.15.368928 NULL RAS KERNEL FATAL Power deactivated: R05-M0-N2"),
]

out = []
with TestClient(app) as client:  # context manager runs the startup hook (builds the graph once)
    for expected, line in SAMPLES:
        r = client.post("/diagnose", json={"raw_log": line})
        d = r.json()
        out.append({"expected": expected, "status": r.status_code, **d})
        ok = "OK " if d.get("classification") == expected else "MISS"
        print(f"--- [{ok}] expected={expected} got={d.get('classification')} by={d.get('triage_method')} "
              f"route={d.get('route')} total={d.get('total_ms')}ms timings={d.get('timings_ms')}")
        ev = d.get("evidence") or {}
        print(f"    evidence: primary={(ev.get('primary') or {}).get('label')} "
              f"supporting={[s.get('label') for s in ev.get('supporting', [])]}")
        print(f"    root_cause:  {d.get('root_cause', d.get('error'))}")
        print(f"    remediation: {d.get('remediation', '')}")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
