"""
The one HTTP entry point. Run with: python app.py
Test with: curl -X POST http://localhost:5000/diagnose \\
             -H "Content-Type: application/json" \\
             -d '{"raw_log": "nova-compute.log.1 ... Failed to allocate network ..."}'

Deliberately thin: this file does not know anything about Drain3, Mongo,
Qdrant, or Gemini -- it just takes a raw_log string, hands it to the
compiled graph, and returns whatever state comes back. If you need to
debug pipeline logic, do it in agents/graph.py directly (python -m
agents.graph) rather than through HTTP -- much faster iteration loop.

Breaks-and-fixes:
  - 500 error with a RuntimeError about a missing env var -> config.py's
    require_all() caught a missing/placeholder key in .env.
  - Hangs on the first request -> that's almost always Mongo connection
    (Atlas IP allow-list), not Flask or LangGraph. Check that first.
  - Works once then fails on the second request in the same process ->
    check you're not accidentally creating a new Mongo/Qdrant client per
    request somewhere; db/mongo_client.py and db/qdrant_client.py are
    written to cache a single client, but a copy-pasted connection
    somewhere else is an easy way to reintroduce this.
"""
from flask import Flask, request, jsonify, send_from_directory

import config
from agents.graph import build_graph

app = Flask(__name__)
_graph = None  # built lazily so `flask --help` etc. don't require a configured .env


def get_graph():
    global _graph
    if _graph is None:
        config.require_all()
        _graph = build_graph()
    return _graph

@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/diagnose")
def diagnose():
    body = request.get_json(silent=True) or {}
    raw_log = body.get("raw_log")
    if not raw_log:
        return jsonify({"error": "expected JSON body with a 'raw_log' string field"}), 400

    result = get_graph().invoke({"raw_log": raw_log, "notes": []})
    return jsonify(result)


@app.errorhandler(Exception)
def handle_error(e):
    # Always answer in JSON so the front page can show the message instead of
    # failing to parse Flask's HTML error page. Most failures here are a
    # database or LLM being unreachable.
    app.logger.exception("request failed")
    status = getattr(e, "code", 500)
    if not isinstance(status, int):
        status = 500
    return jsonify({"error": f"{type(e).__name__}: {e}"}), status


if __name__ == "__main__":
    import os
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
