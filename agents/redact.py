"""
Redaction applied to every prompt before it is sent to a hosted LLM provider
(Groq, OpenRouter, Gemini). Local Ollama calls are not redacted: the text
never leaves the machine. Turn it off with REDACT_HOSTED=0.

What is replaced, by pattern:
  IPv4 addresses (with optional :port)       172.16.96.116:53370 -> <ip>
  BGL hardware locations                     R23-M1-N8-I:J18-U01  -> <node>
  BGL I/O node host names                    bglio318             -> <host>
  hardware serials (6+ colon-separated bytes) FF:F2:9F:16:EB:36:.. -> <id>
  user names in home/project paths           /home/auselton/...   -> /home/<user>/...
                                             /p/gb1/stella/...    -> /p/gb1/<user>/...
  e-mail addresses                           a@b.org              -> <email>
  long hex strings and key-like tokens       0x0f3a9c1e..., sk-...-> <token>

This is pattern matching, not a guarantee: a user name or host that doesn't
fit these shapes goes through unchanged. Things the model needs to reason
about -- component, level, message wording, short register values, mount
points like /p/gb1 -- are kept.
"""
import re

_RULES = [
    (re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b"), "<ip>"),
    # R23-M1-N8-I:J18-U01, R05-M0-N2, R02-M1-NF, R02-M1-L1-U18-A (rack-midplane-...)
    (re.compile(r"\bR\d{2}-M\d(?:-[A-Z0-9]+(?::[A-Z0-9]+)?)*(?:-[A-Z0-9]+)*\b"), "<node>"),
    (re.compile(r"\bbglio\d+\b"), "<host>"),
    # hardware serials such as ido chip ids: FF:F2:9F:16:EB:36:00:0D:...
    (re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5,}\b"), "<id>"),
    (re.compile(r"(/home/)[^/\s]+"), r"\1<user>"),
    (re.compile(r"(/p/gb\d+/)(?!<)[^/\s)]+"), r"\1<user>"),
    (re.compile(r"\b(?:sk|pk|gsk|key|tok)[-_][A-Za-z0-9_-]{12,}\b"), "<token>"),
    (re.compile(r"\b0x[0-9a-fA-F]{12,}\b|\b[0-9a-fA-F]{24,}\b"), "<token>"),
]


def redact(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text


def redact_messages(messages: list[dict]) -> list[dict]:
    return [{**m, "content": redact(m["content"])} for m in messages]
