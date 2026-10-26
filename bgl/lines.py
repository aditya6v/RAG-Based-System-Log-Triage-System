"""
BGL line format (Loghub BGL.log), space separated:

    label  unix_ts  date  node  timestamp  node  type  component  level  message...
    -      1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected

`label` is the ground truth ('-' = normal, otherwise an alert code). It must
never reach Drain3, the embedding model or the LLM, or the answer leaks.

split_line returns three views of a line:
  label    -- the label token, or None if the line had none (e.g. pasted by a user)
  text     -- the line without the label (what the LLM and the UI see)
  content  -- "component level message": what Drain3 templates and what gets
              embedded. Node ids and timestamps are dropped because they are
              per-event noise, not "kind of failure".
"""
import re

_LABEL = re.compile(r"^(-|[A-Z]+)$")
HEADER_FIELDS = 6  # unix_ts date node timestamp node type


def split_line(line: str) -> dict:
    parts = line.strip().split(" ")
    label = None
    if len(parts) > 1 and _LABEL.match(parts[0]) and parts[1].isdigit():
        label, parts = parts[0], parts[1:]
    text = " ".join(parts)
    if len(parts) > HEADER_FIELDS and parts[0].isdigit():
        content = " ".join(parts[HEADER_FIELDS:])
    else:
        content = text  # not BGL-shaped: use the whole line
    return {"label": label, "text": text, "content": content}


def level_of(content: str) -> str | None:
    """Second token of content is the BGL level (INFO, FATAL, ...)."""
    parts = content.split(" ")
    return parts[1] if len(parts) > 1 else None


def has_message(content: str) -> bool:
    """34,470 BGL lines stop after the level with no message at all."""
    return len(content.split(" ")) > 2
