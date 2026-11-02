"""
Log line formats. config.DATASET picks one ("bgl", the default, or
"thunderbird"); both are Loghub logs whose first token is the label.

BGL (Loghub BGL.log), space separated:

    label  unix_ts  date  node  timestamp  node  type  component  level  message...
    -      1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected

Thunderbird (Loghub Thunderbird.log):

    label  unix_ts  date  host  month  day  time  source  message...
    -      1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 crond(pam_unix)[2915]: session closed for user root

`label` is the ground truth ('-' = normal, otherwise an alert code). It must
never reach Drain3, the embedding model or the LLM, or the answer leaks.

split_line returns three views of a line:
  label    -- the label token, or None if the line had none (e.g. pasted by a user)
  text     -- the line without the label (what the LLM and the UI see)
  content  -- what Drain3 templates and what gets embedded. BGL: "component
              level message". Thunderbird: the message. Node/host ids and
              timestamps are dropped because they are per-event noise, not
              "kind of failure".
"""
import re

import config

_LABEL = re.compile(r"^(-|[A-Z][A-Z0-9_]*)$")   # '-', KERNDTLB, EXT_FS, PBS_CHK, ...
HEADER_FIELDS = {"bgl": 6,           # unix_ts date node timestamp node type
                 "thunderbird": 7}   # unix_ts date host month day time source


def split_line(line: str) -> dict:
    parts = line.strip().split(" ")
    label = None
    if len(parts) > 1 and _LABEL.match(parts[0]) and parts[1].isdigit():
        label, parts = parts[0], parts[1:]
    text = " ".join(parts)
    header = HEADER_FIELDS[config.DATASET]
    if len(parts) > header and parts[0].isdigit():
        content = " ".join(parts[header:])
    else:
        content = text  # not in the expected shape: use the whole line
    return {"label": label, "text": text, "content": content}


def level_of(content: str) -> str | None:
    """BGL only: the second token of content is the level (INFO, FATAL, ...).
    Thunderbird lines have no level field."""
    if config.DATASET != "bgl":
        return None
    parts = content.split(" ")
    return parts[1] if len(parts) > 1 else None


def has_message(content: str) -> bool:
    """34,470 BGL lines stop after the level with no message at all."""
    if config.DATASET != "bgl":
        return bool(content.strip())
    return len(content.split(" ")) > 2
