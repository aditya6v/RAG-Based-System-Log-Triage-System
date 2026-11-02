"""BGL line splitting and the category file."""
import pytest

from bgl import categories
from bgl.lines import has_message, level_of, split_line

ALERT = ("KERNDTLB 1118538627 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-18.10.27.692687 "
         "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt")
NORMAL = ("- 1118710081 2005.06.13 R03-M1-N9-C:J04-U01 2005-06-13-17.48.01.577343 "
          "R03-M1-N9-C:J04-U01 RAS KERNEL FATAL instruction address: 0x00004ed8")


def test_label_is_split_off_and_never_in_text_or_content():
    sp = split_line(ALERT)
    assert sp["label"] == "KERNDTLB"
    assert "KERNDTLB" not in sp["text"] and "KERNDTLB" not in sp["content"]
    assert sp["content"] == "KERNEL FATAL data TLB error interrupt"


def test_normal_label_dash():
    sp = split_line(NORMAL)
    assert sp["label"] == "-"
    assert sp["text"].startswith("1118710081 ")


def test_pasted_line_without_label():
    sp = split_line(split_line(ALERT)["text"])
    assert sp["label"] is None
    assert sp["content"] == "KERNEL FATAL data TLB error interrupt"


def test_non_bgl_line_falls_back_to_whole_line():
    sp = split_line("something broke on node 7")
    assert sp["label"] is None
    assert sp["content"] == sp["text"] == "something broke on node 7"


def test_level_and_empty_message():
    assert level_of("KERNEL FATAL data TLB error interrupt") == "FATAL"
    assert has_message("KERNEL FATAL data TLB error interrupt")
    assert not has_message("KERNEL FATAL")


def test_every_code_maps_to_one_category():
    codes = [code for c in categories.all_categories() for code in c["codes"]]
    assert len(codes) == len(set(codes)) == 42  # 41 alert codes + '-'
    assert categories.category_for_code("-") == "normal"
    assert categories.category_for_code("KERNPROG") == "normal"  # deliberate, see categories.json
    assert categories.category_for_code("KERNDTLB") == "memory_cpu_fault"
    assert categories.category_for_code("NOPE") is None


def test_every_category_has_definition_and_valid_severity():
    for c in categories.all_categories():
        assert c["definition"] and c["name"]
        assert c["severity"] in {"none", "low", "medium", "high"}
    assert categories.severity("normal") == "none"
    assert categories.severity("not-a-category") == "medium"


@pytest.mark.parametrize("reply,expected", [
    ("memory_cpu_fault", "memory_cpu_fault"),
    ("  `normal`.", "normal"),
    ("Category: kernel_crash", "kernel_crash"),
    ("memory / CPU fault", "memory_cpu_fault"),
    ("I think this is a filesystem mount failure", "filesystem_mount_failure"),
    ("no idea", "unknown"),
])
def test_llm_answer_parsing(reply, expected):
    assert categories.parse_llm_answer(reply) == expected


def test_majority():
    assert categories.majority({"normal": 3, "kernel_crash": 5}) == "kernel_crash"
    assert categories.majority({}) is None
    assert categories.majority(None) is None


TB_ALERT = ("EXT_FS 1131810726 2005.11.12 an312 Nov 12 07:52:06 an312/an312 kernel: EXT3-fs error "
            "(device sda5) in start_transaction: Journal has aborted")


def test_thunderbird_lines(monkeypatch):
    import config
    monkeypatch.setattr(config, "DATASET", "thunderbird")
    sp = split_line(TB_ALERT)
    assert sp["label"] == "EXT_FS"                      # underscore codes are labels too
    assert "EXT_FS" not in sp["text"] and "EXT_FS" not in sp["content"]
    assert sp["content"] == "kernel: EXT3-fs error (device sda5) in start_transaction: Journal has aborted"
    assert level_of(sp["content"]) is None             # Thunderbird has no level field
    assert has_message("kernel: oops")


def test_underscore_label_is_never_left_in_bgl_text():
    sp = split_line("SOME_CODE 1118538627 2005.06.11 R30 2005-06-11-18.10.27.692687 R30 RAS KERNEL FATAL x y")
    assert sp["label"] == "SOME_CODE" and not sp["text"].startswith("SOME_CODE")
