"""The 'unsupported specifics' check on generated diagnosis text."""
from eval.groundedness import check_one, specifics

DIAG = {
    "text": "1125552602 2005.08.31 R23-M1-N8-I:J18-U01 RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1",
    "definition": "A node could not mount or use its filesystem: Lustre mount FAILED.",
    "evidence": ["KERNEL FATAL Lustre mount FAILED : <*> : point /p/gb1"],
}


def verdicts(root_cause, remediation=""):
    return [s["verdict"] for s in check_one({**DIAG, "root_cause": root_cause, "remediation": remediation})]


def test_invented_paths_and_config_files_are_flagged():
    out = check_one({**DIAG, "root_cause": "The node could not mount /p/gb1.",
                     "remediation": "Check /etc/lustre/conf/lustre.conf and restart the mtio.conf service."})
    assert out[0]["verdict"] == "grounded"                       # /p/gb1 is in the log
    assert out[1]["verdict"] == "unsupported"
    assert set(out[1]["unsupported"]) >= {"/etc/lustre/conf/lustre.conf", "mtio.conf"}


def test_general_advice_is_generic_not_unsupported():
    assert verdicts("The filesystem could not be mounted.", "Check the network connection to the storage servers.") == \
        ["generic", "generic"]


def test_host_from_the_log_is_supported_even_with_other_number():
    # bglio318 is in the log, so naming it is grounded
    assert verdicts("The I/O node bglio318 failed to mount.") == ["grounded"]


def test_invented_numbers_and_identifiers():
    out = check_one({**DIAG, "root_cause": "Port 9443 on the MetadataServer timed out.", "remediation": ""})
    assert out[0]["verdict"] == "unsupported" and {"9443", "MetadataServer"} <= set(out[0]["unsupported"])


def test_common_acronyms_are_not_flagged():
    assert specifics("The RAS kernel reported a FATAL CPU error.") == set()
