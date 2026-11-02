"""Redaction of text sent to hosted providers, on real BGL line shapes."""
import pytest

from agents.redact import redact, redact_messages


@pytest.mark.parametrize("raw,gone,kept", [
    ("1124721545 2005.08.22 R73-M1-N4-I:J18-U01 2005-08-22-07.39.05.715470 R73-M1-N4-I:J18-U01 RAS APP FATAL "
     "ciod: Error reading message prefix on CioStream socket to 172.16.96.116:53370, Link has been severed",
     ["R73-M1-N4-I:J18-U01", "172.16.96.116", "53370"], ["APP FATAL", "Link has been severed", "<node>", "<ip>"]),
    ("RAS KERNEL FATAL Lustre mount FAILED : bglio318 : point /p/gb1", ["bglio318"], ["/p/gb1", "<host>"]),
    ("ciod: LOGIN chdir(/p/gb2/draeger/benchmark/dat32k_060205) failed", ["draeger"], ["/p/gb2/<user>/benchmark"]),
    ("Error creating node map from file /home/auselton/bgl/64mps.sequential.mapfile", ["auselton"], ["/home/<user>/bgl"]),
    ("MidplaneSwitchController::receiveTrain() iap failed: R63-M0-L1-U18-A, status=ffbffbff",
     ["R63-M0-L1-U18-A"], ["iap failed", "status=ffbffbff"]),
    ("Power deactivated: R05-M0-N2", ["R05-M0-N2"], ["Power deactivated: <node>"]),
    ("Ido chip status changed: FF:F2:9F:16:EB:36:00:0D:60:E9:14:C9 ip=10.0.0.1", ["FF:F2:9F", "10.0.0.1"], ["<id>", "<ip>"]),
    ("contact ops@example.org token gsk_abcdefghijklmnop1234", ["ops@example.org", "gsk_abcdefghijklmnop1234"], ["<email>", "<token>"]),
])
def test_identifiers_are_masked_and_meaning_kept(raw, gone, kept):
    out = redact(raw)
    for g in gone:
        assert g not in out
    for k in kept:
        assert k in out


def test_short_register_values_are_kept():
    # register dumps are evidence, not identifiers
    assert redact("instruction address: 0x00004ed8") == "instruction address: 0x00004ed8"


def test_messages_keep_roles():
    msgs = [{"role": "system", "content": "node R05-M0-N2"}, {"role": "user", "content": "why?"}]
    assert redact_messages(msgs) == [{"role": "system", "content": "node <node>"}, {"role": "user", "content": "why?"}]
