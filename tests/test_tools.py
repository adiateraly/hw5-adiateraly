"""Each tool is tested directly, with no agent and no model."""
import json

from helpdesk.tools import check_service_status, find_instructions, prepare_ticket


def test_find_instructions_wifi():
    # Would catch: wrong key in helpdesk.json, or Wi-Fi returning password steps.
    r = find_instructions("wifi")
    assert r["ok"] and r["instruction_id"] == "GUIDE-WIFI" and len(r["steps"]) == 4


def test_find_instructions_alias_and_unknown():
    assert find_instructions("Wi-Fi")["instruction_id"] == "GUIDE-WIFI"
    assert find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"
    assert find_instructions("printer")["ok"] is False


def test_status_differs_by_snapshot():
    # Would catch: always returning 'operational', ignoring the selected snapshot.
    assert check_service_status("wifi", "normal")["status"] == "operational"
    assert check_service_status("wifi", "wifi_outage")["status"] == "outage"
    assert check_service_status("wifi", "status_unavailable")["status"] == "unknown"


def test_password_uses_account_recovery_service():
    r = check_service_status("password", "recovery_outage")
    assert r["service"] == "account_recovery" and r["status"] == "outage"


def test_status_bad_input():
    assert check_service_status("toaster")["ok"] is False
    assert check_service_status("wifi", "nope")["ok"] is False


def test_prepare_ticket_writes_local_draft(tmp_path):
    # Would catch: no file written, wrong fields, or invented steps.
    r = prepare_ticket("Campus-Demo will not connect", ["Enabled Wi-Fi"], "wifi", "operational",
                       user_id="demo-user-a", thread_id="t1", status_profile="normal",
                       tickets_dir=tmp_path)
    assert r["ok"]
    data = json.loads((tmp_path / f"{r['ticket_id']}.json").read_text(encoding="utf-8"))
    assert data["status"] == "draft" and data["steps_tried"] == ["Enabled Wi-Fi"]
    assert data["user_id"] == "demo-user-a" and data["reference_ids"] == ["GUIDE-WIFI"]


def test_prepare_ticket_refuses_missing_details(tmp_path):
    # Would catch: making a ticket from nothing (the S09 mistake).
    r = prepare_ticket("", [], "wifi", user_id="u", thread_id="t", status_profile="normal",
                       tickets_dir=tmp_path)
    assert r["ok"] is False and set(r["missing"]) == {"problem description", "steps already tried"}
    assert list(tmp_path.iterdir()) == []
