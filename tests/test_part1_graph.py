import pytest

pytest.importorskip("langgraph")
from helpdesk.part1 import run_with_trace


def test_original_route_wifi_password():
    r = run_with_trace("My Wi-Fi password does not work.")
    assert r["nodes_visited"] == ["read_request", "wifi_help"]


def test_changed_route_clarifies():
    r = run_with_trace("My Wi-Fi password does not work.", clarify=True)
    assert r["nodes_visited"] == ["read_request", "clarify_problem"]
    assert r["final_state"]["category"] == "both"


def test_unsupported_goes_to_other_help():
    r = run_with_trace("My laptop screen is broken.")
    assert r["nodes_visited"] == ["read_request", "other_help"]
