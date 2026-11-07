from helpdesk.routing import classify, topics_in


def test_wifi_message():
    assert classify("I cannot connect to Wi-Fi.") == "wifi"


def test_wifi_without_hyphen_and_case():
    assert classify("my WIFI is slow") == "wifi"


def test_password_is_case_insensitive():
    assert classify("I forgot my PASSWORD.") == "password"


def test_other():
    assert classify("My laptop screen is broken.") == "other"


def test_both_topics_original_graph_picks_wifi():
    # Original rules: the wifi check comes first, so password is ignored.
    assert classify("My Wi-Fi password does not work.") == "wifi"


def test_both_topics_with_clarification():
    assert classify("My Wi-Fi password does not work.", clarify=True) == "both"
    assert topics_in("My Wi-Fi password does not work.") == {"wifi", "password"}
