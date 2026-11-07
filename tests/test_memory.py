from helpdesk.memory import UserMemory


def test_preference_survives_reopening(tmp_path):
    # Would catch: memory kept only in RAM instead of on disk.
    UserMemory(tmp_path / "m.sqlite").set("demo-user-a", "reply_language", "Kazakh")
    assert UserMemory(tmp_path / "m.sqlite").get_all("demo-user-a") == {"reply_language": "Kazakh"}


def test_users_are_separate(tmp_path):
    # Would catch: a query without WHERE user_id (user B seeing user A's data).
    m = UserMemory(tmp_path / "m.sqlite")
    m.set("demo-user-a", "reply_language", "Kazakh")
    assert m.get_all("demo-user-b") == {}


def test_update_overwrites(tmp_path):
    m = UserMemory(tmp_path / "m.sqlite")
    m.set("u", "reply_language", "Kazakh")
    m.set("u", "reply_language", "English")
    assert m.get_all("u") == {"reply_language": "English"}
