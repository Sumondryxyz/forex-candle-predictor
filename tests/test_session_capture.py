import json

import pytest

from predictor.session_capture import build_session, normalize_cookies, save_session


def test_normalize_cookie_editor_list():
    raw = [
        {"name": "laravel_session", "value": "abc"},
        {"name": "cf_clearance", "value": "xyz"},
    ]
    assert normalize_cookies(raw) == "laravel_session=abc; cf_clearance=xyz"


def test_normalize_accepts_json_string_and_wrapped_object():
    text = json.dumps([{"name": "a", "value": "1"}])
    assert normalize_cookies(text) == "a=1"
    assert normalize_cookies({"cookies": [{"name": "b", "value": "2"}]}) == "b=2"


def test_normalize_passes_through_a_raw_header():
    assert normalize_cookies("a=1;  b=2") == "a=1; b=2"


def test_normalize_dict_of_pairs():
    assert normalize_cookies({"a": "1", "b": "2"}) == "a=1; b=2"


def test_normalize_empty():
    assert normalize_cookies(None) == ""
    assert normalize_cookies("") == ""


def test_build_session_ok():
    entry = build_session(
        token="t" * 32,
        cookies=[{"name": "cf_clearance", "value": "x"}],
        user_agent="UA",
        email="me@example.com",
    )
    assert entry["token"] == "t" * 32
    assert entry["cookies"] == "cf_clearance=x"
    assert entry["user_agent"] == "UA"
    assert entry["email"] == "me@example.com"


def test_build_session_rejects_short_token():
    with pytest.raises(ValueError, match="token looks invalid"):
        build_session(token="short", cookies="a=1")


def test_build_session_rejects_missing_cookies():
    with pytest.raises(ValueError, match="no cookies"):
        build_session(token="t" * 32, cookies="")


def test_save_session_flat(tmp_path):
    path = tmp_path / "session.json"
    save_session(path, {"token": "t" * 32, "cookies": "a=1", "email": "x@y"})
    data = json.loads(path.read_text())
    assert data["token"] == "t" * 32
    assert "email" not in data  # reference-only key is dropped in the flat layout


def test_save_session_keyed_by_email_and_merges(tmp_path):
    path = tmp_path / "session.json"
    save_session(path, {"token": "a" * 32, "cookies": "a=1"}, email="a@x.com")
    save_session(path, {"token": "b" * 32, "cookies": "b=1"}, email="b@x.com")
    data = json.loads(path.read_text())
    assert set(data) == {"a@x.com", "b@x.com"}
    assert data["b@x.com"]["token"] == "b" * 32


def test_save_session_overwrites_a_corrupt_file(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{not json")
    save_session(path, {"token": "t" * 32, "cookies": "a=1"})
    assert json.loads(path.read_text())["token"] == "t" * 32
