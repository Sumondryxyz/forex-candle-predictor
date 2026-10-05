import json

import pytest

from predictor.quotex_session import load_session


def _write(tmp_path, data):
    path = tmp_path / "session.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_flat_session_round_trips(tmp_path):
    path = _write(tmp_path, {"cookies": "laravel_session=x", "token": "t" * 32, "user_agent": "UA"})
    session = load_session(path)
    assert session["token"] == "t" * 32
    assert session["cookies"] == "laravel_session=x"
    assert session["user_agent"] == "UA"


def test_email_keyed_session_picks_the_account(tmp_path):
    path = _write(
        tmp_path,
        {
            "a@example.com": {"cookies": "a", "token": "A" * 32, "user_agent": "UA-a"},
            "b@example.com": {"cookies": "b", "token": "B" * 32, "user_agent": "UA-b"},
        },
    )
    session = load_session(path, "b@example.com")
    assert session["token"] == "B" * 32
    assert session["user_agent"] == "UA-b"


def test_missing_token_is_rejected(tmp_path):
    path = _write(tmp_path, {"cookies": "laravel_session=x", "token": None, "user_agent": "UA"})
    with pytest.raises(ValueError, match="no 'token'"):
        load_session(path)


def test_missing_user_agent_falls_back_to_a_default(tmp_path):
    path = _write(tmp_path, {"cookies": "laravel_session=x", "token": "t" * 32})
    assert load_session(path)["user_agent"]


def test_non_object_session_is_rejected(tmp_path):
    path = _write(tmp_path, ["not", "an", "object"])
    with pytest.raises(TypeError, match="not a JSON object"):
        load_session(path)
