"""Scene diagnostic tests use fabricated inputs, never paid model calls."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aipm3 import creative_review as review


def test_new_calls_have_no_history_or_retries(monkeypatch):
    clients = []
    def factory(**kwargs):
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"scenes": []})))
        ])
        clients.append((client, kwargs))
        return client
    monkeypatch.setattr(review, "OpenAI", factory)
    for _ in range(2):
        review.ask_json("test", review.SCENE_SCHEMA, "not-a-real-key", "VIDEO")
    assert len(clients) == 2
    for client, options in clients:
        assert options["max_retries"] == 0
        assert options["timeout"] == 180
        kwargs = client.chat.completions.create.call_args.kwargs
        assert len(kwargs["messages"]) == 1
        assert kwargs["temperature"] == 0
        assert kwargs["model"] == review.MODEL


@pytest.mark.parametrize("start,end", [(-1, 5), (5, 4), (0, 31), (float("nan"), 3), (0, float("inf"))])
def test_scene_bounds_are_checked(monkeypatch, start, end):
    payload = {"scenes": [{"start_seconds": start, "end_seconds": end, "channel": "visual",
                           "observation": "Показан логотип", "quote": "", "groups": ["brand"]}]}
    monkeypatch.setattr(review, "ask_json", lambda *args: payload)
    with pytest.raises(ValueError):
        review.extract_scene_evidence("VIDEO", 30, "test")


def test_wrong_video_rejected_before_request(monkeypatch):
    ask = MagicMock()
    monkeypatch.setattr(review, "ask_json", ask)
    with pytest.raises(ValueError):
        review.review_uploaded_video(b"another-video", ".mp4", "old-hash", "test")
    ask.assert_not_called()


def test_final_scene_rounding_is_bounded_and_explicit():
    payload = {"scenes": [{"start_seconds": 14, "end_seconds": 21, "channel": "visual",
                           "observation": "Логотип", "quote": "", "groups": ["brand"]}]}
    result = review.validate_scene_evidence(payload, 20.73)
    assert result["scenes"][0]["end_seconds"] == 20.73
    assert result["scenes"][0]["reported_end_seconds"] == 21
    assert payload["scenes"][0]["end_seconds"] == 21


def test_manager_does_not_invent_positive_or_negative_signals():
    for sign in [-1, 1]:
        result = {"interpretation": {"levels": {"Считываемость": "Низкий"}, "group_rows": [
            {"Компонент": "Считываемость", "Группа": "Тест", "Локальное влияние, %": sign * 100}
        ]}}
        readout = review.manager_readout(result)
        assert bool(readout["pillars"]["Считываемость"]["strength"]) == (sign > 0)
        assert bool(readout["pillars"]["Считываемость"]["limit"]) == (sign < 0)
    assert review.manager_readout({"interpretation": {"levels": {}, "group_rows": []}})["pillars"] == {}


def test_wrong_video_scene_button_disabled():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string('''
from aipm3.review_ui import show_scene_review
result = {"video_sha": "old", "source_sha": "old"}
show_scene_review(result, "test", b"different", ".mp4")
''').run()
    assert not at.exception
    assert all(button.disabled for button in at.button)
