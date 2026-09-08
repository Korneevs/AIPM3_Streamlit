"""Comparison must keep independent videos and scoring versions separate."""
from copy import deepcopy

from aipm3.result_history import remember_result


def result(sha="video-a", version="v1", protocol="p1", value=70):
    return {"source_sha": sha, "scoring_version": version, "protocol_version": protocol,
            "aipm3": {"index_100": value, "label": "Средний уровень"},
            "aipm1": {"percentile": 55}, "aipm2": {"percentile": 38},
            "message_delivery": {"percentile": 7}, "prepared_video": "private-path"}


def test_rerun_replaces_same_video_without_mutating_history_or_result():
    first = result()
    history = remember_result([], first, "/private/input.mp4")
    before = deepcopy(history)
    again = result(value=90)
    source = deepcopy(again)
    updated = remember_result(history, again, "renamed.mp4")
    assert len(updated) == 1 and updated[0]["index_100"] == 90
    assert updated[0]["name"] == "renamed.mp4"
    assert history == before and again == source
    assert history[0]["name"] == "input.mp4"
    assert "prepared_video" not in updated[0]


def test_different_content_same_name_and_different_versions_remain_separate():
    history = []
    for item in [result(), result(sha="video-b"), result(version="v2"), result(protocol="p2")]:
        history = remember_result(history, item, "same.mp4")
    assert len(history) == 4


def test_history_is_bounded_and_requires_identity():
    history = []
    for i in range(12):
        history = remember_result(history, result(sha=str(i)), "video.mp4")
    assert len(history) == 10
    assert [row["source_sha"] for row in history] == [str(i) for i in range(2, 12)]
    assert remember_result(history, {"aipm3": {}}, "unknown.mp4") == history
