"""Brief coding tests use fabricated answers, never paid model calls."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aipm3 import creative_review as review


@pytest.fixture
def brief():
    return {"uvp": "Подходящая одежда", "rtb": "Примерка"}


@pytest.fixture
def answers():
    return [
        {"respondent_id": "p01", "answer": "Можно подобрать одежду, потому что есть примерка"},
        {"respondent_id": "p02", "answer": "Есть одежда и примерка"},
        {"respondent_id": "p03", "answer": "Авито"},
    ]


def coding(answers):
    return {"answers": [{"respondent_id": answer["respondent_id"], **{
        key: {"status": "absent", "quote": ""} for key in review.EVALUATIONS
    }} for answer in answers]}


def test_co_mention_is_not_automatically_a_link(brief, answers):
    first = coding(answers)
    for row, answer in zip(first["answers"][:2], answers[:2]):
        for key in ["uvp", "rtb"]:
            row[key] = {"status": "matched", "quote": answer["answer"]}
    first["answers"][0]["uvp_rtb_link"] = {"status": "matched", "quote": answers[0]["answer"]}
    original = copy.deepcopy(first)
    result = review.combine_codings([first, first], brief, answers)
    assert result["both_uvp_rtb_count"] == 2
    assert result["linked_uvp_rtb_count"] == 1
    assert result["human_readability_percent"] is None
    assert first == original
    for item in result["summary"].values():
        assert sum(item["counts"].values()) == len(answers)


def test_disagreement_is_visible_not_majority_fiction(brief, answers):
    first, second = coding(answers), coding(answers)
    first["answers"][0]["uvp"] = {"status": "matched", "quote": "Можно подобрать одежду"}
    result = review.combine_codings([first, second], brief, answers)
    assert result["answers"][0]["uvp"]["status"] == "uncertain"
    assert result["summary"]["uvp"]["counts"]["matched"] == 0
    assert result["summary"]["uvp"]["counts"]["uncertain"] == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "invented_quote", "no_quote", "false_link"])
def test_rejects_invalid_coding(brief, answers, change):
    payload = coding(answers)
    if change == "missing":
        payload["answers"].pop()
    elif change == "duplicate":
        payload["answers"][1]["respondent_id"] = "p01"
    elif change == "false_link":
        payload["answers"][0]["uvp_rtb_link"] = {"status": "matched", "quote": answers[0]["answer"]}
    else:
        payload["answers"][0]["uvp"] = {
            "status": "unexpected" if change == "unknown" else "matched",
            "quote": "доставка за час" if change == "invented_quote" else "",
        }
    with pytest.raises(ValueError):
        review.combine_codings([payload, payload], brief, answers)


def test_brand_only_blank_and_unconfigured_dimensions(brief, answers):
    answers.append({"respondent_id": "p04", "answer": ""})
    payload = coding(answers)
    for row in payload["answers"]:
        for dimension in review.EVALUATIONS:
            row[dimension] = {"status": "matched", "quote": "invented"}
    brief = {"uvp": "посыл", "rtb": ""}
    result = review.combine_codings(
        [{"answers": payload["answers"][2:]}] * 2, brief, answers[2:]
    )
    assert result["summary"]["uvp"]["counts"]["absent"] == 2
    for dimension in ["rtb", "uvp_rtb_link"]:
        assert result["summary"][dimension]["counts"]["not_set"] == 2


def test_absent_is_distinct_from_explicit_contradiction(brief, answers):
    answers[0]["answer"] = "Примерки нет"
    payload = coding(answers)
    payload["answers"][0]["rtb"] = {"status": "contradicted", "quote": "Примерки нет"}
    result = review.combine_codings([payload, payload], brief, answers)
    assert result["summary"]["rtb"]["counts"]["contradicted"] == 1
    assert result["summary"]["rtb"]["counts"]["absent"] == 2


def test_cache_belongs_to_exact_brief_and_answers(brief, answers):
    result = review.combine_codings([coding(answers)] * 2, brief, answers)
    assert review.alignment_is_current(result, brief, list(reversed(answers)))
    assert not review.alignment_is_current(result, {**brief, "uvp": "другая выгода"}, answers)
    changed = [{**answers[0], "answer": "другое"}, *answers[1:]]
    assert not review.alignment_is_current(result, brief, changed)
    assert not review.alignment_is_current({**result, "version": "old"}, brief, answers)
    assert not review.alignment_is_current(result, brief, [])


def test_compare_uses_two_text_only_calls_without_regenerating_answers(monkeypatch, brief, answers):
    original = copy.deepcopy((brief, answers))
    calls = []
    def ask(prompt, schema, api_key, video_base64=None):
        calls.append((prompt, schema, video_base64))
        return coding(answers)
    monkeypatch.setattr(review, "ask_json", ask)
    result = review.compare_brief(brief, answers, "test")
    assert (brief, answers) == original
    assert len(calls) == 2
    assert all(call[2] is None and call[1] == review.ALIGNMENT_SCHEMA for call in calls)
    assert all(answers[0]["answer"] in call[0] for call in calls)
    assert result["input_hash"]
    with pytest.raises(ValueError):
        review.compare_brief({}, answers, "test")
    assert len(calls) == 2


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


def test_ui_alignment_change_and_failure_do_not_change_scores(monkeypatch, brief, answers):
    from streamlit.testing.v1 import AppTest
    result = {"source_sha": "test", "video_sha": "test", "blind_answers": answers,
              "main_idea": "Подобрать подходящую одежду",
              "diagnostic_recovery": [{"condition_group": "full", "respondent_uid": f"g{i // 4 + 1}_full_p{i + 1:02}",
                                       "raw_answer": answers[i % len(answers)]["answer"]} for i in range(12)],
              "aipm1": {"raw_score": 1}, "aipm2": {"raw_score": 2},
              "message_delivery": {"raw_score": .3}, "aipm3": {"index_100": 90},
              "diagnostics": {"current_brief": brief}}
    core = {k: copy.deepcopy(result[k]) for k in ["aipm1", "aipm2", "message_delivery", "aipm3"]}
    at = AppTest.from_string('''
import streamlit as st
from aipm3.review_ui import show_brief_review
show_brief_review(st.session_state["result"], "test")
''')
    at.session_state["result"] = result
    from aipm3 import message_alignment
    def compare(idea, uvp, answers, key):
        run = {"relation": {"status": "equivalent", "reason": "Один смысл"},
               "answers": [{"respondent_id": a["respondent_id"], **{
                   d: {"status": "absent", "quote": ""} for d in ["main_idea", "uvp"]}} for a in answers]}
        return message_alignment.combine([run, run], idea, uvp, answers)
    fake = MagicMock(side_effect=compare)
    monkeypatch.setattr(message_alignment, "compare", fake)
    from aipm3 import brief_details
    monkeypatch.setattr(brief_details, "compare", MagicMock(side_effect=RuntimeError("details unavailable")))
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 0
    assert any("Основная идея совпадает" in m.value for m in at.success)
    assert fake.call_count == 1
    assert not at.button[0].disabled  # failed RTB details can retry without regenerating comparison
    at.button[0].click().run()
    assert fake.call_count == 1
    at.text_area[0].set_value("другая выгода").run()
    assert len(at.dataframe) == 0
    assert any("Бриф изменён" in i.value for i in at.info)
    fake.side_effect = RuntimeError("secret-must-not-leak")
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    assert "secret-must-not-leak" not in at.error[0].value
    assert len(at.dataframe) == 0
    assert {k: at.session_state["result"][k] for k in core} == core


def test_empty_brief_and_wrong_video_buttons_disabled():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string('''
from aipm3.review_ui import show_brief_review, show_scene_review
result = {"video_sha": "old", "source_sha": "old", "blind_answers": [{"respondent_id": "p01", "answer": "Ответ"}]}
show_brief_review(result, "test")
show_scene_review(result, "test", b"different", ".mp4")
''').run()
    assert not at.exception
    assert all(button.disabled for button in at.button)
