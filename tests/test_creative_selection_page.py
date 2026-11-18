"""Offline boundary checks for the separate creative selector page."""
from contextlib import contextmanager
import json

import streamlit as st
from streamlit.testing.v1 import AppTest

from app_pages import creative_selection as page


def _row(source, status="rejected"):
    verdict = "nominate" if status in {"agreed_initial", "agreed_after_discussion", "disputed"} else "reject"
    opinion = {"id": source["id"], "verdict": verdict, "reason": "Основание из сценария", "quote": "Реплика.", "mechanism": "Поворот", "brief_link": page.NOT_ASSESSED}
    return {**source, "status": status, "idea": "Идея", "quote": "Реплика.", "mechanism": "Поворот", "brief_link": page.NOT_ASSESSED,
            "reason": "Идея помогает запомнить продукт", "risk": page.NOT_ASSESSED, "minimal_fix": page.NOT_ASSESSED, "initial": [opinion, {**opinion, "verdict": "reject" if status == "disputed" else verdict}],
            "final": [opinion, {**opinion, "verdict": "reject" if status == "disputed" else verdict}],
            "discussion_happened": status in {"disputed", "agreed_after_discussion"}, "distinctiveness": "Самостоятельный ход", "overflow_reason": ""}


def _result(scenarios, quota=2, statuses=None):
    rows = [_row(source, (statuses or ["rejected"] * len(scenarios))[i]) for i, source in enumerate(scenarios)]
    agreed = [row["id"] for row in rows if row["status"] in page._CONFIRMED]
    for row in rows:
        if row["id"] in agreed[quota:]:
            row["overflow_reason"] = "Самостоятельная идея, которую не покрывают остальные"
    return {"protocol_version": page.PROTOCOL_VERSION, "items": rows, "quota": quota, "models": list(page.DEFAULT_MODELS), "primary_ids": agreed[:quota], "additional_ids": agreed[quota:], "groups": []}


def _app(monkeypatch, call):
    @contextmanager
    def slot(*, wait, on_queue):
        assert wait
        on_queue(1)
        on_queue(0)
        yield
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: [])
    monkeypatch.setattr(page, "api_key", lambda: "offline")
    monkeypatch.setattr(page, "analysis_slot", slot)
    monkeypatch.setattr(page, "run_selection", call)
    return AppTest.from_string("from app_pages.creative_selection import main; main()").run()


def _enter(app, pasted="Реплика."):
    app.text_area(key="creative_pasted").set_value(pasted)
    app.run()
    return app


def test_only_submit_runs_and_zero_is_not_filled_and_changes_hide_old_report(monkeypatch):
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        kwargs["on_progress"]("Сравниваем идеи")
        return _result(kwargs["scenarios"], kwargs["quota"])
    app = _app(monkeypatch, call)
    assert not app.exception and app.button(key="creative_run").disabled
    _enter(app)
    assert not calls
    app.button(key="creative_run").click().run()
    assert not app.exception and len(calls) == 1
    assert "Согласованных креативных кандидатов не найдено" in app.info[0].value
    app.run()
    assert len(calls) == 1
    app.text_area(key="creative_pasted").set_value("Другой сценарий.").run()
    assert len(calls) == 1 and not app.metric
    app.text_area(key="creative_pasted").set_value("Реплика.").run()
    assert len(app.metric) == 3 and len(calls) == 1
    app.number_input(key="creative_quota").set_value(3).run()
    assert not app.metric and len(calls) == 1


def test_failure_keeps_completed_same_input_report_but_never_creates_consensus(monkeypatch):
    failed = False
    def call(**kwargs):
        if failed:
            raise RuntimeError("private provider payload must not leak")
        return _result(kwargs["scenarios"])
    app = _enter(_app(monkeypatch, call))
    app.button(key="creative_run").click().run()
    assert len(app.metric) == 3
    failed = True
    app.button(key="creative_run").click().run()
    assert not app.exception and len(app.metric) == 3
    assert app.error and "private provider" not in str(app.error)
    app.text_area(key="creative_pasted").set_value("Другой сценарий.").run()
    app.button(key="creative_run").click().run()
    assert not app.exception and not app.metric
    assert "Незавершённый отбор" in app.error[0].value


def test_overflow_and_dispute_are_separate_and_full_text_can_be_downloaded(monkeypatch):
    def call(**kwargs):
        return _result(kwargs["scenarios"], kwargs["quota"], ["agreed_initial", "agreed_after_discussion", "agreed_initial", "disputed"])
    pasted = "\n\n".join(f"=== Идея {i} ===\nРеплика." for i in range(4))
    app = _enter(_app(monkeypatch, call), pasted)
    app.button(key="creative_run").click().run()
    assert not app.exception
    assert {"Прошедшие отбор", "Дополнительные идеи", "Спорные кандидаты"} <= {item.value for item in app.subheader}
    assert [m.value for m in app.metric] == ["4", "3", "1"]
    result = app.session_state["creative_result"][1]
    assert len(json.loads(json.dumps(result))["items"]) == 4
    assert b"\xef\xbb\xbf" == page.report_csv(result)[:3]
    assert any("GPT 6.1 Sol" in item.value for item in app.markdown)
    assert any("Claude Opus 5.5" in item.value for item in app.markdown)
    assert not any("GPT 6.1 Sol" in item.value or "Claude Opus 5.5" in item.value
                   for expander in app.expander for item in expander.markdown)
    assert page.NOT_ASSESSED not in page.report_csv(result).decode("utf-8-sig")


def test_input_issue_prevents_model_call(monkeypatch):
    calls = []
    app = _enter(_app(monkeypatch, lambda **kwargs: calls.append(kwargs)), "=== Без текста ===")
    assert app.warning and app.button(key="creative_run").disabled
    assert not calls and not app.exception


def test_no_brief_required_and_old_protocol_result_is_hidden(monkeypatch):
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        return _result(kwargs["scenarios"], statuses=["agreed_initial"])
    app = _enter(_app(monkeypatch, call))
    assert not any(item.key == "creative_brief" for item in app.text_area)
    assert not app.button(key="creative_run").disabled
    app.button(key="creative_run").click().run()
    assert len(calls) == 1 and "brief" not in calls[0]
    assert app.metric
    monkeypatch.setattr(page, "PROTOCOL_VERSION", "different-rubric")
    app.run()
    assert not app.metric and len(calls) == 1
