"""Behavioral guarantees for blind selection, disagreement and empty results."""
import copy
import json
import threading

import httpx
import pytest

from aipm3 import creative_selection as cs
from aipm3 import creative_prompts as prompts


def scenario(sid):
    return {"id": sid, "title": f"Сценарий {sid}",
            "text": f"Герой {sid} открывает дверь. За дверью город из книг. Авито помогает найти книги.",
            "readability_score": 0.99}


def opinion(sid, verdict="nominate", revision=False):
    return {"id": sid, "verdict": verdict, "idea": "Город книг за дверью.",
            "quote": "За дверью город из книг.", "mechanism": "Буквальное превращение библиотеки в город.",
            "brief_link": "Показывает разнообразие книг на Авито.", "risk": "Бренд появляется поздно.",
            "minimal_fix": "Гипотеза: раньше показать поиск на Авито.",
            "reason": "Есть поворот, связанный с ассортиментом книг.",
            "decision_change_reason": "Проверил довод о связи с ассортиментом по финальной реплике." if revision else ""}


def portfolio(ids, *, grouped=False):
    groups = []
    for i, sid in enumerate(ids):
        if grouped and i:
            groups[0]["member_ids"].append(sid)
        else:
            groups.append({"member_ids": [sid], "representative_id": sid,
                           "shared_mechanism": "Город из книг за дверью.",
                           "representative_reason": "Поворот сформулирован в самом сценарии."})
    return {"groups": groups, "details": [
        {"id": sid, "distinctiveness": "Показывает отдельную версию метафоры города.",
         "overflow_reason": "Его отдельный способ раскрыть ассортимент не покрыт предыдущим вариантом.",
         "evidence_quote": "За дверью город из книг."} for sid in ids]}


def run(scenarios, quota=2, **kw):
    return cs.run_selection(brief="Показать ассортимент книг Авито.", scenarios=scenarios, quota=quota,
                            api_key="fake-key", api_base="https://example.invalid/v1", **kw)


def test_no_candidates_does_not_fill_quota_or_call_portfolio(monkeypatch):
    calls = []

    def fake(client, *, model, prompt, payload, validator, **kwargs):
        calls.append((model, prompt, copy.deepcopy(payload)))
        assert prompt == prompts.INDEPENDENT
        assert "quota" not in payload and "models" not in payload
        assert "readability_score" not in json.dumps(payload)
        return validator({"items": [opinion(sid, "reject") for sid in payload["evaluate_ids"]]})

    monkeypatch.setattr(cs, "_call", fake)
    result = run([scenario("a"), scenario("b")], quota=4)
    assert len(calls) == 2
    assert result["primary_ids"] == result["additional_ids"] == result["groups"] == []
    assert result["counts"]["confirmed"] == 0 and result["counts"]["rejected"] == 2


def test_exchange_uses_only_frozen_initial_votes_and_occurs_once(monkeypatch):
    ids = list("abcdefg")
    initial = {
        "a": ("nominate", "nominate"), "b": ("nominate", "reject"),
        "c": ("nominate", "reject"), "d": ("nominate", "reject"),
        "e": ("reject", "reject"), "f": ("insufficient", "insufficient"),
        "g": ("insufficient", "nominate"),
    }
    revised = {"b": ("nominate", "nominate"), "c": ("nominate", "reject"),
               "d": ("reject", "nominate"), "g": ("insufficient", "nominate")}
    calls = []
    progress_threads = []

    def fake(client, *, model, prompt, payload, validator, **kwargs):
        calls.append((model, prompt, copy.deepcopy(payload)))
        index = cs.DEFAULT_MODELS.index(model)
        if prompt == prompts.INDEPENDENT:
            assert "quota" not in payload
            assert len(payload["scenarios"]) == 7  # Both batches see the full pack.
            return validator({"items": [opinion(sid, initial[sid][index]) for sid in payload["evaluate_ids"]]})
        if prompt == prompts.RECONSIDER:
            assert payload["evaluate_ids"] == list("bcdg")
            assert "quota" not in payload and "models" not in payload
            for own, other in zip(payload["own_initial"], payload["other_initial"]):
                assert own["verdict"] == initial[own["id"]][index]
                assert other["verdict"] == initial[other["id"]][1-index]
                assert not own["decision_change_reason"] and not other["decision_change_reason"]
            return validator({"items": [opinion(sid, revised[sid][index], revision=True)
                                        for sid in payload["evaluate_ids"]]})
        assert prompt == prompts.PORTFOLIO and payload["quota"] == 1
        assert [r["id"] for r in payload["candidates"]] == ["a", "b"]
        return validator(portfolio(["a", "b"]))

    monkeypatch.setattr(cs, "_call", fake)
    result = run([scenario(sid) for sid in ids], quota=1,
                 on_progress=lambda _: progress_threads.append(threading.current_thread().name))
    statuses = {r["id"]: r["status"] for r in result["items"]}
    assert statuses == {"a": "agreed_initial", "b": "agreed_after_discussion", "c": "disputed",
                        "d": "disputed", "e": "rejected", "f": "insufficient", "g": "insufficient"}
    assert sum(p == prompts.RECONSIDER for _, p, _ in calls) == 2
    assert result["items"][1]["initial"][1]["verdict"] == "reject"
    assert result["items"][1]["final"][1]["verdict"] == "nominate"
    assert result["primary_ids"] == ["a"] and result["additional_ids"] == ["b"]
    assert result["items"][1]["overflow_reason"]
    assert result["models"] == list(cs.DEFAULT_MODELS)
    assert set(progress_threads) == {threading.current_thread().name}


def test_grouping_keeps_every_confirmed_variant(monkeypatch):
    def fake(client, *, model, prompt, payload, validator, **kwargs):
        if prompt == prompts.INDEPENDENT:
            return validator({"items": [opinion(sid) for sid in payload["evaluate_ids"]]})
        assert prompt == prompts.PORTFOLIO
        return validator(portfolio(["a", "b", "c"], grouped=True))

    monkeypatch.setattr(cs, "_call", fake)
    result = run([scenario(i) for i in "abc"], quota=1)
    assert result["counts"]["confirmed"] == 3
    assert result["counts"]["independent_ideas"] == 1
    assert not result["additional_ids"]
    assert [r["id"] for r in result["items"]] == list("abc")
    assert [r["is_representative"] for r in result["items"]] == [True, False, False]


def test_31_scenarios_all_reach_both_judges_and_all_confirmed_are_retained(monkeypatch):
    ids = [f"s{i:02d}" for i in range(31)]
    seen = {model: [] for model in cs.DEFAULT_MODELS}

    def fake(client, *, model, prompt, payload, validator, **kwargs):
        if prompt == prompts.INDEPENDENT:
            assert [s["id"] for s in payload["scenarios"]] == ids
            assert len(payload["evaluate_ids"]) <= cs.BATCH_SIZE
            seen[model].extend(payload["evaluate_ids"])
            return validator({"items": [opinion(sid) for sid in payload["evaluate_ids"]]})
        assert prompt == prompts.PORTFOLIO
        assert kwargs["tokens"] > 12000
        assert [c["id"] for c in payload["candidates"]] == ids
        return validator(portfolio(ids))

    monkeypatch.setattr(cs, "_call", fake)
    result = run([scenario(sid) for sid in ids], quota=2)
    assert all(seen[model] == ids for model in cs.DEFAULT_MODELS)
    assert [r["id"] for r in result["items"]] == ids
    assert result["counts"]["confirmed"] == result["counts"]["independent_ideas"] == 31
    assert result["counts"]["additional"] == len(result["additional_ids"]) == 29


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invented_quote", "empty_mechanism", "missing_change"])
def test_invalid_judgments_never_become_votes(mutation):
    source = {i: scenario(i) for i in "ab"}
    rows = [opinion(i, revision=True) for i in "ab"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["id"] = "a"
    elif mutation == "invented_quote":
        rows[0]["quote"] = "Герой улетел на Марс."
    elif mutation == "empty_mechanism":
        rows[0]["mechanism"] = ""
    else:
        rows[0]["decision_change_reason"] = ""
    with pytest.raises(ValueError):
        cs._validate_opinions({"items": rows}, list("ab"), source, revision=True)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "bad_representative", "unexplained_overflow", "invented_quote"])
def test_portfolio_cannot_drop_votes_or_fake_overflow(mutation):
    p = portfolio(list("ab"))
    if mutation == "drop":
        p["groups"].pop()
    elif mutation == "duplicate":
        p["groups"][1]["member_ids"] = ["a"]
        p["groups"][1]["representative_id"] = "a"
    elif mutation == "bad_representative":
        p["groups"][0]["representative_id"] = "b"
    elif mutation == "unexplained_overflow":
        p["details"][1]["overflow_reason"] = ""
    else:
        p["details"][0]["evidence_quote"] = "Несуществующая сцена."
    with pytest.raises(ValueError):
        cs._validate_portfolio(p, list("ab"), {i: scenario(i) for i in "ab"}, 1)


def test_provider_failure_aborts_instead_of_rejecting(monkeypatch):
    def fake(*a, **kw):
        raise cs.CreativeServiceError("Модель недоступна")
    monkeypatch.setattr(cs, "_call", fake)
    with pytest.raises(cs.CreativeServiceError):
        run([scenario("a")])


def test_schema_repair_bounded_and_failure_does_not_leak_provider_text():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
            "message": {"content": "private-source-key-do-not-display"}}]})

    with httpx.Client(base_url="https://example.invalid/v1/", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(cs.CreativeServiceError) as err:
            cs._call(client, model=cs.DEFAULT_MODELS[0], prompt=prompts.INDEPENDENT,
                     payload={"evaluate_ids": ["a"]}, validator=lambda p: p)
    assert len(requests) == 3
    assert "private-source" not in str(err.value)
    assert all("private-source" not in json.dumps(req) for req in requests)


def test_no_silent_model_fallback_on_http_error():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(403, text="PRIVATE_PROVIDER_ERROR")
    with httpx.Client(base_url="https://example.invalid/v1/", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(cs.CreativeServiceError) as err:
            cs._call(client, model=cs.DEFAULT_MODELS[1], prompt=prompts.INDEPENDENT,
                     payload={}, validator=lambda p: p)
    assert len(requests) == 1 and requests[0]["model"] == cs.DEFAULT_MODELS[1]
    assert "PRIVATE_PROVIDER_ERROR" not in str(err.value)


def test_input_limits_and_models_are_explicit():
    with pytest.raises(ValueError):
        cs.validate_inputs("Бриф", [scenario("a"), scenario("a")], 2, cs.DEFAULT_MODELS)
    with pytest.raises(ValueError):
        cs.validate_inputs("Бриф", [{**scenario("a"), "text": "x" * (cs.MAX_SCENARIO_CHARS+1)}], 2, cs.DEFAULT_MODELS)
    with pytest.raises(ValueError):
        cs.validate_inputs("Бриф", [scenario("a")], 2, (cs.DEFAULT_MODELS[0], cs.DEFAULT_MODELS[0]))
