from copy import deepcopy

from aipm3 import latest_uvp
from aipm3.latest_interpretation import public_result


def sample():
    return {"source_sha": "a" * 64, "scores": {"Q": .3}, "feature_rows": {},
            "evidence": {"panel": [[{"respondent_id": str(i),
                                      "main_message_summary": f"Ответ {i}"}
                                     for i in range(30)] for _ in range(3)]}}


def test_panel_survives_export_without_changing_scores_or_combining_repeated_personas():
    source = sample()
    before = deepcopy(source)
    enriched = latest_uvp.with_blind_answers(source)
    assert source == before
    assert len(enriched["blind_answers"]) == 30
    exported = public_result(enriched, {})
    assert "evidence" not in exported
    assert latest_uvp.with_blind_answers(exported)["blind_answers"] == enriched["blind_answers"]
    assert exported["scores"] == before["scores"]


def test_uvp_uses_blind_panel_and_cannot_change_model_result(monkeypatch, tmp_path):
    source = sample()
    before = deepcopy(source)
    target = latest_uvp.target_for_vertical("Goods", "Ресейл")
    calls = []
    def evaluate(result, chosen, key, cache):
        calls.append((result, chosen))
        assert result["blind_answers"][0]["answer"] == "Ответ 0"
        return {"status": "matched", "target": chosen, "source_sha": result["source_sha"]}
    monkeypatch.setattr(latest_uvp, "evaluate_uvp", evaluate)
    out = latest_uvp.with_uvp(source, target, "key", tmp_path)
    assert source == before
    assert out["scores"] == source["scores"]
    assert out["feature_rows"] == source["feature_rows"]
    assert out["vertical_uvp"]["status"] == "matched"
    assert len(calls) == 1


def test_missing_answers_are_not_fabricated_and_failure_preserves_scores(monkeypatch, tmp_path):
    source = {"source_sha": "a", "scores": {"Q": .3}, "independent_evidence": []}
    assert "blind_answers" not in latest_uvp.with_blind_answers(source)
    def fail(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(latest_uvp, "evaluate_uvp", fail)
    out = latest_uvp.with_uvp(source, latest_uvp.target_for_vertical("Travel"), "", tmp_path)
    assert out["scores"] == source["scores"]
    assert out["vertical_uvp"]["status"] == "error"


def test_goods_direction_is_explicit_and_auto_keeps_original_target():
    assert latest_uvp.target_for_vertical("Goods") is None
    assert latest_uvp.target_for_vertical("Goods", "Ресейл")["key"] == "assortment"
    assert latest_uvp.target_for_vertical("Goods", "Распродажа")["key"] == "value"
    assert latest_uvp.target_for_vertical("Auto")["key"] == "trust_safety"
