"""The marker for a disputed reading and the four-of-five strength gate are independent."""
import json
import threading
import pytest

from aipm3 import uvp_scoring as scoring


def image(key, strength=3, direction="positive"):
    return {"id": key, "strength": strength, "direction": direction, "clarity": 3, "takeaway": "Услуги Авито",
            "reason_to_believe": "Герой выбирает мастера через сервис", "alternative_reading": "",
            "limitations": [], "evidence": [{"source_ids": ["V01"], "observation": "Герой выбирает мастера"}]}


def test_protocol_on_new_creative_is_blind_and_preserves_unflagged_ratings(monkeypatch):
    calls = []
    questions = [{"id": "q1", "statement": "Первый имидж", "group": "Удобство"},
                 {"id": "q2", "statement": "Второй имидж", "group": "Ассортимент", "secondary_groups": ["Выгода"]}]
    monkeypatch.setattr(scoring, "_MEANING_CACHE", {})

    def fake_call(client, model, prompt, packet, sheets, validate, tokens):
        judge = scoring.PANEL.index(model)
        if prompt == scoring._BLIND:
            assert "questions" not in packet and not any(q["statement"] in json.dumps(packet, ensure_ascii=False) for q in questions)
            result = {"main_idea": "Авито Услуги", "impression": [], "recall": []}
        elif prompt == scoring.prompts.MEANINGS:
            assert not sheets and "speech" not in packet
            result = {"items": [dict(key=q["id"], belief="смысл", object="бренд", stage="поиск", scope_and_parts="всё", not_equivalent="другое", valid_routes=["образ"]) for q in questions]}
        elif prompt == scoring.prompts.EVALUATE:
            assert packet["blind_first_impression"]["main_idea"] == "Авито Услуги"
            assert all("group" not in q for q in packet["questions"])
            result = {"main_idea": "main", "images": [image("q1", 3 if judge < 3 else 2), image("q2")]}
        elif prompt == scoring._AUDIT:
            first = (model, prompt) not in calls
            result = {"checked_images": 10, "main_idea_judge": 2,
                      "issues": [{"model_index": 0, "image_id":"q1", "type":"scope_error", "reason":"Этап", "quote":"Сцена"}] if first else []}
        else:
            assert prompt == scoring.prompts.REPAIR and packet["repair_ids"] == ["q1"] and judge == 0
            result = {"main_idea": "main", "images": [image("scope:q1", 3), image("q2", 0, "neutral")]}
        validate(result)
        calls.append((model, prompt))
        return result

    monkeypatch.setattr(scoring, "_call", fake_call)
    times = {f"V{i:02d}": float(i) for i in range(1, 41)}
    thread = threading.get_ident()
    def progress(text):
        assert threading.get_ident() == thread
    result = scoring.score_video(vertical="Услуги", questions=questions, evidence={"sheets":["data:image/jpeg;base64,ABC"],
        "frame_times":times, "speech":[]}, api_base="https://gateway.example/v1", api_key="test", on_progress=progress)
    assert result["main_idea"] == "Авито Услуги" and len(result["images"]) == 2
    assert [r["score"] for r in result["images"]] == [2, 3]
    assert not any(r["review"] for r in result["images"])
    assert result["images"][1]["secondary_groups"] == ["Выгода"]
    assert sum(p == scoring._AUDIT for _, p in calls) == 2
    assert sum(p == scoring.prompts.REPAIR for _, p in calls) == 1


def test_five_historical_votes_no_false_promotion():
    from aipm3.uvp_strength import corroborated
    assert corroborated([3, 3, 3, 2, 2], [True]*5) == (2, 3)
    assert corroborated([3]*5, [True]*4 + [False]) == (3, 4)
    assert corroborated([-2, -2, -2, -1, 0], [True]*5) == (-1, 3)
    assert corroborated([-1, -1, 0, 1, 2], [True]*5)[0] == 0
    with pytest.raises(ValueError):
        corroborated([3]*4, [True]*4)
