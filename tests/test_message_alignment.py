from copy import deepcopy

import pytest

from aipm3 import message_alignment as alignment


ANSWERS = [
    {"respondent_id": "p1", "answer": "Можно найти любую машину: новую и с пробегом."},
    {"respondent_id": "p2", "answer": "Большой выбор машин, включая проверенные."},
    {"respondent_id": "p3", "answer": "Найти автомобиль."},
]
MAIN = "Можно найти любую машину"
UVP = "Широта ассортимента"


def coding(main=("matched", "matched", "partial"), uvp=None, relation="equivalent", answers=ANSWERS):
    uvp = uvp or main
    return {
        "relation": {"status": relation, "reason": "Сравнение существенного смысла формулировок."},
        "answers": [{
            "respondent_id": answer["respondent_id"],
            "main_idea": {"status": m, "quote": answer["answer"] if m != "absent" else ""},
            "uvp": {"status": u, "quote": answer["answer"] if u != "absent" else ""},
        } for answer, m, u in zip(answers, main, uvp)],
    }


def combined(first=None, second=None, main_idea=MAIN, uvp=UVP, answers=ANSWERS):
    first = first or coding()
    return alignment.combine([first, second or deepcopy(first)], main_idea, uvp, answers)


def test_equivalent_ideas_equal_rates_and_same_panel():
    result = combined()
    assert result["relation"]["status"] == "equivalent"
    assert result["summary"]["uvp"]["matched_count"] == 2
    assert result["summary"]["uvp"]["matched_pct"] == pytest.approx(200 / 3)
    assert result["n"] == 3
    assert result["gap"] == {"pp": 0, "lower_pp": 0, "upper_pp": 0, "direction": "equal"}
    assert result["human_readability_percent"] is None


def test_equal_rates_do_not_imply_same_meaning():
    result = combined(coding(relation="different"))
    assert result["gap"]["direction"] == "equal"
    assert result["relation"]["status"] == "different"


@pytest.mark.parametrize("main,uvp,direction,gap", [
    (("matched", "matched", "partial"), ("matched", "partial", "absent"), "main_higher", 100 / 3),
    (("partial", "partial", "absent"), ("matched", "partial", "absent"), "uvp_higher", -100 / 3),
])
def test_gap_direction_is_signed(main, uvp, direction, gap):
    result = combined(coding(main, uvp, "partial"))
    assert result["gap"]["direction"] == direction
    assert result["gap"]["pp"] == pytest.approx(gap)
    assert result["gap"]["lower_pp"] == result["gap"]["upper_pp"]


def test_uncertain_coding_bounds_keep_original_denominator():
    first = coding(("matched", "matched", "absent"), ("matched", "partial", "absent"), "partial")
    second = coding(("matched", "partial", "absent"), ("matched", "matched", "absent"), "partial")
    result = combined(first, second)
    assert result["n"] == 3
    assert result["answers"][1]["main_idea"]["status"] == "uncertain"
    assert result["summary"]["main_idea"]["counts"]["uncertain"] == 1
    assert result["summary"]["main_idea"]["lower_matched_count"] == 1
    assert result["summary"]["main_idea"]["upper_matched_count"] == 2
    assert result["gap"]["direction"] == "uncertain"
    assert result["gap"]["pp"] == 0
    assert result["gap"]["lower_pp"] == pytest.approx(-100 / 3)
    assert result["gap"]["upper_pp"] == pytest.approx(100 / 3)


def test_bounds_are_paired_not_independent_rate_endpoints():
    first = coding(("matched", "matched", "partial"))
    second = coding(("matched", "partial", "partial"))
    result = combined(first, second)
    assert result["summary"]["uvp"]["lower_matched_count"] == 1
    assert result["summary"]["uvp"]["upper_matched_count"] == 2
    # Disagreement affects both equivalent ideas together, not their difference.
    assert result["gap"]["lower_pp"] == result["gap"]["upper_pp"] == 0


def test_semantic_disagreement_explicit():
    result = combined(coding(), coding(relation="partial"))
    assert result["relation"]["status"] == "uncertain"
    assert result["relation"]["coder_statuses"] == ["equivalent", "partial"]


def test_equivalence_with_inconsistent_answer_codes_suppresses_gap():
    first = coding(uvp=("partial", "matched", "partial"))
    original = deepcopy(first)
    result = combined(first)
    assert result["diagnostic_conflicts"]
    assert result["gap"] == {"pp": None, "lower_pp": None, "upper_pp": None, "direction": "conflict"}
    assert result["summary"]["uvp"]["matched_count"] == 1  # No silent normalization.
    assert first == original


def test_identical_input_wrong_semantic_label_is_conflict():
    result = combined(coding(relation="different"), main_idea=UVP)
    assert result["diagnostic_conflicts"]
    assert result["gap"]["direction"] == "conflict"


def test_reproducible_and_inputs_not_mutated():
    source = coding()
    saved = deepcopy((source, ANSWERS))
    first = combined(source)
    assert first == combined(source)
    assert (source, ANSWERS) == saved
    assert alignment.is_current(first, MAIN, UVP, list(reversed(ANSWERS)))


@pytest.mark.parametrize("change", ["idea", "uvp", "answer", "version", "model"])
def test_cache_invalidation(change):
    result = combined()
    main, uvp, answers = MAIN, UVP, deepcopy(ANSWERS)
    if change == "idea":
        main += " проверенную"
    elif change == "uvp":
        uvp += " и надежность"
    elif change == "answer":
        answers[0]["answer"] += " Проверенную."
    else:
        result[change] = "old"
    assert not alignment.is_current(result, main, uvp, answers)


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "unknown", "forged_quote", "empty_quote", "absent_quote", "status", "relation", "extra"])
def test_invalid_coding_rejected(malformation):
    bad = coding()
    if malformation == "missing":
        bad["answers"].pop()
    elif malformation == "duplicate":
        bad["answers"][1]["respondent_id"] = "p1"
    elif malformation == "unknown":
        bad["answers"][1]["respondent_id"] = "p4"
    elif malformation in {"forged_quote", "empty_quote"}:
        bad["answers"][0]["uvp"]["quote"] = "Выдуманная цитата" if malformation == "forged_quote" else ""
    elif malformation == "absent_quote":
        bad["answers"][0]["uvp"]["status"] = "absent"
    elif malformation == "status":
        bad["answers"][0]["uvp"]["status"] = "not_set"
    elif malformation == "relation":
        bad["relation"]["status"] = "same"
    else:
        bad["answers"][0]["rtb"] = bad["answers"][0]["uvp"]
    with pytest.raises(ValueError):
        combined(bad)


def test_brand_only_blank_forced_absent_and_not_discarded():
    answers = [{"respondent_id": "p1", "answer": "Авито!"}, {"respondent_id": "p2", "answer": " "}]
    run = coding(("matched", "matched"), answers=answers)
    result = combined(run, answers=answers)
    assert result["n"] == 2
    assert result["summary"]["main_idea"]["counts"]["absent"] == 2
    assert result["summary"]["uvp"]["matched_count"] == 0


def test_contradiction_is_not_absence():
    result = combined(coding(("matched", "contradicted", "absent")))
    assert result["summary"]["uvp"]["counts"]["contradicted"] == 1
    assert result["answers"][1]["uvp"]["quotes"] == [ANSWERS[1]["answer"]]


def test_compare_two_fresh_text_calls_no_rtb_and_no_mutation(monkeypatch):
    calls = []

    def fake(prompt, schema, api_key, video_base64=None):
        calls.append((prompt, schema, api_key, video_base64))
        return coding()

    original = deepcopy(ANSWERS)
    monkeypatch.setattr(alignment.review, "ask_json", fake)
    result = alignment.compare(MAIN, UVP, ANSWERS, "test")
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]
    assert all(call[3] is None and call[1] == alignment.SCHEMA for call in calls)
    assert '"rtb"' not in calls[0][0]
    assert ANSWERS == original
    assert result["summary"]["uvp"]["matched_count"] == 2


def test_blank_inputs_and_duplicate_respondents_rejected():
    with pytest.raises(ValueError):
        alignment.compare(" ", UVP, ANSWERS, "test")
    with pytest.raises(ValueError):
        alignment.compare(MAIN, UVP, [ANSWERS[0], ANSWERS[0]], "test")
    assert not alignment.is_current(combined(), "", UVP, ANSWERS)


def recovery_result():
    full = [{
        "condition_group": "full",
        "respondent_uid": f"g{1 + (number - 1) // 4}_full_p{number:02d}",
        "respondent_id": f"p{number:02d}",
        "raw_answer": f"Ответ {number}",
        "answer_type": "valid",
    } for number in range(1, 13)]
    full[0].update(raw_answer="", answer_type="no_idea")
    full[1].update(raw_answer="Авито Авто", answer_type="brand_only")
    full[2].update(raw_answer="Я не понял идею.", answer_type="no_idea")
    masks = [{**row, "condition_group": "nested_50_m1", "raw_answer": "НЕ ВКЛЮЧАТЬ"}
             for row in full]
    return {"diagnostic_recovery": [*full, *masks], "blind_answers": ANSWERS}


def test_source_panel_keeps_blank_brand_only_and_no_idea_denominator():
    result = recovery_result()
    before = deepcopy(result)
    answers = alignment.source_answers(result)
    assert len(answers) == 12
    assert answers[0] == {"respondent_id": "g1_full_p01", "answer": ""}
    assert answers[1]["answer"] == "Авито Авто"
    assert answers[2]["answer"] == "Я не понял идею."
    assert not any(row["answer"] == "НЕ ВКЛЮЧАТЬ" for row in answers)
    assert all(set(row) == {"respondent_id", "answer"} for row in answers)
    assert result == before


@pytest.mark.parametrize("malformation", ["missing", "extra", "duplicate", "blank_id", "bad_group", "wrong_group_size", "nonstring_answer", "no_panel"])
def test_source_panel_rejects_incomplete_or_noncanonical_ids(malformation):
    result = recovery_result()
    if malformation == "missing":
        result["diagnostic_recovery"].pop(0)
    elif malformation == "extra":
        result["diagnostic_recovery"].append(deepcopy(result["diagnostic_recovery"][0]))
    elif malformation == "duplicate":
        result["diagnostic_recovery"][1]["respondent_uid"] = "g1_full_p01"
    elif malformation == "blank_id":
        result["diagnostic_recovery"][0]["respondent_uid"] = ""
    elif malformation == "bad_group":
        result["diagnostic_recovery"][0]["respondent_uid"] = "g4_full_p01"
    elif malformation == "wrong_group_size":
        result["diagnostic_recovery"][4]["respondent_uid"] = "g1_full_p05"
    elif malformation == "nonstring_answer":
        result["diagnostic_recovery"][0]["raw_answer"] = None
    else:
        result.pop("diagnostic_recovery")
    with pytest.raises(ValueError):
        alignment.source_answers(result)  # Never falls back to blind_answers30.


@pytest.mark.parametrize("answer", [
    "Авито Авто", "Реклама Авито Авто.", "Авито Услуги", "Авито Недвижимость", "Авито Работа",
    "Avito Auto", "Avito Services", "Avito Real Estate", "Avito Jobs", "Реклама Avito Work!",
    "Advertisement for Avito Auto.",
])
def test_vertical_brand_only_forced_absent(answer):
    answers = [{"respondent_id": "p1", "answer": answer}]
    result = combined(coding(("matched",), answers=answers), answers=answers)
    assert result["summary"]["uvp"]["counts"]["absent"] == 1
    assert result["answers"][0]["main_idea"]["quotes"] == []


@pytest.mark.parametrize("answer", [
    "Авито Авто — большой выбор автомобилей.",
    "Реклама Авито Услуги: быстро найти мастера.",
    "Авито Работа помогает найти подработку рядом.",
    "Avito Real Estate has flats without commission.",
])
def test_meaningful_brand_sentences_are_not_forced_absent(answer):
    answers = [{"respondent_id": "p1", "answer": answer}]
    result = combined(coding(("matched",), answers=answers), answers=answers)
    assert result["summary"]["uvp"]["counts"]["matched"] == 1
