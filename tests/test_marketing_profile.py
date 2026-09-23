"""Semantic coverage, honest priorities, incomplete data, and real result replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import models
from aipm3.feature_profile import GROUPS
from aipm3.marketing_profile import TOPICS, build_marketing_profile
from aipm3.profile_ui import topic_figure


def source():
    result = {"scoring_version": models.SCORING_VERSION, "model_sha256": deepcopy(models.EXPECTED_ARTIFACT_SHA256)}
    for component, (_, groups) in GROUPS.items():
        result[component] = {"feature_effects": {f: 0.0 for names in groups.values() for f in names},
                             "feature_values": {}}
    result["aipm2"]["feature_effects"]["brand_mean_adrecall"] = 1000
    return result


def outcome(profile, topic_id, component):
    return next(row for topic in profile["topics"] if topic["id"] == topic_id
                for row in topic["outcomes"] if row["component"] == component)


def test_five_topics_cover_all_original_groups_and_features_exactly_once():
    profile = build_marketing_profile(source())
    assert len(profile["topics"]) == 5
    for component, (_, groups) in GROUPS.items():
        rows = [r for t in profile["topics"] for r in t["outcomes"] if r["component"] == component]
        covered = [f for row in rows for f in row["features"]]
        expected = [f for fs in groups.values() for f in fs]
        assert len(covered) == len(set(covered)) == len(expected)
        assert set(covered) == set(expected)
        assert sorted(g for row in rows for g in row["source_groups"]) == sorted(groups)
    assert {t["id"] for t in profile["topics"]} == {t["id"] for t in TOPICS}


def test_mixed_topic_keeps_both_signs_in_importance_and_net_in_priority():
    result = source()
    result["aipm1"]["feature_effects"].update(main_character=4, state_transformation=-3, humor=0,
                                            promo=-2, message_focus_seconds=-1)
    p = build_marketing_profile(result)
    story = outcome(p, "story", "aipm1")
    assert story["importance"] == 70
    assert story["positive_share"] == 40
    assert story["negative_share"] == 30
    assert story["direction"] == "up" and story["mixed"]
    assert [row["importance"] for row in story["drivers"]] == [40, 30, 0]
    assert sum(row["importance"] for row in story["drivers"]) == story["importance"]
    message = outcome(p, "message", "aipm1")
    assert message["importance"] == 30 and message["direction"] == "down"
    assert p["priorities"][0]["topics"] == [{"id": "message", "title": "Предложение и главная мысль"}]
    assert sum(r["importance"] for t in p["topics"] for r in t["outcomes"] if r["component"] == "aipm1") == 100


def test_cancelled_topic_retains_importance_but_is_not_a_problem():
    result = source()
    result["aipm1"]["feature_effects"].update(main_character=3, state_transformation=-3)
    p = build_marketing_profile(result)
    row = outcome(p, "story", "aipm1")
    assert row["importance"] == 100 and row["direction"] == "balanced"
    assert row["positive_share"] == row["negative_share"] == 50
    assert not p["priorities"][0]["topics"]


def test_unrelated_outcome_and_fixed_brand_context_cannot_change_a_topic_weight():
    result = source()
    result["aipm2"]["feature_effects"].update(brand_logo_screen_seconds=-3, jingle_present=1)
    original = outcome(build_marketing_profile(result), "brand", "aipm2")
    result["aipm2"]["feature_effects"]["brand_mean_adrecall"] = -1e20
    result["aipm1"]["feature_effects"]["main_character"] = 1e20
    assert outcome(build_marketing_profile(result), "brand", "aipm2") == original
    assert original["importance"] == 75


def test_changing_model_output_units_does_not_change_weights_signs_or_priorities():
    result = source()
    result["aipm1"]["feature_effects"].update(main_character=3, humor=-2, promo=-1)
    original = build_marketing_profile(result)
    for factor in [1e-18, 1e18]:
        scaled = deepcopy(result)
        scaled["aipm1"]["feature_effects"] = {f: v * factor for f, v in result["aipm1"]["feature_effects"].items()}
        report = build_marketing_profile(scaled)
        assert report["priorities"] == original["priorities"]
        for topic in ["story", "message"]:
            before, after = outcome(original, topic, "aipm1"), outcome(report, topic, "aipm1")
            assert before["importance"] == pytest.approx(after["importance"])
            assert before["direction"] == after["direction"]


@pytest.mark.parametrize("bad", [None, "missing", float("nan"), float("inf"), "invalid"])
def test_missing_effect_withholds_entire_outcome_weights_and_priority(bad):
    result = source()
    if bad == "missing":
        del result["aipm1"]["feature_effects"]["humor"]
    else:
        result["aipm1"]["feature_effects"]["humor"] = bad
    p = build_marketing_profile(result)
    for t in p["topics"]:
        for row in t["outcomes"]:
            assert row["available"] == (row["component"] != "aipm1")
            if not row["available"]:
                assert "importance" not in row
    assert not p["priorities"][0]["available"] and not p["priorities"][0]["topics"]


def test_zero_effects_and_equal_negative_contributions_do_not_create_fake_ranking():
    result = source()
    p = build_marketing_profile(result)
    assert all(not row["topics"] for row in p["priorities"])
    assert all(row["importance"] == 0 for topic in p["topics"] for row in topic["outcomes"])
    result["aipm2"]["feature_effects"].update(brand_logo_screen_seconds=-1, jingle_present=-1)
    p = build_marketing_profile(result)
    assert {t["id"] for t in p["priorities"][1]["topics"]} == {"brand", "sound"}


@pytest.mark.parametrize("field", ["scoring_version", "model_sha256"])
def test_incompatible_result_has_no_weights_priorities_or_observed_claims(field):
    result = source()
    result[field] = "wrong"
    result["aipm1"]["feature_values"]["main_character"] = 1
    p = build_marketing_profile(result)
    assert all(not row["available"] for topic in p["topics"] for row in topic["outcomes"])
    assert all(not topic["observations"] for topic in p["topics"])
    assert all(not row["available"] for row in p["priorities"])


def test_missing_observation_is_not_absence_and_fractional_conditions_are_preserved():
    result = source()
    result["aipm1"]["feature_values"] = {"promo": 0, "has_screen_offer_text": None}
    result["message_delivery_business"] = {"offer_condition_count": .1, "cta_clarity": False}
    p = build_marketing_profile(result)
    observed = {label: value for t in p["topics"] for label, _, value in t["observations"]}
    assert observed["Главный герой"] == "Нет данных"
    assert observed["Конкретное предложение на экране"] == "Нет данных"
    assert observed["Акция или бонус"] == "Не обнаружено"
    assert observed["Условия предложения"] == "0,1"


def test_plot_shows_positive_and_negative_mass_on_shared_fixed_axis():
    result = source()
    result["aipm1"]["feature_effects"].update(main_character=4, state_transformation=-3)
    rows = next(t["outcomes"] for t in build_marketing_profile(result)["topics"] if t["id"] == "story")
    fig = topic_figure(rows)
    assert list(fig.layout.xaxis.range) == [-100, 100]
    assert fig.layout.barmode == "relative"
    assert list(fig.data[0].x) == pytest.approx([-300 / 7, 0])
    assert list(fig.data[1].x) == pytest.approx([400 / 7, 0])


def test_read_only_replay_on_three_saved_results_without_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("UI must not run inference or call an API")
    import openai
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    for name in ["score_aipm1", "score_aipm2", "score_message_delivery", "aipm3_score"]:
        monkeypatch.setattr(models, name, forbidden)
    paths = sorted((Path.home() / "Downloads").glob("aipm3_result*.json"))
    if len(paths) < 3:
        pytest.skip("Private saved results unavailable")
    for path in paths:
        result = json.loads(path.read_text())
        original = deepcopy(result)
        p = build_marketing_profile(result)
        assert p == build_marketing_profile(result)
        assert result == original
        for component in GROUPS:
            rows = [r for t in p["topics"] for r in t["outcomes"] if r["component"] == component]
            assert sum(r["importance"] for r in rows) == pytest.approx(100)
            assert sum(r["effect"] for r in rows) == pytest.approx(sum(
                result[component]["feature_effects"][f] for names in GROUPS[component][1].values() for f in names))


def test_ui_has_plain_topics_priorities_definitions_and_no_old_score_scale():
    app = AppTest.from_string('''
import streamlit as st
from aipm3.profile_ui import show_feature_profile
show_feature_profile(st.session_state['result'])
''')
    app.session_state["result"] = source()
    app.run(timeout=30)
    assert not app.exception
    text = " ".join(str(item.value) for kind in ["markdown", "caption", "subheader"] for item in getattr(app, kind))
    for topic in TOPICS:
        assert topic["title"] in text and topic["question"] in text
    assert "С чего начать разбор" in text and "Темы весят по-разному" in text
    assert "Фирменная мелодия" in text and "Призыв к действию" in text
    assert "50 — нейтральный" not in text and "0–100" not in text
    assert len(app.get("plotly_chart")) == 5
    assert not app.button and not app.text_input and not app.text_area
    assert len(app.table) == 8


def test_each_driver_has_a_plain_label_and_its_weight_is_in_the_same_outcome_units():
    result = source()
    for component, (_, groups) in GROUPS.items():
        for i, name in enumerate(f for names in groups.values() for f in names):
            result[component]["feature_effects"][name] = (-1) ** i * (i + 1)
    p = build_marketing_profile(result)
    for topic in p["topics"]:
        for row in topic["outcomes"]:
            assert sum(d["importance"] for d in row["drivers"]) == pytest.approx(row["importance"])
            assert sum(d["effect"] for d in row["drivers"]) == pytest.approx(row["effect"])
            assert all("__" not in d["label"] and "SHAP" not in d["label"] for d in row["drivers"])
            assert [d["importance"] for d in row["drivers"]] == sorted((d["importance"] for d in row["drivers"]), reverse=True)
