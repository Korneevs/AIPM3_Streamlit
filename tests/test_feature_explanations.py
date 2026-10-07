from copy import deepcopy

import pytest

from aipm3.feature_explanations import FEATURE_MEANINGS, feature_observation
from aipm3.latest_interpretation import LABELS
from aipm3.manager_report import HIDDEN_MANAGER_FEATURES


FACT = "Герой открывает приложение и выбирает товар."


def driver(feature="product_demo_seconds", *, verified=True, observation=FACT, **values):
    return dict(feature=feature, evidence=dict(verified=verified, observation=observation, episodes=[]), **values)


def test_every_visible_feature_has_a_definition():
    visible = set(LABELS) - HIDDEN_MANAGER_FEATURES
    assert visible <= FEATURE_MEANINGS.keys()
    assert not {"is_celeb", "brand_history"} & FEATURE_MEANINGS.keys()
    assert all(FEATURE_MEANINGS[feature].strip() for feature in visible)


@pytest.mark.parametrize("value", [0, 3.6666666667, 99])
def test_verified_fact_is_independent_of_scoring_value_and_does_not_mutate_input(value):
    item = driver(value=value, usable=False, interpretation_kind="association_only", direction="limits")
    before = deepcopy(item)
    assert feature_observation(item) == FACT
    assert item == before


@pytest.mark.parametrize("verified", [False, None, "true", 1])
def test_unverified_absence_is_not_published_even_with_a_usable_zero_value(verified):
    claim = "Демонстрации сервиса нет."
    result = feature_observation(driver(verified=verified, observation=claim, value=0, usable=True))
    assert claim not in result
    assert "не удалось" in result


def test_separately_verified_fact_is_neutral_and_unverified_fallback_is_rejected():
    item = driver(verified=False, observation="Демонстрации сервиса нет.", value=0)
    item["factual_observation"] = dict(value=8, evidence=dict(verified=True, observation=FACT))
    assert feature_observation(item) == "Отдельное наблюдение: " + FACT
    item["factual_observation"]["evidence"]["verified"] = False
    assert FACT not in feature_observation(item)


def test_verified_primary_fact_takes_precedence_over_separate_fallback():
    item = driver()
    item["factual_observation"] = dict(evidence=dict(verified=True, observation="В финале показан логотип."))
    assert feature_observation(item) == FACT


@pytest.mark.parametrize("feature", ["fresh__audiovisual_claim_alignment", "brand_first_mention_seconds", "jingle_present"])
def test_incomplete_audio_blocks_audio_facts_but_preserves_visual_observation(feature):
    result = feature_observation(driver(feature), audio_incomplete=True)
    assert "Без полной озвучки" in result
    assert FACT not in result
    assert feature_observation(driver(), audio_incomplete=True) == FACT


@pytest.mark.parametrize("location", ["observation", "episode"])
def test_unverified_identity_in_description_or_episode_cannot_leak(location):
    item = driver("main_character")
    name = "Иван Иванов"
    if location == "observation":
        item["evidence"]["observation"] = name + " играет главного героя."
    else:
        item["evidence"]["episodes"] = [dict(start=1, end=2, description=name + " открывает приложение.")]
    result = feature_observation(item)
    assert name not in result
    assert "требует проверки" in result


def test_duration_from_evidence_is_approximate_and_never_replaced_by_averaged_input():
    item = driver(value=3.6666666667, observation="Демонстрация сервиса длится 5 секунд.")
    result = feature_observation(item)
    assert "приблизительной" in result and "5 секунд" in result
    assert "3.666" not in result
    item["evidence"]["observation"] = "Демонстрация сервиса длится примерно 5 секунд."
    assert feature_observation(item) == item["evidence"]["observation"]


def test_alignment_uses_confirmed_presence_without_inferring_absence_from_rating():
    item = driver("fresh__audiovisual_claim_alignment", value=0, observation="Rating 2: action is shown.")
    item["evidence"].update(shown_action=True, shown_result=False)
    result = feature_observation(item)
    assert "Действие" in result and "показано" in result
    assert "подтвердить не удалось" in result
    assert "Rating" not in result and "результата нет" not in result
    item["evidence"].update(shown_action=False, shown_result=False)
    assert "нельзя уверенно подтвердить" in feature_observation(item)


@pytest.mark.parametrize("claim", [
    "Смысл понятен каждому.",
    "Каждый зритель запомнит бренд.",
    "Все сразу поймут предложение.",
])
def test_corroborated_feature_does_not_verify_absolute_audience_claims(claim):
    item = driver("panel__mandatory_inference_chain_length", value=0, observation=claim)
    result = feature_observation(item)
    assert claim not in result
    assert "требует проверки" in result
    item["evidence"]["observation"] = claim + " " + FACT
    assert feature_observation(item) == FACT


def test_pack_shot_wording_keeps_a_single_final_frame_and_logo():
    item = driver("pack_shot_duration_seconds", observation=
                  "Финальный пэкшот с логотипом Авито и слоганом длится 3 секунды.")
    assert feature_observation(item) == (
        "По приблизительной оценке времени: Финальный кадр с логотипом Авито и слоганом длится 3 секунды."
    )


def test_timestamp_ranges_remain_approximate_video_observations():
    observation = "Крупные планы лица героя показаны с 0:02 по 0:04 и с 0:08 по 0:10."
    item = driver("character_close_up_seconds", value=3, observation=observation)
    assert feature_observation(item) == "По приблизительной оценке времени: " + observation


@pytest.mark.parametrize("observation", [
    "Процесс покупки и продажи вещей на Авито привычен и не требует дополнительных объяснений.",
    "Продукт привычен, дополнительных объяснений не требуется.",
])
def test_offer_complexity_reports_review_findings_without_claiming_audience_familiarity(observation):
    item = driver("panel__offer_novelty_explanation_need", value=0, observation=observation,
                  interpretation_kind="association_only", direction="limits")
    assert feature_observation(item) == (
        "В проверках ролика не выделено сложных для объяснения условий предложения."
    )
