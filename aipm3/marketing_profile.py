"""Read-only grouping of local attributions into marketer-facing topics.

No cross-model sums, score averaging, new inference, or causal recommendations.
Percentages describe attribution mass within one outcome for this one video.
"""
from __future__ import annotations

import math

from .feature_profile import GROUPS
from .models import EXPECTED_ARTIFACT_SHA256, SCORING_VERSION


VERSION = "marketing-topics-v1"
FEATURE_LABELS = {
    "main_character": "Один главный герой",
    "promo": "Акция или бонус",
    "humor": "Юмор",
    "state_transformation": "Переход от проблемы к решению",
    "message_focus_seconds": "Лёгкость пересказа предложения",
    "has_screen_offer_text": "Конкретное предложение на экране",
    "unique_offer_count": "Число разных предложений",
    "brand_logo_screen_seconds": "Время логотипа в кадре",
    "pack_shot_duration_seconds": "Длительность финального кадра с брендом",
    "ends_with_brand_logo_alone": "Финал с брендом без продолжения сюжета",
    "jingle_present": "Фирменная мелодия",
    "vocal_song_present": "Песня со словами",
    "silence_or_music_only_seconds": "Время без речи",
    "monologue_to_camera": "Обращение прямо к зрителю",
    "scene_pace_high": "Быстрая смена кадров",
    "eng__gemini_round4__signed_mean8": "Сложность и раскрытие предложения",
    "offer_condition_count__r4_mean3": "Число условий предложения",
    "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity": "Призыв к действию и согласие о содержании идеи",
    "eng__product__words_per_second__X__audio_only_message_completeness__r4_value": "Темп речи в сочетании с понятностью звука",
    "words_per_second": "Темп речи отдельно",
    "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3": "Различия между фрагментами с учётом конкретности посыла",
    "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3": "Сходство ответов на фрагменты с учётом конкретности посыла",
    "p12__cluster_valid_mask_mae_smoothed": "Различия между фрагментами отдельно",
}
TOPICS = (
    {
        "id": "message", "title": "Предложение и главная мысль",
        "question": "Что предлагают, в чём выгода и что нужно сделать зрителю?",
        "checks": "Ясность основной мысли, конкретное предложение, число разных предложений, "
                  "условия и ограничения, понятный призыв к действию.",
        "groups": {"aipm1": ("Выделение оффера", "Фокус сообщения"),
                   "message_delivery": ("Полнота и конкретность предложения", "Смысловой фокус и CTA")},
        "look_at": "Проверьте, можно ли пересказать предложение одной фразой: что, для кого, "
                   "на каких условиях и какое действие предлагается.",
        "combined_checks": "«Сложность и раскрытие» объединяет число условий, необходимость объяснять "
                           "непривычное предложение, зависимость смысла от шутки и понятность звука. "
                           "«Согласие о содержании идеи» — совпадение автоматических оценок числа "
                           "элементов, нужных для пересказа: продукт, пользователь, действие, выгода. "
                           "Это оценивается вместе с наличием призыва к действию.",
    },
    {
        "id": "brand", "title": "Бренд в кадре",
        "question": "Помогает ли оформление ролика запомнить рекламируемый бренд?",
        "checks": "Сколько времени виден логотип или фирменный персонаж, как оформлен "
                  "последний кадр с брендом и сколько он длится.",
        "groups": {"aipm2": ("Видимость бренда",)},
        "look_at": "Посмотрите, как бренд связан с историей и предложением, особенно в финале. "
                   "Само по себе увеличение времени логотипа не гарантирует улучшения.",
    },
    {
        "id": "sound", "title": "Музыка и речь",
        "question": "Как звук помогает запомнить ролик и понять предложение?",
        "checks": "Короткая фирменная мелодия, песня со словами, время без речи, "
                  "скорость речи и понятность предложения без изображения.",
        "groups": {"aipm2": ("Аудиальный крючок",),
                   "message_delivery": ("Речь и автономность звука",)},
        "look_at": "Прослушайте ролик без картинки: понятны ли предложение и действие, "
                   "не приходится ли спешить за речью? Отдельно оцените роль музыки.",
        "combined_checks": "Скорость речи учитывается и сама по себе, и в сочетании с тем, "
                           "насколько понятен посыл без изображения. Поэтому понятный звук "
                           "сам по себе не гарантирует положительный вклад всей темы.",
    },
    {
        "id": "story", "title": "Сюжет и подача",
        "question": "Как герой, юмор и темп работают на внимание и запоминание?",
        "checks": "Главный герой, переход от проблемы к решению, заметный юмор, "
                  "обращение прямо к зрителю и скорость смены сцен.",
        "groups": {"aipm1": ("Фокус внимания", "Эмоциональный крючок"),
                   "aipm2": ("Подача и темп",)},
        "look_at": "Посмотрите, помогают ли герой, шутка и монтаж следить за предложением. "
                   "Наличие или отсутствие каждого приёма само по себе не означает ошибку.",
    },
    {
        "id": "partial", "title": "Понятность при неполном просмотре",
        "question": "Что остаётся от смысла, если зритель пропустил часть ролика?",
        "checks": "Как меняются ответы при показе разных фрагментов одной длины, "
                  "и насколько конкретно сформулировано сообщение.",
        "groups": {"message_delivery": ("Устойчивость при неполном просмотре",)},
        "look_at": "Проверьте отдельные фрагменты: остаётся ли понятным предложение, "
                   "если пропустить начало или финал?",
        "combined_checks": "Сравниваются два набора фрагментов для каждой длительности: "
                           "25%, 50% и 75% ролика. Программа учитывает различия в ответах "
                           "и их сочетание с конкретностью сообщения. Это связанные измерения "
                           "одной проверки, поэтому они собраны в одну тему.",
    },
)


def _finite(value):
    if value is None or isinstance(value, (dict, list)):
        raise ValueError("Missing or nonnumeric value")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite value")
    return value


def _number(values, key, suffix="", digits=0):
    try:
        value = _finite(values[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return "Нет данных"
    return f"{value:.{digits}f}".replace(".", ",") + suffix


def _choice(values, key, choices):
    try:
        value = _finite(values[key])
        if not value.is_integer():
            return "Нет данных"
        return choices.get(int(value), "Нет данных")
    except (KeyError, TypeError, ValueError, OverflowError):
        return "Нет данных"


def _observations(result):
    def values(name, field="feature_values"):
        source = result.get(name, {})
        found = source.get(field, {}) if isinstance(source, dict) else {}
        return found if isinstance(found, dict) else {}

    a1, a2 = values("aipm1"), values("aipm2")
    business = result.get("message_delivery_business", {})
    business = business if isinstance(business, dict) else {}
    yes_no = {0: "Не обнаружено", 1: "Есть"}
    return {
        "message": [
            ("Основная мысль после просмотра", "Насколько легко сформулировать предложение одной фразой.",
             _choice(a1, "message_focus_seconds", {1: "Не удаётся сформулировать", 2: "Формулируется с усилием", 3: "Формулируется сразу"})),
            ("Акция или бонус", "Скидка, промокод, подарок или другая явная промо-выгода.", _choice(a1, "promo", yes_no)),
            ("Конкретное предложение на экране", "Текст с ценой, скидкой, сроком или бесплатным предметом/услугой.", _choice(a1, "has_screen_offer_text", yes_no)),
            ("Разные предложения", "Повтор одного предложения в речи и на экране считается один раз.", _number(a1, "unique_offer_count")),
            ("Условия предложения", "Сколько ограничений нужно учесть: например, срок или условия получения выгоды. Среднее автоматических оценок.", _number(business, "offer_condition_count", digits=1)),
            ("Призыв к действию", "Конкретное следующее действие: например, найти специалиста или разместить объявление.", _choice(business, "cta_clarity", yes_no)),
        ],
        "brand": [
            ("Время логотипа в кадре", "Суммарное время заметного логотипа или фирменного персонажа; мелкий знак в углу не учитывается.", _number(a2, "brand_logo_screen_seconds", " сек.")),
            ("Финальный кадр с брендом", "Последний кадр, где основное место занимают бренд, слоган или призыв к действию.", _number(a2, "pack_shot_duration_seconds", " сек.")),
            ("Финал только с брендом и призывом", "Без героев и продолжения сюжета.", _choice(a2, "ends_with_brand_logo_alone", yes_no)),
        ],
        "sound": [
            ("Фирменная мелодия", "Короткий узнаваемый мотив, связанный с брендом; обычная фоновая музыка не считается.", _choice(a2, "jingle_present", yes_no)),
            ("Песня со словами", "Вокальная музыка, а не обычная речь героя или диктора.", _choice(a2, "vocal_song_present", yes_no)),
            ("Время без речи", "Тишина, музыка или звуковые эффекты без речи; это длительность, а не оценка качества.", _number(a2, "silence_or_music_only_seconds", " сек.")),
            ("Темп речи", "Среднее число произнесённых слов за секунду ролика.", _number(business, "words_per_second", " слова/сек.", digits=1)),
            ("Предложение без картинки", "Что можно понять только по звуку, по автоматическому разбору.", _choice(business, "audio_completeness", {0: "Посыл не определяется", 1: "Частично понятен (1 из 3)", 2: "Частично понятен (2 из 3)", 3: "Можно полностью пересказать"})),
        ],
        "story": [
            ("Главный герой", "Один персонаж, на котором держится история и внимание.", _choice(a1, "main_character", yes_no)),
            ("Переход «до — после»", "Показано, как продукт меняет ситуацию героя: было трудно — стало проще.", _choice(a1, "state_transformation", yes_no)),
            ("Юмор", "Заметная шутка или комедийная ситуация; просто позитивный тон не считается.", _choice(a1, "humor", yes_no)),
            ("Обращение к зрителю", "Герой говорит прямо в камеру.", _choice(a2, "monologue_to_camera", yes_no)),
            ("Быстрая смена кадров", "В среднем не реже одной смены кадра в секунду.", _choice(a2, "scene_pace_high", yes_no)),
        ],
        "partial": [
            ("Различие между фрагментами", "Разница долей содержательных автоматических ответов на два набора фрагментов одной длины, в процентных пунктах. Близость ответов сама по себе не доказывает, что смысл понятен.", _number({"gap": business.get("recovery_mask_gap", float("nan")) * 100 if isinstance(business.get("recovery_mask_gap"), (int, float)) else None}, "gap", " п.п.", digits=1)),
            ("Конкретность сообщения", "От упоминания бренда до конкретного объекта, действия и результата. Среднее автоматических оценок по шкале 0–3.", _number(business, "message_specificity", " из 3", digits=1)),
        ],
    }


def build_marketing_profile(result):
    compatible = (result.get("model_sha256") == EXPECTED_ARTIFACT_SHA256
                  and result.get("scoring_version") == SCORING_VERSION)
    sources, totals = {}, {}
    for component, (_, groups) in GROUPS.items():
        raw = result.get(component, {})
        raw = raw.get("feature_effects", {}) if isinstance(raw, dict) else {}
        try:
            if not compatible:
                raise ValueError("Version mismatch")
            sources[component] = {name: _finite(raw[name]) for names in groups.values() for name in names}
            totals[component] = math.fsum(abs(v) for v in sources[component].values())
            if not math.isfinite(totals[component]):
                raise ValueError("Nonfinite total")
        except (KeyError, TypeError, ValueError, OverflowError):
            sources[component], totals[component] = None, None

    observations = _observations(result) if compatible else {}
    topics = []
    for definition in TOPICS:
        topic = {key: value for key, value in definition.items() if key != "groups"}
        topic["observations"] = observations.get(topic["id"], [])
        topic["outcomes"] = []
        for component, group_names in definition["groups"].items():
            features = [feature for group in group_names for feature in GROUPS[component][1][group]]
            outcome = {"component": component, "label": GROUPS[component][0],
                       "features": features, "source_groups": list(group_names),
                       "available": sources[component] is not None}
            if outcome["available"]:
                effects = [sources[component][name] for name in features]
                positive = math.fsum(v for v in effects if v > 0)
                negative = -math.fsum(v for v in effects if v < 0)
                denominator = totals[component]
                net = positive - negative
                outcome.update(
                    effect=net, positive_effect=positive, negative_effect=negative,
                    positive_share=100 * positive / denominator if denominator else 0.0,
                    negative_share=100 * negative / denominator if denominator else 0.0,
                    importance=100 * (positive + negative) / denominator if denominator else 0.0,
                    direction="balanced" if math.isclose(positive, negative, rel_tol=1e-12, abs_tol=0.0) else "up" if net > 0 else "down",
                    mixed=positive > 0 and negative > 0,
                )
                outcome["drivers"] = sorted([
                    {"feature": name, "label": FEATURE_LABELS[name], "effect": sources[component][name],
                     "importance": 100 * abs(sources[component][name]) / denominator if denominator else 0.0,
                     "direction": "balanced" if sources[component][name] == 0 else "up" if sources[component][name] > 0 else "down"}
                    for name in features
                ], key=lambda row: -row["importance"])
            topic["outcomes"].append(outcome)
        topics.append(topic)

    priorities = []
    for component, (label, _) in GROUPS.items():
        candidates = [(topic, outcome) for topic in topics for outcome in topic["outcomes"]
                      if outcome["component"] == component and outcome["available"]
                      and outcome["direction"] == "down"]
        priority = {"component": component, "label": label, "available": sources[component] is not None,
                    "topics": []}
        if candidates:
            weakest = min(outcome["effect"] for _, outcome in candidates)
            priority["topics"] = [{"id": topic["id"], "title": topic["title"]}
                                  for topic, outcome in candidates
                                  if math.isclose(outcome["effect"], weakest, rel_tol=1e-9, abs_tol=0.0)]
        priorities.append(priority)
    return {"version": VERSION, "topics": topics, "priorities": priorities,
            "importance_basis": "Sum of absolute feature attributions within each creative topic / sum of absolute creative feature attributions within that outcome. Fixed brand context excluded.",
            "priority_basis": "Most negative net topic attribution within each outcome; no cross-outcome ranking or causal improvement forecast."}
