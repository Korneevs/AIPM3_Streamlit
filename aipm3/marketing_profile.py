"""Read-only grouping of local attributions into marketer-facing topics.

No cross-model sums, score averaging, new inference, or causal recommendations.
Percentages describe attribution mass within one outcome for this one video.
"""
from __future__ import annotations

import math

from .feature_profile import GROUPS
from .models import EXPECTED_ARTIFACT_SHA256, SCORING_VERSION


VERSION = "marketing-topics-v2"
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
        "id": "offer", "title": "Предложение и выгода",
        "groups": {"aipm1": ("Выделение оффера",),
                   "message_delivery": ("Полнота и конкретность предложения",)},
        "combined_checks": "Сложность предложения учитывает его условия, необходимость объяснять "
                           "непривычную выгоду, зависимость смысла от шутки и понятность звука.",
    },
    {
        "id": "message", "title": "Главная мысль и призыв к действию",
        "groups": {"aipm1": ("Фокус сообщения",),
                   "message_delivery": ("Смысловой фокус и CTA",)},
        "combined_checks": "Призыв к действию оценивается вместе с совпадением автоматических "
                           "оценок числа элементов, нужных для пересказа: продукт, пользователь, "
                           "действие и выгода.",
    },
    {"id": "brand", "title": "Бренд в кадре", "groups": {"aipm2": ("Видимость бренда",)}},
    {"id": "music", "title": "Музыка", "groups": {"aipm2": ("Аудиальный крючок",)}},
    {
        "id": "speech", "title": "Речь и понятность на слух",
        "groups": {"message_delivery": ("Речь и автономность звука",)},
        "combined_checks": "Скорость речи учитывается сама по себе и в сочетании с понятностью "
                           "посыла без изображения. Вклад относится к этому сочетанию.",
    },
    {
        "id": "story", "title": "Сюжет и подача",
        "groups": {"aipm1": ("Фокус внимания", "Эмоциональный крючок"),
                   "aipm2": ("Подача и темп",)},
    },
    {
        "id": "partial", "title": "Понятность при неполном просмотре",
        "groups": {"message_delivery": ("Устойчивость при неполном просмотре",)},
        "combined_checks": "Сравниваются два набора фрагментов для каждой длительности: "
                           "25%, 50% и 75% ролика. Учитываются различия в автоматических "
                           "ответах и конкретность сообщения.",
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
        "offer": [
            ("Акция или бонус", "Скидка, промокод, подарок или другая явная промо-выгода.", _choice(a1, "promo", yes_no)),
            ("Конкретное предложение на экране", "Текст с ценой, скидкой, сроком или бесплатным предметом/услугой.", _choice(a1, "has_screen_offer_text", yes_no)),
            ("Условия предложения", "Сколько ограничений нужно учесть: например, срок или условия получения выгоды. Среднее автоматических оценок.", _number(business, "offer_condition_count", digits=1)),
        ],
        "message": [
            ("Основная мысль после просмотра", "Насколько легко сформулировать предложение одной фразой.",
             _choice(a1, "message_focus_seconds", {1: "Не удаётся сформулировать", 2: "Формулируется с усилием", 3: "Формулируется сразу"})),
            ("Разные предложения", "Повтор одного предложения в речи и на экране считается один раз.", _number(a1, "unique_offer_count")),
            ("Призыв к действию", "Конкретное следующее действие: например, найти специалиста или разместить объявление.", _choice(business, "cta_clarity", yes_no)),
        ],
        "brand": [
            ("Время логотипа в кадре", "Суммарное время заметного логотипа или фирменного персонажа; мелкий знак в углу не учитывается.", _number(a2, "brand_logo_screen_seconds", " сек.")),
            ("Финальный кадр с брендом", "Последний кадр, где основное место занимают бренд, слоган или призыв к действию.", _number(a2, "pack_shot_duration_seconds", " сек.")),
            ("Финал только с брендом и призывом", "Без героев и продолжения сюжета.", _choice(a2, "ends_with_brand_logo_alone", yes_no)),
        ],
        "music": [
            ("Фирменная мелодия", "Короткий узнаваемый мотив, связанный с брендом; обычная фоновая музыка не считается.", _choice(a2, "jingle_present", yes_no)),
            ("Песня со словами", "Вокальная музыка, а не обычная речь героя или диктора.", _choice(a2, "vocal_song_present", yes_no)),
            ("Время без речи", "Тишина, музыка или звуковые эффекты без речи; это длительность, а не оценка качества.", _number(a2, "silence_or_music_only_seconds", " сек.")),
        ],
        "speech": [
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


def topic_finding(topic, result):
    """Use recorded observations only; missing observations never become absence."""
    known = {label: value for label, _, value in topic["observations"] if value != "Нет данных"}
    if not known:
        return "Недостаточно сохранённых данных, чтобы описать эту сторону ролика."
    if topic["id"] == "partial":
        records = result.get("diagnostic_recovery", [])
        if isinstance(records, list):
            conditions = {f"nested_{fraction}_m{mask}" for fraction in (25, 50, 75) for mask in (1, 2)}
            rows = [row for row in records if isinstance(row, dict)]
            full = [row for row in rows if row.get("condition_group") == "full"]
            parts = [row for row in rows if row.get("condition_group") in conditions]
            if (full and {row["condition_group"] for row in parts} == conditions
                    and all(row.get("answer_type") in {"valid", "brand_only", "no_idea"}
                            for row in full + parts)):
                full_rate = sum(row["answer_type"] == "valid" for row in full) / len(full)
                part_rate = sum(row["answer_type"] == "valid" for row in parts) / len(parts)
                return (f"В автоматической проверке содержательный пересказ получился в {full_rate:.0%} "
                        f"ответов на полный ролик и в {part_rate:.0%} ответов на фрагменты длиной 25–75% ролика.")
        return "Полные данные автоматической проверки фрагментов не сохранились; сделать вывод о понятности при неполном просмотре нельзя."
    binary_phrases = {
        "Акция или бонус": ("Акция или бонус не обнаружены", "Есть акция или бонус"),
        "Конкретное предложение на экране": ("Конкретное предложение текстом на экране не обнаружено", "Конкретное предложение вынесено на экран"),
        "Призыв к действию": ("Явный призыв к действию не обнаружен", "Есть явный призыв к действию"),
        "Фирменная мелодия": ("Фирменная мелодия не обнаружена", "Есть фирменная мелодия"),
        "Песня со словами": ("Песня со словами не обнаружена", "Звучит песня со словами"),
        "Главный герой": ("Единого главного героя не обнаружено", "История строится вокруг одного главного героя"),
        "Переход «до — после»": ("Переход от проблемы к решению не обнаружен", "Показан переход от проблемы к решению"),
        "Юмор": ("Выраженный юмор не обнаружен", "Есть юмор"),
        "Обращение к зрителю": ("Прямого обращения героя к зрителю нет", "Герой обращается прямо к зрителю"),
        "Быстрая смена кадров": ("Быстрая смена кадров не обнаружена", "Кадры сменяются быстро"),
    }
    phrases = {
        "offer": [("Акция или бонус", "Акция или бонус — {}"),
                  ("Конкретное предложение на экране", "конкретное предложение текстом на экране — {}")],
        "message": [("Основная мысль после просмотра", "Главная мысль: {}"),
                    ("Призыв к действию", "явный призыв к действию — {}")],
        "brand": [("Время логотипа в кадре", "Логотип или фирменный персонаж видны {}"),
                  ("Финальный кадр с брендом", "финальный кадр с брендом длится {}")],
        "music": [("Фирменная мелодия", "Фирменная мелодия — {}"),
                  ("Песня со словами", "песня со словами — {}")],
        "speech": [("Предложение без картинки", "Посыл без изображения: {}"),
                   ("Темп речи", "средний темп речи — {}")],
        "story": [("Главный герой", "Главный герой — {}"),
                  ("Переход «до — после»", "переход «до — после» — {}"),
                  ("Юмор", "юмор — {}"),
                  ("Обращение к зрителю", "обращение прямо к зрителю — {}"),
                  ("Быстрая смена кадров", "быстрая смена кадров — {}")],
    }
    parts = []
    for label, template in phrases[topic["id"]]:
        if label not in known:
            continue
        if label in binary_phrases:
            part = binary_phrases[label][known[label] == "Есть"]
        else:
            part = template.format(known[label].lower())
        parts.append(part[0].upper() + part[1:].rstrip("."))
    return ". ".join(parts) + "." if parts else "Для вывода по этой теме не хватает сохранённых наблюдений."


def topic_conclusion(outcomes):
    labels = {"aipm1": "заметность", "aipm2": "запоминаемость", "message_delivery": "считываемость"}
    clauses = []
    for direction, verb in [("up", "поддерживает"), ("down", "снижает")]:
        names = [labels[row["component"]] for row in outcomes if row["available"] and row["direction"] == direction]
        if names:
            clauses.append(verb + " прогнозируемую " + " и ".join(names))
    if clauses:
        conclusion = "По расчёту модели, этот блок " + ", но ".join(clauses) + "."
    elif any(row["available"] and row["importance"] > 0 for row in outcomes):
        conclusion = "В расчёте модели плюсы и минусы этого блока уравновешены."
    elif any(row["available"] for row in outcomes):
        conclusion = "В этом ролике блок не меняет оценку модели."
    else:
        return "Недостаточно данных, чтобы оценить вклад этого блока."
    if any(not row["available"] for row in outcomes):
        missing = [row["label"].lower() for row in outcomes if not row["available"]]
        conclusion += " Не хватает данных для остальных оценок: " + ", ".join(missing) + "."
    return conclusion


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
        topic["finding"] = topic_finding(topic, result)
        topic["conclusion"] = topic_conclusion(topic["outcomes"])
        topics.append(topic)

    return {"version": VERSION, "topics": topics,
            "importance_basis": "Sum of absolute feature attributions within each creative topic / sum of absolute creative feature attributions within that outcome. Fixed brand context excluded."}
