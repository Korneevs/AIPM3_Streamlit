"""Read-only grouping of local attributions into marketer-facing topics.

No cross-model sums, score averaging, new inference, or causal recommendations.
Percentages describe attribution mass within one outcome for this one video.
"""
from __future__ import annotations

import math

from .feature_profile import GROUPS
from .models import EXPECTED_ARTIFACT_SHA256, SCORING_VERSION


VERSION = "marketing-topics-v3"
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
    "eng__gemini_round4__signed_mean8": "Объяснение предложения целиком",
    "offer_condition_count__r4_mean3": "Число условий предложения",
    "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity": "Призыв и состав сообщения",
    "eng__product__words_per_second__X__audio_only_message_completeness__r4_value": "Смысл речи при текущем темпе",
    "words_per_second": "Скорость речи",
    "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3": "Разброс понимания и конкретность",
    "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3": "Повторяемость смысла и конкретность",
    "p12__cluster_valid_mask_mae_smoothed": "Разница между фрагментами",
}
# One topic belongs to exactly one predicted outcome. Original features are
# indivisible: engineered interactions remain joint checks, never causal parts.
TOPICS = (
    ("attention", "Герой и развитие истории", "aipm1", "Фокус внимания"),
    ("humor", "Юмор", "aipm1", "Эмоциональный крючок"),
    ("offer", "Промо и предложение на экране", "aipm1", "Выделение оффера"),
    ("message", "Одна главная мысль", "aipm1", "Фокус сообщения"),
    ("brand", "Бренд в кадре", "aipm2", "Видимость бренда"),
    ("music", "Музыка и паузы в речи", "aipm2", "Аудиальный крючок"),
    ("pacing", "Обращение к зрителю и монтаж", "aipm2", "Подача и темп"),
    ("complexity", "Сложность объяснения и условий", "message_delivery", "Полнота и конкретность предложения"),
    ("cta", "Следующее действие", "message_delivery", "Смысловой фокус и CTA"),
    ("speech", "Темп речи и смысл без картинки", "message_delivery", "Речь и автономность звука"),
    ("partial", "Понятность при неполном просмотре", "message_delivery", "Устойчивость при неполном просмотре"),
)

OBSERVATION_LABELS = {
    "main_character": ("Главный герой",),
    "state_transformation": ("Переход «до — после»",),
    "humor": ("Юмор",),
    "promo": ("Акция или бонус",),
    "has_screen_offer_text": ("Конкретное предложение на экране",),
    "message_focus_seconds": ("Основная мысль после просмотра",),
    "unique_offer_count": ("Разные предложения",),
    "brand_logo_screen_seconds": ("Время логотипа в кадре",),
    "pack_shot_duration_seconds": ("Финальный кадр с брендом",),
    "ends_with_brand_logo_alone": ("Финал только с брендом и призывом",),
    "jingle_present": ("Фирменная мелодия",),
    "vocal_song_present": ("Песня со словами",),
    "silence_or_music_only_seconds": ("Время без речи",),
    "monologue_to_camera": ("Обращение к зрителю",),
    "scene_pace_high": ("Быстрая смена кадров",),
    "eng__gemini_round4__signed_mean8": ("Условия предложения", "Предложение без картинки", "Непривычность предложения", "Зависимость смысла от шутки"),
    "offer_condition_count__r4_mean3": ("Условия предложения",),
    "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity": ("Призыв к действию", "Согласие о составе главной мысли"),
    "eng__product__words_per_second__X__audio_only_message_completeness__r4_value": ("Темп речи", "Предложение без картинки"),
    "words_per_second": ("Темп речи",),
    "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3": ("Различие между фрагментами", "Конкретность сообщения"),
    "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3": ("Различие между фрагментами", "Конкретность сообщения"),
    "p12__cluster_valid_mask_mae_smoothed": ("Различие между фрагментами",),
}

CHECKS = {
    "main_character": "Сравнить текущую версию с историей, в которой внимание удерживает один герой.",
    "state_transformation": "Сделать результат использования сервиса отдельным понятным моментом истории.",
    "humor": "Сравнить шутку с более прямой подачей того же предложения.",
    "promo": "Проверить, выделяется ли выгода и понятны ли условия акции с первого просмотра.",
    "has_screen_offer_text": "Проверить версию с коротким, читаемым предложением на экране.",
    "message_focus_seconds": "Сформулировать предложение одной фразой и связать с ней сюжет и финал.",
    "unique_offer_count": "Оставить в центре одно предложение; второстепенные тезисы проверить на необходимость.",
    "brand_logo_screen_seconds": "Проверить заметное появление бренда рядом с демонстрацией выгоды.",
    "pack_shot_duration_seconds": "Дать время прочитать бренд и предложение на финальном кадре.",
    "ends_with_brand_logo_alone": "Проверить финал, где сюжет уже завершён и внимание остаётся на бренде и предложении.",
    "jingle_present": "Сравнить текущий звук с коротким узнаваемым мотивом бренда.",
    "vocal_song_present": "Проверить, помогает ли текст песни запомнить именно бренд и предложение.",
    "silence_or_music_only_seconds": "Проверить баланс речевого сообщения и музыкальных пауз.",
    "monologue_to_camera": "Сравнить текущую подачу с прямым обращением героя к зрителю.",
    "scene_pace_high": "Проверить монтаж: хватает ли времени заметить бренд и прочитать предложение.",
    "eng__gemini_round4__signed_mean8": "Проверить прямое объяснение выгоды и условий без необходимости разгадывать шутку.",
    "offer_condition_count__r4_mean3": "Проверить понятность необходимых условий; необязательные детали вынести из главного сообщения.",
    "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity": "Связать одно конкретное действие с понятной выгодой: что сделать и зачем.",
    "eng__product__words_per_second__X__audio_only_message_completeness__r4_value": "Прослушать ролик без картинки: проверить темп и полноту фразы о предложении.",
    "words_per_second": "Сравнить текущую речь с версией, где меньше слов и есть пауза на ключевую выгоду.",
    "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3": "Проверить, сохраняется ли конкретная выгода в начале, середине и конце ролика.",
    "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3": "Проверить, можно ли понять одно и то же предложение по разным фрагментам.",
    "p12__cluster_valid_mask_mae_smoothed": "Продублировать ключевую мысль в разных частях и проверить ролик с пропусками.",
}

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


def _partial_finding(result):
    """Report full and partial-view answer rates only with complete evidence."""
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


def _profile_observations(result):
    observed = {label: (label, definition, value)
                for rows in _observations(result).values() for label, definition, value in rows}
    panel = result.get("diagnostic_panel", [])
    for key, label, definition in [
        ("offer_novelty_explanation_need", "Непривычность предложения", "Сколько объяснения требует продукт или условие: 0 — не требует, 3 — требует подробного объяснения. Среднее автоматических оценок."),
        ("irony_or_twist_dependency", "Зависимость смысла от шутки", "Насколько для понимания нужно разгадать шутку или поворот: 0 — не нужно, 3 — смысл сильно зависит от этого. Среднее автоматических оценок."),
    ]:
        values = []
        for row in panel if isinstance(panel, list) else []:
            try:
                values.append(_finite(row[key]))
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        value = _number({"v": math.fsum(values) / len(values)}, "v", " из 3", 1) if values else "Нет данных"
        observed[label] = (label, definition, value)
    business = result.get("message_delivery_business", {})
    observed["Согласие о составе главной мысли"] = (
        "Согласие о составе главной мысли",
        "Насколько совпали автоматические оценки числа элементов для пересказа: продукт, пользователь, действие и выгода. Совпадение не означает простоту мысли.",
        _number(business, "semantic_agreement", digits=2))
    return observed


def _check(feature, result):
    business = result.get("message_delivery_business", {})
    a1 = result.get("aipm1", {}).get("feature_values", {})
    a2 = result.get("aipm2", {}).get("feature_values", {})
    conditional = {
        ("main_character", 1): "Проверить, связывает ли история главного героя внимание с продуктом и его выгодой.",
        ("state_transformation", 1): "Проверить, ясно ли показано, что именно продукт изменил ситуацию героя.",
        ("humor", 0): "Сравнить текущую подачу с понятным юмористическим моментом, связанным с предложением.",
        ("promo", 0): "Проверить версию, в которой конкретная выгода для зрителя выделена отдельной фразой.",
        ("has_screen_offer_text", 1): "Проверить читаемость предложения: длину текста, размер и время показа.",
        ("jingle_present", 1): "Проверить, связывается ли фирменная мелодия с появлением бренда и главным предложением.",
        ("vocal_song_present", 0): "Сравнить текущий звук с песенной подачей, в которой звучит бренд или ключевая выгода.",
        ("monologue_to_camera", 1): "Проверить, выделяет ли прямое обращение героя конкретную выгоду для зрителя.",
        ("ends_with_brand_logo_alone", 1): "Проверить, хватает ли времени прочитать бренд и предложение на уже выделенном финальном кадре.",
    }
    value = a1.get(feature, a2.get(feature))
    if isinstance(value, (int, float, str)) and (feature, value) in conditional:
        return conditional[feature, value]
    if feature == "unique_offer_count" and isinstance(value, (int, float)) and value <= 1:
        return "Уточнить главную выгоду так, чтобы её можно было сформулировать одной фразой."
    if feature in {"eng__gemini_round4__signed_mean8", "offer_condition_count__r4_mean3"} and business.get("offer_condition_count") == 0:
        return "Проверить версию, где конкретная польза для зрителя названа прямо и подкреплена действием. Добавлять условия ради оценки не нужно."
    return CHECKS[feature]


def build_marketing_profile(result):
    compatible = (result.get("model_sha256") == EXPECTED_ARTIFACT_SHA256
                  and result.get("scoring_version") == SCORING_VERSION)
    observations = _profile_observations(result) if compatible else {}
    sources, totals = {}, {}
    for component, (_, groups) in GROUPS.items():
        try:
            if not compatible:
                raise ValueError("Version mismatch")
            raw = result[component]["feature_effects"]
            sources[component] = {f: _finite(raw[f]) for names in groups.values() for f in names}
            totals[component] = _finite(math.fsum(abs(v) for v in sources[component].values()))
        except (KeyError, TypeError, ValueError, OverflowError):
            sources[component], totals[component] = None, None
    topics = []
    for topic_id, title, component, group in TOPICS:
        features = GROUPS[component][1][group]
        available = sources[component] is not None
        # No cancellation inside a card and no pooling across model outcomes.
        partitions = {}
        for feature in features:
            effect = sources[component][feature] if available else None
            direction = "unknown" if effect is None else "up" if effect > 0 else "down" if effect < 0 else "balanced"
            partitions.setdefault(direction, []).append(feature)
        for direction, selected in partitions.items():
            drivers = []
            for feature in selected:
                details = [observations[label] for label in OBSERVATION_LABELS[feature] if label in observations]
                drivers.append({"feature": feature, "label": FEATURE_LABELS[feature],
                    "observations": details, "direction": direction,
                    "effect": sources[component][feature] if available else None,
                    "importance": 100 * abs(sources[component][feature]) / totals[component] if available and totals[component] else 0,
                    "check": _check(feature, result) if direction == "down" and any(value != "Нет данных" for _, _, value in details) else None})
            drivers.sort(key=lambda row: -row["importance"])
            outcome = {"component": component, "label": GROUPS[component][0],
                       "features": selected, "source_groups": [group], "available": available,
                       "direction": direction, "mixed": False, "drivers": drivers}
            if available:
                effect = math.fsum(sources[component][f] for f in selected)
                importance = math.fsum(d["importance"] for d in drivers)
                outcome.update(effect=effect, importance=importance,
                    positive_share=importance if direction == "up" else 0.0,
                    negative_share=importance if direction == "down" else 0.0)
            details = list(dict.fromkeys(item for driver in drivers for item in driver["observations"]))
            known = [label + ": " + value.lower().rstrip(".") for label, _, value in details if value != "Нет данных"]
            finding = ". ".join(known) + "." if known else "Недостаточно сохранённых наблюдений по этому свойству."
            if topic_id == "partial" and details and any(f.startswith("eng__") for f in selected):
                finding = _partial_finding(result) + " " + finding
            # A split card names its actual properties, not a broad mixed topic.
            specific_title = " · ".join(d["label"] for d in drivers) if len(selected) == 1 else title
            conclusion = {
                "down": "Эти свойства снижают оценку «" + GROUPS[component][0] + "» в расчёте модели.",
                "up": "Эти свойства поддерживают оценку «" + GROUPS[component][0] + "» в расчёте модели.",
                "balanced": "В этом ролике эти свойства не меняют оценку модели.",
                "unknown": "Недостаточно данных для оценки вклада.",
            }[direction]
            topics.append({"id": topic_id + "_" + direction, "topic_id": topic_id,
                "title": specific_title, "component": component, "direction": direction,
                "finding": finding, "conclusion": conclusion, "observations": details,
                "outcomes": [outcome], "importance": outcome.get("importance", 0)})
    order = {"down": 0, "up": 1, "balanced": 2, "unknown": 3}
    topics.sort(key=lambda t: (list(GROUPS).index(t["component"]), order[t["direction"]], -t["importance"]))
    return {"version": VERSION, "topics": topics,
            "importance_basis": "Absolute feature attribution / total absolute creative attribution within one model outcome. Each feature appears once. Fixed brand context excluded. Signs never cancel inside a card."}
