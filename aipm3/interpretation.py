from __future__ import annotations

from typing import Any

import numpy as np

from .models import level_from_percentile


AIPM1_GROUPS = {
    "Фокус внимания": ["main_character", "state_transformation"],
    "Эмоциональный крючок": ["humor"],
    "Выделение оффера": ["promo", "has_screen_offer_text"],
    "Фокус сообщения": ["message_focus_seconds", "unique_offer_count"],
}

AIPM2_GROUPS = {
    "Видимость бренда": [
        "brand_logo_screen_seconds",
        "pack_shot_duration_seconds",
        "ends_with_brand_logo_alone",
    ],
    "Аудиальный крючок": [
        "jingle_present",
        "vocal_song_present",
        "silence_or_music_only_seconds",
    ],
    "Подача и темп": ["monologue_to_camera", "scene_pace_high"],
    "Контекст бренда": ["brand_mean_adrecall"],
}

MD_GROUPS = {
    "Полнота и конкретность предложения": [
        "eng__gemini_round4__signed_mean8",
        "offer_condition_count__r4_mean3",
    ],
    "Устойчивость при неполном просмотре": [
        "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3",
        "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3",
        "p12__cluster_valid_mask_mae_smoothed",
    ],
    "Речь и автономность звука": [
        "eng__product__words_per_second__X__audio_only_message_completeness__r4_value",
        "words_per_second",
    ],
    "Смысловой фокус и CTA": [
        "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity",
    ],
}


def _yes_no(value: Any) -> str:
    return "есть" if bool(value) else "нет"


def _effect_rows(
    pillar: str,
    effects: dict[str, float],
    groups: dict[str, list[str]],
    evidence: dict[str, str],
) -> list[dict[str, Any]]:
    raw = {
        group: float(sum(effects.get(feature, 0.0) for feature in features))
        for group, features in groups.items()
    }
    scale = float(sum(abs(value) for value in raw.values())) or 1.0
    rows = []
    for group, value in raw.items():
        signed_share = 100.0 * value / scale
        if group == "Контекст бренда":
            status = "Контекст"
        elif signed_share >= 5.0:
            status = "Работает"
        elif signed_share <= -5.0:
            status = "Ограничивает"
        else:
            status = "Нейтрально"
        rows.append({
            "Компонент": pillar,
            "Группа": group,
            "Статус": status,
            "Локальное влияние, %": signed_share,
            "Наблюдение": evidence[group],
        })
    return rows


def _aipm1_evidence(values: dict[str, Any]) -> dict[str, str]:
    focus = {1: "идея не формулируется", 2: "идея считывается с усилием", 3: "идея считывается сразу"}
    return {
        "Фокус внимания": (
            f"Главный герой: {_yes_no(values['main_character'])}; "
            f"контраст «до/после»: {_yes_no(values['state_transformation'])}."
        ),
        "Эмоциональный крючок": f"Заметный юмор: {_yes_no(values['humor'])}.",
        "Выделение оффера": (
            f"Промо-механика: {_yes_no(values['promo'])}; "
            f"конкретный оффер на экране: {_yes_no(values['has_screen_offer_text'])}."
        ),
        "Фокус сообщения": (
            f"{focus.get(int(values['message_focus_seconds']), 'оценка не определена')}; "
            f"разных офферов: {int(values['unique_offer_count'])}."
        ),
    }


def _aipm2_evidence(values: dict[str, Any]) -> dict[str, str]:
    return {
        "Видимость бренда": (
            f"Логотип заметен около {int(values['brand_logo_screen_seconds'])} сек.; "
            f"финальный пэкшот — {int(values['pack_shot_duration_seconds'])} сек.; "
            f"чистый брендовый финал: {_yes_no(values['ends_with_brand_logo_alone'])}."
        ),
        "Аудиальный крючок": (
            f"Джингл: {_yes_no(values['jingle_present'])}; "
            f"вокальная музыка: {_yes_no(values['vocal_song_present'])}; "
            f"времени без речи: {int(values['silence_or_music_only_seconds'])} сек."
        ),
        "Подача и темп": (
            f"Обращение в камеру: {_yes_no(values['monologue_to_camera'])}; "
            f"высокий темп монтажа: {_yes_no(values['scene_pace_high'])}."
        ),
        "Контекст бренда": (
            "Базовый уровень Avito зафиксирован в модели и не является свойством ролика."
        ),
    }


def _md_evidence(values: dict[str, Any]) -> dict[str, str]:
    return {
        "Полнота и конкретность предложения": (
            f"Индекс полноты раскрытия: {values['completeness_index']:.0f}/100; "
            f"условий предложения: {values['offer_condition_count']:.1f}."
        ),
        "Устойчивость при неполном просмотре": (
            f"Устойчивость: {values['recovery_consistency']:.0%}; "
            f"расхождение двух recovery-масок: {values['recovery_mask_gap']:.3f}."
        ),
        "Речь и автономность звука": (
            f"Полнота сообщения только по звуку: {int(values['audio_completeness'])}/3; "
            f"темп речи: {values['words_per_second']:.2f} слова/сек."
        ),
        "Смысловой фокус и CTA": (
            f"Ясный CTA: {_yes_no(values['cta_clarity'])}; "
            f"согласие по смысловым элементам: {values['semantic_agreement']:.0%}."
        ),
    }


def _top_level_summary(aipm1_level: str, aipm2_level: str, md_level: str) -> str:
    high = "Высокий"
    low = "Низкий"
    if aipm1_level == high and aipm2_level == high and md_level == high:
        return "Ролик сбалансирован: он заметен, запоминается и доносит основную идею."
    if md_level == high and aipm1_level == low:
        return "Сообщение понятно тем, кто увидел ролик; главное ограничение — заметность."
    if aipm1_level == high and md_level == low:
        return "Ролик привлекает внимание, но внимание не превращается в точное понимание идеи."
    if aipm2_level == high and md_level == low:
        return "Ролик может запомниться, но в памяти закрепляется не обязательно главная идея."
    if aipm1_level == high and aipm2_level == low:
        return "Ролик заметен в контакте, но контакт слабо закрепляется в памяти."
    if aipm1_level == low and aipm2_level == low and md_level == low:
        return "Ограничения есть на всех трёх уровнях: заметность, запоминаемость и считываемость."
    if md_level == low:
        return "Основное ограничение ролика — считываемость главной идеи."
    if aipm2_level == low:
        return "Основное ограничение ролика — ожидаемая запоминаемость."
    if aipm1_level == low:
        return "Основное ограничение ролика — ожидаемая заметность."
    return "У ролика нет одного доминирующего провала; итог складывается из нескольких средних сигналов."


def build_interpretation(
    aipm1: dict[str, Any],
    aipm2: dict[str, Any],
    md: dict[str, Any],
    md_business: dict[str, Any],
) -> dict[str, Any]:
    levels = {
        "Заметность": level_from_percentile(float(aipm1["percentile"])),
        "Запоминаемость": level_from_percentile(float(aipm2["percentile"])),
        "Считываемость": level_from_percentile(float(md["percentile"])),
    }

    rows = []
    rows.extend(_effect_rows(
        "Заметность",
        aipm1["feature_effects"],
        AIPM1_GROUPS,
        _aipm1_evidence(aipm1["feature_values"]),
    ))
    rows.extend(_effect_rows(
        "Запоминаемость",
        aipm2["feature_effects"],
        AIPM2_GROUPS,
        _aipm2_evidence(aipm2["feature_values"]),
    ))
    rows.extend(_effect_rows(
        "Считываемость",
        md["feature_effects"],
        MD_GROUPS,
        _md_evidence(md_business),
    ))

    strengths: list[dict[str, str]] = []
    limits: list[dict[str, str]] = []
    for pillar, level in levels.items():
        if level == "Высокий":
            strengths.append({
                "title": f"Высокая {pillar.lower()}",
                "detail": {
                    "Заметность": "Ролик имеет сильный потенциал выделиться в рекламном контакте.",
                    "Запоминаемость": "Элементы ролика поддерживают закрепление контакта в памяти.",
                    "Считываемость": "Главная идея считывается устойчиво и без существенной потери смысла.",
                }[pillar],
            })
        elif level == "Низкий":
            limits.append({
                "title": f"Низкая {pillar.lower()}",
                "detail": {
                    "Заметность": "Ролик рискует не выделиться среди других рекламных контактов.",
                    "Запоминаемость": "Контакт с роликом слабо закрепляется в памяти.",
                    "Считываемость": "Главная идея считывается реже и менее устойчиво.",
                }[pillar],
            })

    creative_rows = [row for row in rows if row["Группа"] != "Контекст бренда"]
    positive = sorted(
        (row for row in creative_rows if row["Статус"] == "Работает"),
        key=lambda row: row["Локальное влияние, %"],
        reverse=True,
    )
    negative = sorted(
        (row for row in creative_rows if row["Статус"] == "Ограничивает"),
        key=lambda row: row["Локальное влияние, %"],
    )
    for row in positive[:3]:
        strengths.append({"title": row["Группа"], "detail": row["Наблюдение"]})
    for row in negative[:3]:
        limits.append({"title": row["Группа"], "detail": row["Наблюдение"]})

    if not strengths:
        strongest = max(creative_rows, key=lambda row: row["Локальное влияние, %"])
        strengths.append({"title": strongest["Группа"], "detail": strongest["Наблюдение"]})
    if not limits:
        weakest = min(creative_rows, key=lambda row: row["Локальное влияние, %"])
        limits.append({"title": weakest["Группа"], "detail": weakest["Наблюдение"]})

    return {
        "levels": levels,
        "summary": _top_level_summary(
            levels["Заметность"], levels["Запоминаемость"], levels["Считываемость"]
        ),
        "strengths": strengths[:5],
        "limits": limits[:5],
        "group_rows": rows,
    }
