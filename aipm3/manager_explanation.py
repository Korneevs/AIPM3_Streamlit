"""Deterministic, evidence-limited explanations of an existing AIPM result.

This module only reads saved results. It does not rescore a video, combine
diagnostic cohorts, or treat synthetic answers as audience measurements.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any


MANAGER_EXPLANATION_VERSION = "manager-explanation-v1"

_COMPONENTS = (
    ("aipm1", "Заметность",
     "Что помогает выделить ролик при коротком контакте: герой, ситуация или визуальный ход?"),
    ("aipm2", "Запоминаемость",
     "Что можно вспомнить после просмотра и связано ли это с нужным брендом?"),
    ("message_delivery", "Считываемость",
     "Совпадает ли пересказ ролика с задуманным посылом и какие альтернативные трактовки возникают?"),
)
_SENTINELS = {"<BRAND_ONLY>", "<NO_IDEA>"}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number_text(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".").replace(".", ",")


def _component_level(percentile: float) -> str:
    # These are the existing display bands from models.level_from_percentile.
    # Keep this presentation-only module independent of model-loading packages.
    if percentile < 33.0:
        return "Низкий"
    if percentile < 67.0:
        return "Средний"
    return "Высокий"


def build_main_message_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Count only classified full-view recovery answers, retaining uncertainty.

    ``full_count`` is the observed number of rows explicitly labelled ``full``;
    it is never an assumed cohort size. Counts exclude malformed classifications
    and every row sharing a duplicate full-view respondent UID. With partial
    data the counts remain inspectable, but ``majority`` is intentionally None.
    """
    candidate = _text(result.get("main_idea"))
    if candidate in _SENTINELS:
        candidate = ""
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "displayed_candidate": candidate or None,
        "full_count": 0,
        "classified_count": 0,
        "valid_count": 0,
        "brand_only_count": 0,
        "no_idea_count": 0,
        "unclassified_count": 0,
        "unassigned_count": 0,
        "duplicate_row_count": 0,
        "clusters": [],
        "top_ideas": [],
        "support_count": None,
        "support_denominator": 0,
        "majority": None,
        "dominant_idea": None,
        "issues": [],
        "summary": "В сохранённом результате нет ответов после полного просмотра; проверить поддержку посыла нельзя.",
        "caveat": (
            "Это синтетические ответы модели после полного просмотра, а не ответы людей. "
            "Знаменатель — все сохранённые ответы этой группы; ответы после частичного просмотра "
            "и отдельная диагностическая панель сюда не входят. Полнота исходной выгрузки не подтверждена."
        ),
    }
    recovery = result.get("diagnostic_recovery")
    if not isinstance(recovery, list) or not recovery:
        if recovery is not None and not isinstance(recovery, list):
            evidence["issues"].append("Ответы диагностики сохранены в неподдерживаемом формате.")
        return evidence

    full_rows = []
    for row in recovery:
        if not isinstance(row, dict) or not _text(row.get("condition_group")):
            evidence["unassigned_count"] += 1
        elif row["condition_group"] == "full":
            full_rows.append(row)
    evidence["full_count"] = len(full_rows)
    evidence["support_denominator"] = len(full_rows)
    if evidence["unassigned_count"]:
        evidence["issues"].append(
            f"У {evidence['unassigned_count']} строк не удалось определить режим просмотра."
        )
    if not full_rows:
        return evidence

    uid_counts = Counter(_text(row.get("respondent_uid")) for row in full_rows)
    duplicated_uids = {uid for uid, count in uid_counts.items() if uid and count > 1}
    missing_uids = uid_counts.get("", 0)
    if missing_uids:
        evidence["issues"].append(
            f"У {missing_uids} ответов нет идентификатора; проверить отсутствие повторов нельзя."
        )
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for row in full_rows:
        if _text(row.get("respondent_uid")) in duplicated_uids:
            evidence["duplicate_row_count"] += 1
            continue
        answer_type = row.get("answer_type")
        idea = _text(row.get("canonical_idea"))
        if answer_type == "valid" and idea and idea not in _SENTINELS:
            counts[idea] += 1
            evidence["valid_count"] += 1
            raw_answer = _text(row.get("raw_answer"))
            bucket = examples.setdefault(idea, [])
            if raw_answer and raw_answer not in bucket and len(bucket) < 2:
                bucket.append(raw_answer)
        elif answer_type == "brand_only" and idea == "<BRAND_ONLY>":
            evidence["brand_only_count"] += 1
        elif answer_type == "no_idea" and idea == "<NO_IDEA>":
            evidence["no_idea_count"] += 1
        else:
            continue
        evidence["classified_count"] += 1
    evidence["unclassified_count"] = len(full_rows) - evidence["classified_count"]
    if evidence["duplicate_row_count"]:
        evidence["issues"].append(
            f"Исключены {evidence['duplicate_row_count']} строк с повторяющимися идентификаторами; "
            "число независимых ответов по этим строкам не подтверждено."
        )
    invalid_count = evidence["unclassified_count"] - evidence["duplicate_row_count"]
    if invalid_count:
        evidence["issues"].append(
            f"У {invalid_count} ответов отсутствует или противоречива классификация посыла."
        )
    evidence["status"] = "partial" if evidence["issues"] else "available"
    evidence["clusters"] = [
        {"idea": idea, "count": count, "examples": examples.get(idea, [])}
        for idea, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    if counts:
        largest_count = max(counts.values())
        evidence["top_ideas"] = [
            cluster["idea"] for cluster in evidence["clusters"] if cluster["count"] == largest_count
        ]
    else:
        largest_count = 0
    evidence["support_count"] = counts[candidate] if candidate else None

    total = len(full_rows)
    if evidence["status"] == "partial":
        evidence["summary"] = (
            f"Диагностика неполна: сохранено {total} строк полного просмотра, "
            f"классифицировано {evidence['classified_count']}. "
            "По этим данным нельзя надёжно определить преобладающую трактовку."
        )
        return evidence

    evidence["majority"] = largest_count > total / 2
    if evidence["majority"]:
        evidence["dominant_idea"] = evidence["top_ideas"][0]
    if not counts:
        evidence["summary"] = (
            f"Ни в одном из {total} синтетических ответов после полного просмотра "
            "не выделен содержательный посыл: есть только бренд или отсутствие идеи."
        )
    elif len(evidence["top_ideas"]) > 1:
        alternatives = "; ".join(f"«{idea}»" for idea in evidence["top_ideas"])
        evidence["summary"] = (
            f"Одной ведущей трактовки нет: {alternatives} получили по {largest_count} "
            f"из {total} ответов полного просмотра. Ни одна не получила больше половины ответов."
        )
    elif evidence["majority"]:
        evidence["summary"] = (
            f"Трактовка «{evidence['top_ideas'][0]}» встречается в {largest_count} из {total} "
            "ответов полного просмотра — это больше половины сохранённых синтетических ответов."
        )
    else:
        evidence["summary"] = (
            f"Самая частая трактовка — «{evidence['top_ideas'][0]}»: {largest_count} из {total} "
            "ответов полного просмотра. Она не получила больше половины ответов; "
            "однозначное прочтение посыла не подтверждено."
        )
    if candidate and candidate not in evidence["top_ideas"]:
        evidence["summary"] += (
            f" Показанный приложением вариант «{candidate}» поддержали "
            f"{evidence['support_count']} из {total} ответов."
        )
    return evidence


def _profile_value(result: dict[str, Any], feature: str) -> float | None:
    measurements = _mapping(result.get("feature_profile")).get("measurements", [])
    if not isinstance(measurements, list):
        return None
    for measurement in measurements:
        if isinstance(measurement, dict) and measurement.get("feature") == feature:
            return _finite_number(measurement.get("value"))
    return None


def build_manager_explanation(result: dict[str, Any]) -> dict[str, Any]:
    """Describe saved scores and diagnostics without changing their meaning."""
    overall_result = _mapping(result.get("aipm3"))
    index = _finite_number(overall_result.get("index_100"))
    label = _text(overall_result.get("label"))
    headline = "Итоговый индекс отсутствует в сохранённом результате."
    if index is not None:
        headline = f"AIPM 3.0 — {_number_text(index)}"
        if label:
            headline += f": {label.lower()}"
        headline += "."
    overall = {
        "index_100": index,
        "label": label or None,
        "headline": headline,
        "scale_note": (
            "100 — средний индекс референсной выборки; индекс может быть выше 100. "
            "Категория результата определяется границами модели, поэтому индекс ниже 100 "
            "может оставаться в категории «средний». Индекс не равен проценту успеха рекламы."
        ),
    }
    components = []
    for key, component_label, question in _COMPONENTS:
        percentile = _finite_number(_mapping(result.get(key)).get("percentile"))
        if percentile is not None and not 0 <= percentile <= 100:
            percentile = None
        level = _component_level(percentile) if percentile is not None else None
        if level is None:
            outcome = "Оценка компонента отсутствует; вывод о его уровне недоступен."
        else:
            comparative = {"Низкий": "низкая", "Средний": "средняя", "Высокий": "высокая"}[level]
            outcome = (
                f"По прогнозу модели, {component_label.lower()} ролика {comparative} "
                "относительно исторической базы."
            )
        components.append({
            "key": key,
            "label": component_label,
            "percentile": percentile,
            "level": level,
            "position": (
                f"Позиция в исторической базе: {_number_text(percentile)}/100"
                if percentile is not None else "Нет оценки"
            ),
            "outcome": outcome,
            "inspect_question": question,
        })
        # This legacy field is an ordinal prompt rubric, not measured seconds.
        if key == "aipm1":
            focus = _finite_number(_mapping(_mapping(result.get(key)).get("feature_values")).get("message_focus_seconds"))
            rubric = {1: "содержательное предложение не сформулировано",
                      2: "с усилием", 3: "мгновенно"}.get(focus)
            if rubric:
                components[-1]["source_observation"] = (
                    "В разметке AIPM 1.0 формулирование предложения одной фразой после просмотра "
                    f"отнесено к категории «{rubric}»."
                )
    scored = [component for component in components if component["percentile"] is not None]
    weakest = min((component["percentile"] for component in scored), default=None)
    strongest = max((component["percentile"] for component in scored), default=None)
    weakest_keys = [component["key"] for component in scored if component["percentile"] == weakest]
    strongest_keys = [component["key"] for component in scored if component["percentile"] == strongest]
    if not scored:
        summary = "Сравнение компонентов недоступно."
    elif len(scored) != len(_COMPONENTS):
        summary = "Сохранены не все компоненты; полный профиль результата сравнить нельзя."
    elif weakest == strongest:
        summary = "Позиции трёх компонентов совпадают; один относительно слабый компонент не выделяется."
    else:
        weakest_names = ", ".join(component["label"].lower() for component in scored if component["key"] in weakest_keys)
        if len(weakest_keys) == 1:
            summary = (
                f"Самая низкая позиция в исторической базе: {weakest_names}. "
                "Это первый компонент для дополнительной проверки."
            )
        else:
            summary = (
                f"Самые низкие позиции в исторической базе: {weakest_names}. "
                "Эти компоненты стоит проверить в первую очередь."
            )

    message_evidence = build_main_message_evidence(result)
    if weakest_keys == ["aipm1"] and components[0].get("source_observation"):
        summary += " " + components[0]["source_observation"]
    if weakest_keys == ["message_delivery"] and message_evidence["status"] == "available":
        summary += (
            " После полного просмотра содержательный посыл сформулирован в "
            f"{message_evidence['valid_count']} из {message_evidence['full_count']} ответов модели. "
            "Варианты прочтения показаны ниже."
        )

    audio = _profile_value(result, "audio_only_message_completeness")
    visual = _profile_value(result, "visual_only_message_completeness")
    channel_parts = []
    if audio is not None:
        channel_parts.append(f"полнота по звуку — {_number_text(audio)}/3")
    if visual is not None:
        channel_parts.append(f"без звука — {_number_text(visual)}/3")
    observed_channels = ("В профиле " + "; ".join(channel_parts) + ". ") if channel_parts else ""
    discrepancies = [{
        "title": "Полнота сообщения и считываемость отвечают на разные вопросы",
        "detail": (
            observed_channels
            + "Полнота описывает, сколько информации передаёт отдельный канал. "
            "Считываемость — сравнительная оценка модели, которая использует несколько признаков. "
            "Даже высокая полнота по звуку не означает, что все ответы содержат один и тот же посыл. "
            "Проверьте трактовки в ответах после полного просмотра."
        ),
    }, {
        "title": "Наличие приёма ещё не объясняет результат",
        "detail": (
            "Герой, юмор и логотип описывают содержание ролика. Их наличие само по себе "
            "не показывает, насколько удачно они работают, и не гарантирует заметность или запоминаемость. "
            "По одному признаку нельзя объяснить итоговый балл или обещать рост после правки."
        ),
    }]
    return {
        "version": MANAGER_EXPLANATION_VERSION,
        "overall": overall,
        "components": components,
        "weakest_keys": weakest_keys,
        "strongest_keys": strongest_keys,
        "summary": summary,
        "main_message": message_evidence,
        "discrepancies": discrepancies,
        "component_scale_note": (
            "Шкала показывает относительную позицию оценки в исторической базе роликов; "
            "значение /100 не означает долю людей. "
            "Низкий: ниже 33; средний: от 33 до 67, не включая 67; высокий: от 67. "
            "Одинаковая категория не означает одинаковые численные оценки."
        ),
        "uncertainty_notes": [
            "Оценки модели и синтетические ответы служат поводом для проверки гипотез на целевой аудитории.",
            "Сравнение позиций компонентов помогает выбрать приоритет проверки, но не доказывает причины итогового балла или эффект будущей правки.",
            "Для разницы баллов между роликами здесь нет оценки статистической значимости; близкие итоги не доказывают превосходство одного ролика.",
            "Проверка отдельных признаков в пилоте не подтверждает точность итогового индекса и не доказывает эффект будущих правок.",
        ],
    }
