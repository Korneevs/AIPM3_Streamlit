"""Explicit user input, applied after frozen inference without biometric matching.

The fixed policy multiplies the AIPM2 component index, not raw ad recall or
SHAP. Historical reference means stay fixed. Shared inference caches therefore
remain independent of each user's selection; session/export results carry it.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .models import SCORING_VERSION, aipm3_score


VERSION = "manual-celebrity-v1"
CELEBRITIES = {
    "none": "Нет",
    "fomenko": "Николай Фоменко",
    "zhuravlyov": "Дмитрий Журавлёв",
    "kurkova": "Равшана Куркова",
}
MULTIPLIER = 1.3


def selected_celebrity(result: dict) -> str:
    return result.get("celebrity_adjustment", {}).get("selection", "none")


def apply_celebrity(result: dict, selection: str) -> dict:
    """Replace a manual choice, always starting from the original score.

    Base percentiles still describe the frozen model. norm_level classifies
    the adjusted index against the same index thresholds used by the UI;
    it does not pretend the adjusted index is an observed percentile.
    """
    if selection not in CELEBRITIES:
        raise ValueError("Неизвестный участник")
    if result.get("scoring_version") != SCORING_VERSION:
        raise ValueError("Результат относится к другой версии расчёта")
    previous = result.get("celebrity_adjustment")
    if previous is not None:
        if (not isinstance(previous, dict) or previous.get("version") != VERSION
                or previous.get("source") != "manual"
                or previous.get("selection") not in CELEBRITIES
                or previous.get("source_sha") != result.get("source_sha")):
            raise ValueError("Не удалось проверить сохранённый выбор селебрити")
        base_a2, base_a3 = previous.get("base_aipm2"), previous.get("base_aipm3")
    else:
        base_a2, base_a3 = result.get("aipm2"), result.get("aipm3")
    try:
        base_index = float(base_a2["reference_index"])
        total_index = float(base_a3["index"])
        if not all(math.isfinite(v) and v > 0 for v in (base_index, total_index)):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Исходные оценки отсутствуют или некорректны") from None

    adjusted = deepcopy(result)
    adjusted["aipm2"] = deepcopy(base_a2)
    adjusted["aipm3"] = deepcopy(base_a3)
    present = int(selection != "none")
    multiplier = MULTIPLIER if present else 1.0
    if present:
        index = base_index * multiplier
        adjusted["aipm2"]["reference_index"] = index
        adjusted["aipm2"]["norm_level"] = 0 if index < .83 else 1 if index < 1.17 else 2
        adjusted["aipm3"] = aipm3_score(
            adjusted["aipm1"], adjusted["aipm2"], adjusted["message_delivery"],
        )
    adjusted["celebrity_adjustment"] = {
        "version": VERSION,
        "source": "manual",
        "source_sha": result.get("source_sha"),
        "selection": selection,
        "present": present,
        "component": "aipm2",
        "multiplier": multiplier,
        "base_aipm2": deepcopy(base_a2),
        "base_aipm3": deepcopy(base_a3),
        # Share of the final index, deliberately separate from SHAP weights.
        "final_share_percent": 100.0 * (multiplier - 1.0) / multiplier,
    }
    return adjusted
