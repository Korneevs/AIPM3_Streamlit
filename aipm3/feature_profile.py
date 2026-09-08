"""Fixed higher-is-better scores for the original creative feature groups.

Presentation-only transform: no extraction, fitting, prediction changes or API.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

from .interpretation import AIPM1_GROUPS, AIPM2_GROUPS, MD_GROUPS
from .models import EXPECTED_ARTIFACT_SHA256, SCORING_VERSION

VERSION = "creative-group-scores-v2"
GROUPS = {
    "aipm1": ("Заметность", AIPM1_GROUPS),
    "aipm2": ("Запоминаемость", {
        name: features for name, features in AIPM2_GROUPS.items()
        if name != "Контекст бренда"
    }),
    "message_delivery": ("Считываемость", MD_GROUPS),
}


def score_effect(effect: float, scale: float) -> float:
    """Increasing signed effect means a nondecreasing score; zero is exactly 50.

    Each component has ONE reference-output scale shared by all its groups,
    so tiny-effect groups are not inflated to fill the display range.
    """
    effect, scale = float(effect), float(scale)
    if not math.isfinite(effect) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite effect and positive fixed scale are required")
    return 50.0 + 50.0 * math.tanh(effect / scale)


@lru_cache(maxsize=1)
def calibration_manifest():
    data = json.loads(Path(__file__).with_name("group_calibration.json").read_text())
    if (data.get("profile_version") != VERSION
            or data.get("model_sha256") != EXPECTED_ARTIFACT_SHA256
            or data.get("scoring_version") != SCORING_VERSION):
        raise ValueError("Group calibration does not match the frozen models")
    for component in GROUPS:
        score_effect(0, data["components"][component]["scale"])
    return data


def build_profile(result, calibration=None):
    calibration = calibration_manifest() if calibration is None else calibration
    if (calibration.get("profile_version") != VERSION
            or calibration.get("model_sha256") != EXPECTED_ARTIFACT_SHA256
            or calibration.get("scoring_version") != SCORING_VERSION):
        raise ValueError("Group calibration does not match the frozen models")
    compatible = (result.get("model_sha256") == EXPECTED_ARTIFACT_SHA256
                  and result.get("scoring_version") == SCORING_VERSION)
    rows = []
    for component, (pillar, groups) in GROUPS.items():
        component_data = result.get(component, {})
        effects = component_data.get("feature_effects", {}) if isinstance(component_data, dict) else {}
        effects = effects if isinstance(effects, dict) else {}
        scale = calibration["components"][component]["scale"]
        score_effect(0, scale)
        for label, features in groups.items():
            row = {"component": component, "pillar": pillar, "label": label,
                   "features": list(features), "score": None, "effect": None}
            try:
                if not compatible:
                    raise ValueError("Model version mismatch")
                # Missing effects must never become zero or a neutral score.
                effect = math.fsum(float(effects[name]) for name in features)
                value = score_effect(effect, scale)
            except (KeyError, TypeError, ValueError, OverflowError):
                row["reason"] = "Недостаточно данных этой версии модели"
            else:
                row.update(score=value, display_score=int(round(value)), effect=effect)
            rows.append(row)
    return {"version": VERSION, "groups": rows,
            "calibration_id": calibration["calibration_id"],
            "meaning": "Model-assessed group support, not a probability or a causal effect",
            "validation": calibration.get("validation", {})}
