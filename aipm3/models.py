from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


AIPM1_FEATURES = [
    "main_character",
    "promo",
    "humor",
    "state_transformation",
    "message_focus_seconds",
    "has_screen_offer_text",
    "unique_offer_count",
]
AIPM1_CAT_FEATURES = ["message_focus_seconds", "has_screen_offer_text"]

AIPM2_FEATURES = [
    "brand_mean_adrecall",
    "brand_logo_screen_seconds",
    "pack_shot_duration_seconds",
    "monologue_to_camera",
    "jingle_present",
    "ends_with_brand_logo_alone",
    "scene_pace_high",
    "vocal_song_present",
    "silence_or_music_only_seconds",
]
AIPM2_CAT_FEATURES = [
    "monologue_to_camera",
    "jingle_present",
    "ends_with_brand_logo_alone",
    "scene_pace_high",
    "vocal_song_present",
]
AVITO_MEAN_ADRECALL = 0.1358

AIPM1_INDEX_BY_CLASS = {
    0: 0.6938775510204082,
    1: 1.0510204081632653,
    2: 1.3571428571428572,
}

AIPM2_REFERENCE = np.asarray([
    0.0926, 0.1434, 0.0911, 0.1, 0.0736, 0.1311, 0.0662, 0.0961,
    0.0926, 0.1047, 0.2775, 0.1381, 0.1523, 0.0736, 0.1484, 0.0928,
    0.1267, 0.1064, 0.1487, 0.101, 0.0926, 0.1569, 0.1381, 0.1103,
    0.1, 0.0662, 0.1544, 0.0516, 0.1333, 0.187, 0.1055, 0.1125,
    0.0928, 0.1107, 0.0928, 0.099, 0.1132, 0.1, 0.1381, 0.0742,
    0.1055, 0.1042, 0.1, 0.1125, 0.1567, 0.1005, 0.0926,
], dtype=float)

MD_REFERENCE = np.asarray([
    2.0641868626803896, 1.0674670977647516, 1.3008370457028335,
    1.8103506585403355, 2.08824235285791, 1.167679020121804,
    1.3232528443435212, 1.1123770735718477, 1.5437942797858608,
    1.64841890782847, 1.461444517516201, 1.2056729246105506,
    1.5015494044246842, 1.5346072115449378, 0.7560103559229814,
    0.1815336375638023, 1.3545809595572162, 0.2408467528659426,
    0.97172428268922, 1.543864410864559, 1.1180145627911329,
    1.0427832133627783, 1.7260409484022046, 0.834575415441845,
    0.7647086265152185, 1.165440567526825, 1.7152941449147712,
    2.0161362838146486, 0.3107057939567305, 0.6604654000674689,
    0.0342460390695837, 1.559399900541424, 0.4838809219691839,
    -0.0538910080867202, 1.4576209535186253, 0.797595786997414,
    0.5672062727068332, 1.601045287309571, 1.19189872215031,
    1.5089583242176707, 0.6192729619073737, -0.0681612966894921,
    0.992489527986838, 0.7795467160771208, 0.2985187657528976,
    1.8048340134068088, 1.8931142473073477, 1.0751613954757155,
], dtype=float)

AIPM3_PRODUCT_MEAN = 0.975021850969246
AIPM3_Q33 = 0.7527663069338925
AIPM3_Q67 = 1.2729990980210948


@dataclass(frozen=True)
class FrozenModels:
    aipm1: CatBoostClassifier
    aipm2: CatBoostRegressor
    message_delivery_bundle: dict[str, Any]
    artifact_sha256: dict[str, str] = field(default_factory=dict)


def load_frozen_models(aipm1_path: str, aipm2_path: str, md_path: str) -> FrozenModels:
    aipm1 = CatBoostClassifier()
    aipm1.load_model(aipm1_path)
    aipm2 = CatBoostRegressor()
    aipm2.load_model(aipm2_path)
    md_bundle = joblib.load(md_path)
    for name, model, expected in [
        ("AIPM1", aipm1, AIPM1_FEATURES), ("AIPM2", aipm2, AIPM2_FEATURES),
    ]:
        if list(model.feature_names_) != expected:
            raise ValueError(f"{name}: модель в Secrets не соответствует выбранной версии признаков")
    hashes = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [("aipm1", aipm1_path), ("aipm2", aipm2_path), ("message_delivery", md_path)]
    }
    return FrozenModels(
        aipm1=aipm1, aipm2=aipm2, message_delivery_bundle=md_bundle,
        artifact_sha256=hashes,
    )


def _to_int(value: Any) -> int:
    return int(bool(value)) if isinstance(value, (bool, np.bool_)) else int(value)


def _catboost_multiclass_quality_shap(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    cat_features: list[str],
) -> np.ndarray:
    pool = Pool(frame, cat_features=cat_features)
    values = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=float)
    if values.ndim != 3:
        return np.zeros(frame.shape[1], dtype=float)
    # CatBoost: objects × classes × (features + expected value).
    if values.shape[0] == 1 and values.shape[1] >= 3:
        return values[0, 2, :-1] - values[0, 0, :-1]
    # Defensive support for classes × objects × features.
    if values.shape[0] >= 3 and values.shape[1] == 1:
        return values[2, 0, :-1] - values[0, 0, :-1]
    return np.zeros(frame.shape[1], dtype=float)


def score_aipm1(features: dict[str, Any], model: CatBoostClassifier) -> dict[str, Any]:
    row = {feature: _to_int(features[feature]) for feature in AIPM1_FEATURES}
    frame = pd.DataFrame([row], columns=AIPM1_FEATURES)
    for feature in AIPM1_CAT_FEATURES:
        frame[feature] = frame[feature].astype(int)
    probability = np.asarray(model.predict_proba(frame)[0], dtype=float)
    predicted_class = int(np.argmax(probability))
    creative_score = float(probability[1] * 0.5 + probability[2])
    shap_values = _catboost_multiclass_quality_shap(model, frame, AIPM1_CAT_FEATURES)
    return {
        "raw_class": predicted_class,
        "probability": probability.tolist(),
        "creative_score": creative_score,
        "reference_index": AIPM1_INDEX_BY_CLASS[predicted_class],
        "percentile": 100.0 * (AIPM1_INDEX_BY_CLASS[predicted_class] - 0.5),
        "feature_values": row,
        "feature_effects": dict(zip(AIPM1_FEATURES, shap_values.tolist())),
    }


def score_aipm2(features: dict[str, Any], model: CatBoostRegressor) -> dict[str, Any]:
    row: dict[str, Any] = {"brand_mean_adrecall": AVITO_MEAN_ADRECALL}
    for feature in AIPM2_FEATURES[1:]:
        row[feature] = _to_int(features[feature])
    frame = pd.DataFrame([row], columns=AIPM2_FEATURES)
    for feature in AIPM2_CAT_FEATURES:
        frame[feature] = frame[feature].astype(int)
    score = float(model.predict(frame)[0])
    pool = Pool(frame, cat_features=AIPM2_CAT_FEATURES)
    values = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=float)
    effects = values[0, :-1] if values.ndim == 2 else np.zeros(len(AIPM2_FEATURES))
    index = percentile_index(score, AIPM2_REFERENCE, denominator=49.0)
    return {
        "raw_score": score,
        "reference_index": index,
        "percentile": 100.0 * (index - 0.5),
        "feature_values": row,
        "feature_effects": dict(zip(AIPM2_FEATURES, effects.tolist())),
    }


def score_message_delivery(
    technical_features: dict[str, float],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    columns = list(bundle["feature_columns"])
    frame = pd.DataFrame([{feature: technical_features[feature] for feature in columns}])
    score = float(bundle["model"].predict(frame)[0])
    predicted_class = int(
        0 if score < bundle["threshold_t0"]
        else 2 if score >= bundle["threshold_t2"]
        else 1
    )
    values = np.asarray(shap.TreeExplainer(bundle["model"]).shap_values(frame), dtype=float)
    effects = values[0] if values.ndim == 2 else values.reshape(-1)[: len(columns)]
    index = percentile_index(score, MD_REFERENCE, denominator=49.0)
    return {
        "raw_score": score,
        "raw_class": predicted_class,
        "creative_score": float(np.clip(score / 2.0, 0.0, 1.0)),
        "reference_index": index,
        "percentile": 100.0 * (index - 0.5),
        "feature_values": dict(technical_features),
        "feature_effects": dict(zip(columns, effects.tolist())),
        "thresholds": {
            "class_0_below": float(bundle["threshold_t0"]),
            "class_2_from": float(bundle["threshold_t2"]),
        },
    }


def percentile_index(value: float, reference: np.ndarray, denominator: float) -> float:
    less = int(np.sum(reference < value))
    equal = int(np.sum(np.isclose(reference, value, rtol=0.0, atol=1e-12)))
    rank = less + ((equal + 1.0) / 2.0 if equal else 0.5)
    return float(np.clip(0.5 + rank / denominator, 0.5, 1.5))


def aipm3_score(aipm1: dict[str, Any], aipm2: dict[str, Any], md: dict[str, Any]) -> dict[str, Any]:
    product = float(
        aipm1["reference_index"]
        * aipm2["reference_index"]
        * md["reference_index"]
    )
    index = product / AIPM3_PRODUCT_MEAN
    if index < AIPM3_Q33:
        level, label = 0, "Ниже среднего"
    elif index < AIPM3_Q67:
        level, label = 1, "Средний уровень"
    else:
        level, label = 2, "Выше среднего"
    return {
        "index": float(index),
        "index_100": float(index * 100.0),
        "level": level,
        "label": label,
        "component_indices": {
            "aipm1": float(aipm1["reference_index"]),
            "aipm2": float(aipm2["reference_index"]),
            "message_delivery": float(md["reference_index"]),
        },
    }


def level_from_percentile(percentile: float) -> str:
    if percentile < 33.0:
        return "Низкий"
    if percentile < 67.0:
        return "Средний"
    return "Высокий"
