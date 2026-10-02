"""Exact inference adapter for the separately accepted October 1 three-head model.

The existing Streamlit scorer is left intact. No training or inference calls to
an external service take place in this module.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent / "latest_bundle" / "20261001"
SCORING_VERSION = "three-heads-effect15-20261001-979fdba8b527-f0aa4e015216-6d75e5e23d76"
NEUROMATICS_SCORING_VERSION = "neuromatics-nclip-ridge100-20261002-c9e7282c127d"
PROTOCOL_VERSION = "latest-exact-inputs-20261001-v1"
MODEL_TASKS = ("n", "m", "r")
SCORE_NAMES = {"n": "noticeability", "m": "message_delivery", "r": "norm_ad_recall"}
MATERIAL_LABELS = {"finished": "AIPM3.0 (для готовых)",
                   "neuromatics": "AIPM3.0 (для нейроматиков)"}


def scoring_version_for(material_kind: str = "finished") -> str:
    if material_kind not in MATERIAL_LABELS:
        raise ValueError("Неизвестный тип материала.")
    return SCORING_VERSION if material_kind == "finished" else NEUROMATICS_SCORING_VERSION


def material_kind_for_result(result: dict) -> str:
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Некорректные метаданные результата.")
    kind = result.get("material_kind", metadata.get("material_kind", "finished"))
    expected = scoring_version_for(kind)
    if result.get("scoring_version") != expected:
        raise ValueError("Сохранённый результат рассчитан другой версией модели.")
    if metadata.get("material_kind", kind) != kind:
        raise ValueError("Тип материала в результате и метаданных не совпадает.")
    return kind


def artifact_hashes() -> dict[str, str]:
    """Verify the frozen package before loading trusted local estimators."""
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((BUNDLE_DIR / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Accepted model artifact mismatch: {relative}")
    return manifest


@lru_cache(maxsize=1)
def model_module():
    artifact_hashes()
    name = "_aipm_latest_accepted_20261001"
    spec = importlib.util.spec_from_file_location(name, BUNDLE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_latest_models():
    """Return the original AIPM3 runtime object, without refitting."""
    return model_module().AIPM3(BUNDLE_DIR / "models")


@lru_cache(maxsize=2)
def load_models(material_kind: str = "finished"):
    scoring_version_for(material_kind)
    if material_kind == "finished":
        return load_latest_models()
    from .neuromatics_models import NeuromaticsModels
    return NeuromaticsModels(load_latest_models(), model_module().coefficient)


def artifact_hashes_for(material_kind: str = "finished") -> dict[str, str]:
    scoring_version_for(material_kind)
    original = artifact_hashes()
    if material_kind == "finished":
        return original
    from .neuromatics_models import artifact_hashes as neuro_hashes
    return {**{"finished/" + k: v for k, v in original.items()},
            **{"neuromatics/" + k: v for k, v in neuro_hashes().items()}}


@lru_cache(maxsize=1)
def known_video_families() -> dict[str, tuple[str, ...]]:
    """Exact source hashes retain their frozen historical family exclusion."""
    artifact_hashes()
    families: dict[str, set[str]] = {}
    for task in MODEL_TASKS:
        frame = pd.read_csv(BUNDLE_DIR / "data" / f"fit_{task}.csv")
        if "sha" not in frame:
            continue
        for sha, family in frame[["sha", "family"]].dropna().itertuples(index=False, name=None):
            families.setdefault(str(sha), set()).add(str(family))
    return {sha: tuple(sorted(values)) for sha, values in families.items()}


def family_for_video(source_sha: str) -> str:
    matches = known_video_families().get(source_sha, ())
    if len(matches) > 1:
        raise ValueError("Source video maps to several historical families; specify family explicitly")
    return matches[0] if matches else "upload_" + source_sha


def clean_json(value):
    """Convert numpy values and unavailable measurements to portable JSON."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def score_feature_rows(feature_rows: dict[str, list[dict] | pd.DataFrame],
                       metadata: dict | None = None, model=None,
                       *, material_kind: str = "finished") -> dict[str, Any]:
    """Score one video's ten independent measurement repeats exactly.

    Input seconds remain seconds; the accepted model applies its own duration
    normalization, panel rounding and recall consensus once, during inference.
    ``family`` must identify an already known family if this is a reference
    video. For a new upload use a new family rather than borrowing another one.
    """
    version = scoring_version_for(material_kind)
    if metadata and metadata.get("material_kind", material_kind) != material_kind:
        raise ValueError("Метаданные относятся к другому типу материала.")
    model = model or load_models(material_kind)
    if any(task not in feature_rows for task in MODEL_TASKS):
        raise ValueError("All three component measurements are required")
    if any(not {"record", "repeat"}.issubset(pd.DataFrame(feature_rows[task]).columns)
           for task in MODEL_TASKS):
        raise ValueError("Every measurement needs record and repeat IDs")
    frames = {task: pd.DataFrame(feature_rows[task]).sort_values(
        ["record", "repeat"]).reset_index(drop=True) for task in MODEL_TASKS}
    if len({"sha" in frame for frame in frames.values()}) != 1:
        raise ValueError("Source hashes must be present for all components or none")
    keys = frames["n"][["record", "repeat"]]
    if len(keys) != 10 or keys.record.nunique() != 1:
        raise ValueError("Exactly one video with ten repeats is required")
    if keys.repeat.tolist() != list(range(1, 11)) or keys.duplicated().any():
        raise ValueError("Repeat IDs must be exactly 1 through 10")
    for task, frame in frames.items():
        if not keys.equals(frame[["record", "repeat"]]):
            raise ValueError("The three component measurement keys differ")
        for column in ("family", "brand", "vertical"):
            if column not in frame or frame[column].isna().any() or frame[column].nunique() != 1:
                raise ValueError(f"Missing or inconsistent metadata: {column}")
            if frame[column].iloc[0] != frames["n"][column].iloc[0]:
                raise ValueError(f"Component metadata differ: {column}")
        if "sha" in frame and frame.sha.nunique(dropna=False) != 1:
            raise ValueError("Measurements must belong to one source video")
        if "sha" in frames["n"] and ("sha" not in frame or not frame.sha.equals(frames["n"].sha)):
            raise ValueError("Component video hashes differ")
    if metadata and metadata.get("source_sha") and "sha" in frames["n"]:
        if metadata["source_sha"] != frames["n"].sha.iloc[0]:
            raise ValueError("Result source hash differs from feature rows")
    predictions = {task: model.heads[task].predict(frame)
                   for task, frame in frames.items()}
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("The frozen model produced a nonfinite prediction")
    per_repeat = keys.copy()
    for task, column in SCORE_NAMES.items():
        per_repeat[column] = predictions[task]
    per_repeat["OPM"] = predictions["n"] * predictions["m"]
    per_repeat["Q"] = per_repeat.OPM * predictions["r"]
    # Check against the frozen public entry point, including its exact average
    # of products (which is not a product of the three displayed averages).
    scored = model.score(frames["n"], frames["m"], frames["r"]).iloc[0]
    scores = {name: float(scored[name]) for name in [*SCORE_NAMES.values(), "OPM", "Q"]}
    for name, value in scores.items():
        if not np.isclose(value, per_repeat[name].mean(), rtol=1e-12, atol=1e-14):
            raise AssertionError("Frozen score aggregation changed")
    result = dict(metadata or {})
    result["metadata"] = {**dict(metadata or {}), "material_kind": material_kind}
    result.update(
        material_kind=material_kind,
        scoring_version=version,
        protocol_version=PROTOCOL_VERSION,
        model_sha256=artifact_hashes_for(material_kind),
        record=str(keys.record.iloc[0]),
        scores=scores,
        per_repeat=per_repeat.to_dict("records"),
        feature_rows={task: frame.to_dict("records") for task, frame in frames.items()},
        model_inputs={task: model.heads[task].state["columns"] for task in MODEL_TASKS},
    )
    return clean_json(result)


def validate_cached_result(result: dict, *, material_kind: str | None = None) -> dict:
    """Recompute a portable result with the same accepted model, offline."""
    actual_kind = material_kind_for_result(result)
    if material_kind is not None:
        scoring_version_for(material_kind)
        if actual_kind != material_kind:
            raise ValueError("Результат относится к другому типу материала. Откройте соответствующий раздел.")
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Сохранённый результат использует другой протокол измерений.")
    if result.get("model_sha256") != artifact_hashes_for(actual_kind):
        raise ValueError("Артефакты сохранённого результата не совпадают с текущей моделью.")
    reserved = {"metadata", "scores", "per_repeat", "feature_rows", "model_inputs",
                "scoring_version", "protocol_version", "model_sha256", "interpretation"}
    metadata = {k: v for k, v in result.items() if k not in reserved}
    recomputed = score_feature_rows(result["feature_rows"], metadata=metadata, material_kind=actual_kind)
    for name, actual in recomputed["scores"].items():
        if not np.isclose(result.get("scores", {}).get(name, np.nan), actual,
                          rtol=1e-12, atol=1e-14):
            raise ValueError("Сохранённые оценки не совпадают с признаками ролика.")
    return recomputed


def rows_from_measurements(measurements: list[dict], *, source_sha: str,
                           physical: dict[str, float], duration: float,
                           brand: str = "Avito", vertical: str = "Goods",
                           family: str | None = None, record: str | None = None,
                           material_kind: str = "finished"):
    """Convert the original observation protocols into canonical model inputs."""
    model = load_models(material_kind)
    if len(measurements) != 10:
        raise ValueError("Exactly ten complete measurements are required")
    family = family or family_for_video(source_sha)
    record = record or source_sha
    rows = {task: [] for task in MODEL_TASKS}
    for repeat, measurement in enumerate(measurements, start=1):
        if int(measurement.get("repeat", repeat)) != repeat:
            raise ValueError("Measurements are not in repeat order")
        if measurement.get("source_sha", source_sha) != source_sha:
            raise ValueError("Measurement belongs to another source video")
        objective = measurement["objective_features"]
        a1, a2 = objective["aipm1"], objective["aipm2"]
        panel = pd.DataFrame(measurement["diagnostic_panel"])
        if len(panel) != 30 or panel.respondent_id.nunique() != 30:
            raise ValueError("A complete original panel of 30 is required")
        fresh = measurement["fresh"]
        fresh = fresh.get("values", fresh)
        common = dict(record=record, family=family, brand=brand, vertical=vertical,
                      sha=source_sha, repeat=repeat, phys__duration=float(duration),
                      total_video_duration_sec=a2["total_video_duration_sec"])
        for task in MODEL_TASKS:
            row = common.copy()
            for feature in model.heads[task].state["columns"]:
                if feature == "brand_history":
                    continue  # Learned history is resolved by the frozen head.
                if feature.startswith("phys__"):
                    value = physical[feature]
                elif feature.startswith("panel__"):
                    column = feature.removeprefix("panel__")
                    value = pd.to_numeric(panel[column], errors="raise").mean()
                elif feature.startswith("fresh__"):
                    value = fresh[feature.removeprefix("fresh__")]
                elif feature in ("main_character", "promo"):
                    value = a1[feature]
                else:
                    value = a2[feature]
                if value is None or not np.isfinite(float(value)):
                    raise ValueError(f"Unavailable fresh observation: {feature}")
                row[feature] = float(value)
            if task == "n" and material_kind == "neuromatics":
                # One semantic input, measured by the original conservative A1
                # rubric. Preserve it beside raw A2 for audit, never overwrite.
                value = a1.get("state_transformation")
                if value is None or float(value) not in (0., 1.):
                    raise ValueError("Unavailable AIPM1 state transformation observation")
                row["state_transformation"] = float(value)
            rows[task].append(row)
    return rows
