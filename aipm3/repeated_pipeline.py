"""Ten complete frozen-model predictions, averaged only after inference.

Each repeat owns its legacy/MD cache. Resuming a repeat may reuse that repeat's
finished MD requests; it never borrows another repeat or the old single cache.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading

from .legacy_contracts import AIPM1_N_ITERATIONS, AIPM2_N_ITERATIONS
from .manual_celebrity import apply_celebrity
from .models import AIPM2_CAT_FEATURES, AIPM3_Q33, AIPM3_Q67, SCORING_VERSION
from .objective_features import PROTOCOL_VERSION
from .pipeline import run_analysis
from .runtime_resources import AnalysisBusy, file_sha256
from .vertical_uvp import evaluate_uvp, summarize_alignment

REPEAT_COUNT = 10
REPETITION_VERSION = "full-repeat-mean10-v1"
_LOCK = threading.Lock()
_COMPONENTS = ("aipm1", "aipm2", "message_delivery")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        stream.write(_json(value))
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value):
    if isinstance(value, bool):
        raise ValueError("A score must be numeric, not boolean")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite repeated score")
    return value


def _mean(values):
    values = [_number(v) for v in values]
    return math.fsum(values) / len(values)


def _validate_child(result, source_sha, model_sha256):
    if (not isinstance(result, dict) or result.get("source_sha") != source_sha
            or result.get("model_sha256") != model_sha256
            or result.get("protocol_version") != PROTOCOL_VERSION
            or result.get("scoring_version") != SCORING_VERSION
            or "repeat_results" in result):
        raise ValueError("Repeat does not match the source/model contract")
    _json(result)  # Reject NaN/Infinity anywhere, including saved feature effects.
    for name in _COMPONENTS:
        for key in ("reference_index", "percentile"):
            _number(result[name][key])
        for key in ("feature_values", "feature_effects"):
            if not isinstance(result[name][key], dict):
                raise ValueError("Incomplete repeat features")
    _number(result["aipm3"]["index"])
    for name, count in (("aipm1", AIPM1_N_ITERATIONS), ("aipm2", AIPM2_N_ITERATIONS)):
        if len(result.get("objective_runs", {}).get(name, [])) != count:
            raise ValueError("Incomplete objective extraction")
    for key, count in (("diagnostic_panel", 30), ("transcripts", 3),
                       ("diagnostic_recovery", 84), ("blind_answers", 30)):
        if len(result.get(key, [])) != count:
            raise ValueError("Incomplete repeat: " + key)


def _mapping_mean(rows):
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Repeats contain different feature sets")
    return {key: _mean([row[key] for row in rows]) for key in sorted(keys)}


def _observed_values(rows, categorical=()):
    """Display summaries only; these values are never submitted to a model."""
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Repeats contain different observation sets")
    result = {}
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if key in categorical or any(isinstance(v, (bool, str)) for v in values):
            result[key] = Counter(values).most_common(1)[0][0]
        else:
            result[key] = _mean(values)
    return result


def _components(results):
    result = {}
    for name in _COMPONENTS:
        rows = [r[name] for r in results]
        out = deepcopy(rows[0])
        # Classes are categorical: retain their distribution, never average them.
        out.pop("raw_class", None)
        if all("raw_class" in row for row in rows):
            out["class_counts"] = dict(Counter(str(row["raw_class"]) for row in rows))
        for key in ("reference_index", "percentile", "raw_score", "reference_score", "creative_score"):
            if all(key in row for row in rows):
                out[key] = _mean([row[key] for row in rows])
        if all("probability" in row for row in rows):
            out["probability"] = [_mean([row["probability"][i] for row in rows])
                                  for i in range(len(rows[0]["probability"]))]
        out["feature_effects"] = _mapping_mean([row["feature_effects"] for row in rows])
        categorical = (set(rows[0]["feature_values"]) - {"unique_offer_count"}
                       if name == "aipm1" else set(AIPM2_CAT_FEATURES) if name == "aipm2" else set())
        out["feature_values"] = _observed_values([row["feature_values"] for row in rows], categorical)
        percentile = out["percentile"]
        out["norm_level"] = 0 if percentile < 33 else 1 if percentile < 67 else 2
        if name == "aipm2" and any(r.get("celebrity_adjustment", {}).get("present") for r in results):
            index = out["reference_index"]
            out["norm_level"] = 0 if index < .83 else 1 if index < 1.17 else 2
        result[name] = out
    index = _mean([r["aipm3"]["index"] for r in results])
    level = 0 if index < AIPM3_Q33 else 1 if index < AIPM3_Q67 else 2
    result["aipm3"] = {
        "index": index, "index_100": index * 100, "level": level,
        "label": ("Ниже среднего", "Средний уровень", "Выше среднего")[level],
        "component_indices": {name: result[name]["reference_index"] for name in _COMPONENTS},
    }
    return result


def aggregate_repeats(results, celebrity="none"):
    if len(results) != REPEAT_COUNT:
        raise ValueError("Для результата нужны все 10 полных прогонов")
    source_sha, model_sha256 = results[0]["source_sha"], results[0]["model_sha256"]
    for result in results:
        _validate_child(result, source_sha, model_sha256)
    # The frozen per-repeat product is computed before averaging, including the
    # existing celebrity policy. Product(mean components) is intentionally unused.
    effective = [apply_celebrity(result, celebrity) for result in results]
    aggregated = _components(effective)
    aggregated.update({key: deepcopy(results[0][key]) for key in
                       ("source_sha", "model_sha256", "protocol_version", "scoring_version")})
    aggregated.update(repetition_version=REPETITION_VERSION, repeat_count=REPEAT_COUNT,
                      aggregation="arithmetic_mean_of_per_repeat_outputs",
                      repeat_results=deepcopy(results),
                      repeat_scores=[{"repeat": number, "celebrity": celebrity,
                                      **{name: row[name]["reference_index"] for name in _COMPONENTS},
                                      "aipm3": row["aipm3"]["index"]}
                                     for number, row in enumerate(effective, 1)],
                      main_idea=Counter(str(r.get("main_idea", "")) for r in results).most_common(1)[0][0],
                      main_ideas=[str(r.get("main_idea", "")) for r in results])
    if all("duration_seconds" in r for r in results):
        aggregated["duration_seconds"] = _mean([r["duration_seconds"] for r in results])
    aggregated["message_delivery_business"] = _observed_values(
        [r["message_delivery_business"] for r in results], {"audio_completeness", "cta_clarity"})
    aggregated["recovery_curve"] = _mapping_mean([r["recovery_curve"] for r in results])
    for key in ("diagnostic_panel", "diagnostic_recovery", "blind_answers"):
        aggregated[key] = []
        for repeat, result in enumerate(results, 1):
            for item in result[key]:
                row = {**deepcopy(item), "repeat": repeat}
                for identifier in ("respondent_id", "respondent_uid"):
                    if identifier in row:
                        row[identifier] = f"repeat_{repeat:02d}:{row[identifier]}"
                aggregated[key].append(row)
    base = _components([apply_celebrity(r, "none") for r in results])
    aggregated["celebrity_adjustment"] = {
        **deepcopy(effective[0]["celebrity_adjustment"]),
        "base_aipm2": base["aipm2"], "base_aipm3": base["aipm3"],
        "applied_per_repeat": True,
    }
    return aggregated


def is_repeated_result(result):
    if (not isinstance(result, dict) or result.get("repetition_version") != REPETITION_VERSION
            or result.get("repeat_count") != REPEAT_COUNT
            or not isinstance(result.get("repeat_results"), list)
            or len(result.get("repeat_results", [])) != REPEAT_COUNT):
        return False
    try:
        for child in result["repeat_results"]:
            _validate_child(child, result["source_sha"], result["model_sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def apply_repeated_celebrity(result, selection):
    if not is_repeated_result(result):
        raise ValueError("Сохранённый результат не содержит 10 полных прогонов")
    adjusted = aggregate_repeats(result["repeat_results"], celebrity=selection)
    for key in ("vertical_uvp", "repeat_cache"):
        if key in result:
            adjusted[key] = deepcopy(result[key])
    return adjusted


def evaluate_repeated_uvp(result, target, api_key, cache_root):
    if not is_repeated_result(result):
        raise ValueError("UVP requires ten complete repeats")
    assessments, answers = [], []
    # Keep the existing <=100-answer validator and each 30-person cache intact.
    for repeat, child in enumerate(result["repeat_results"], 1):
        assessment = evaluate_uvp(child, target, api_key, cache_root)
        if assessment.get("total") != 30:
            raise ValueError("Incomplete UVP repeat")
        assessments.append(assessment)
        answers.extend({**row, "respondent_id": f"repeat_{repeat:02d}:{row['respondent_id']}"}
                       for row in assessment["answers"])
    return {**summarize_alignment(answers), "target": deepcopy(target),
            "source_sha": result["source_sha"], "repeat_count": REPEAT_COUNT,
            "repeat_assessments": assessments}


def run_repeated_analysis(*, source_video, output_root, api_key, models, progress=None,
                          celebrity="none"):
    if not _LOCK.acquire(blocking=False):
        raise AnalysisBusy("Сервер уже анализирует другой ролик. Дождитесь завершения и повторите запуск.")
    try:
        source_video, output_root = Path(source_video), Path(output_root)
        source_sha = file_sha256(source_video)
        contract = dict(version=REPETITION_VERSION, count=REPEAT_COUNT, source_sha=source_sha,
                        source_suffix=source_video.suffix.lower(), protocol_version=PROTOCOL_VERSION,
                        scoring_version=SCORING_VERSION, model_sha256=models.artifact_sha256)
        fingerprint = hashlib.sha256(_json(contract).encode()).hexdigest()
        root = output_root / REPETITION_VERSION / fingerprint
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "contract.json", contract)
        # A stable source path also makes prepared-media recovery resumable after
        # Streamlit creates a different temporary upload filename on the next click.
        source = root / ("source" + source_video.suffix.lower())
        if not source.exists() or file_sha256(source) != source_sha:
            shutil.copyfile(source_video, source)
        if file_sha256(source) != source_sha:
            raise ValueError("Source video changed during cache preparation")
        results, paths = [], []
        for repeat in range(1, REPEAT_COUNT + 1):
            repeat_root = root / f"repeat_{repeat:02d}"
            complete = repeat_root / "complete.json"
            result = None
            if complete.exists():
                try:
                    saved = json.loads(complete.read_text(encoding="utf-8"))
                    if saved["contract"] != contract or saved["repeat"] != repeat:
                        raise ValueError("Different repeat cache contract")
                    if hashlib.sha256(_json(saved["result"]).encode()).hexdigest() != saved["result_sha256"]:
                        raise ValueError("Damaged repeat result")
                    _validate_child(saved["result"], source_sha, models.artifact_sha256)
                    result = saved["result"]
                except (KeyError, TypeError, ValueError):
                    result = None
            if progress:
                progress(f"Прогон {repeat}/{REPEAT_COUNT}" + (" · сохранён" if result else ""))
            if result is None:
                # MD's fresh=False is safe here: all request caches are local to
                # this repeat. Successful partial MD requests survive retries.
                try:
                    result = run_analysis(source_video=source, output_root=repeat_root,
                                          api_key=api_key, models=models,
                                          progress=(lambda message, n=repeat: progress(
                                              f"Прогон {n}/{REPEAT_COUNT} · {message}")) if progress else None)
                except AnalysisBusy:
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Не завершён прогон {repeat}/{REPEAT_COUNT}. Итоговая оценка не рассчитана; "
                        "готовые прогоны сохранены. Повторите запуск для продолжения."
                    ) from exc
                _validate_child(result, source_sha, models.artifact_sha256)
                _atomic_json(complete, dict(contract=contract, repeat=repeat, result=result,
                                           result_sha256=hashlib.sha256(_json(result).encode()).hexdigest()))
            results.append(result)
            paths.append(str(complete))
        aggregate = aggregate_repeats(results, celebrity=celebrity)
        aggregate["repeat_cache"] = paths
        _atomic_json(root / "result.json", aggregate)
        if progress:
            progress("Завершены все 10 прогонов; оценки усреднены")
        return aggregate
    finally:
        _LOCK.release()
