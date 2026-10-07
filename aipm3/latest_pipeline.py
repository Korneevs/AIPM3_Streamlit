"""Rerunnable exact-protocol extraction for the accepted 9–9–7 model.

Every successful request has its own immutable repeat/stage cache. Inference
uses the configured complete repeats; a retry is not another observation.
Paid calls are disabled unless the caller explicitly passes allow_live=True.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np

from . import latest_contracts as fresh_contract
from . import neuromatics_contracts as neuro_contract
from . import message_delivery_runtime as md
from .latest_physical import physical_updates
from .latest_runtime import (ANALYSIS_REPEATS, PROTOCOL_VERSION, clean_json, scoring_version_for,
                             validate_repeat_count,
                             family_for_video, rows_from_measurements, score_feature_rows)
from .objective_features import (aggregate_aipm1, aggregate_aipm2,
                                  prepare_legacy_video, request_kwargs)
from .runtime_resources import analysis_slot, file_sha256, run_video_command


class MissingMeasurement(RuntimeError):
    pass


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=".stage-", suffix=".json") as stream:
        json.dump(clean_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(path)


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _request(kwargs: dict, path: Path, *, video_sha: str, api_key: str,
             allow_live: bool, validate: Callable[[dict], None] | None = None):
    # The prompt/schema/model/temperature and exact prepared media determine
    # compatibility. The repeat directory supplies independence between runs.
    compact = {**kwargs, "messages": [{**message, "content": [
        {"video_sha256": video_sha} if block.get("type") == "image_url" else block
        for block in message["content"]]} for message in kwargs["messages"]]}
    contract = _fingerprint(compact)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("request_fingerprint") != contract:
            raise ValueError(f"Extraction cache contract mismatch: {path.name}")
        payload = record["response"]
        if validate:
            validate(payload)
        return payload
    if not allow_live:
        raise MissingMeasurement(f"Missing cached observation: {path}. Live calls are disabled.")
    if not api_key:
        raise ValueError("An API key is required for a new measurement")
    from openai import OpenAI
    for attempt in range(3):
        try:
            started = time.monotonic()
            with OpenAI(api_key=api_key, base_url=md.BASE_URL, timeout=300, max_retries=0) as client:
                response = client.chat.completions.create(**kwargs)
            payload = json.loads(response.choices[0].message.content)
            if validate:
                validate(payload)
            _write_json(path, dict(
                response=payload, raw_response=response.model_dump(mode="json"),
                request_fingerprint=contract, request_contract=compact,
                prepared_sha=video_sha,
                captured_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - started,
            ))
            return payload
        except Exception as exc:
            _write_json(path.with_name(path.stem + f".failure_{attempt + 1}.json"),
                        dict(error_type=type(exc).__name__, status=getattr(exc, "status_code", None)))
            if attempt == 2:
                raise RuntimeError(f"Measurement failed: {path.name} ({type(exc).__name__})") from None
            time.sleep(20 if getattr(exc, "status_code", None) == 429 else 5)


def _objective(component: str, encoded: str, prepared_sha: str, folder: Path,
               api_key: str, allow_live: bool, *, material_kind: str = "finished"):
    kwargs = request_kwargs(component, encoded)
    if material_kind == "neuromatics" and component == "aipm2":
        kwargs["messages"][0]["content"][0]["text"] = (
            neuro_contract.NEUROMATICS_AIPM2_PROMPT.format(brand="Avito")
            + neuro_contract.NEUROMATICS_AIPM2_SUPPLEMENT
        )
    required = kwargs["response_format"]["json_schema"]["schema"]["required"]

    def validate(payload):
        if not isinstance(payload, dict) or any(k not in payload for k in required):
            raise ValueError("Incomplete objective feature response")

    count = 3 if component == "aipm1" else 2
    runs = [_request(kwargs, folder / f"call_{number:02d}.json", video_sha=prepared_sha,
                     api_key=api_key, allow_live=allow_live, validate=validate)
            for number in range(1, count + 1)]
    aggregate = aggregate_aipm1 if component == "aipm1" else aggregate_aipm2
    return aggregate(runs), runs


def _panel(encoded: str, prepared_sha: str, folder: Path, api_key: str, allow_live: bool,
           *, material_kind: str = "finished"):
    """Keep panel30 intact; append the supplied semantics only for neuromatics."""
    def one(call_id):
        personas = md.PANEL30_PERSONAS[(call_id - 1) * 3:call_id * 3]
        task_key = f"panel30_{call_id:02d}"
        token = hashlib.sha256(f"md-final-v1|{prepared_sha}|{task_key}".encode()).hexdigest()[:16]
        prompt = md.panel_prompt(personas) + f"\n\nREQUEST_TOKEN: {token}\nВерни request_token дословно."
        if material_kind == "neuromatics":
            prompt += "\n" + neuro_contract.NEUROMATICS_SEMANTICS
        kwargs = dict(model=md.MODEL_NAME, temperature=0.2, response_format=md.panel_schema(),
                      messages=[{"role": "user", "content": [
                          {"type": "text", "text": prompt},
                          {"type": "image_url", "image_url": {"url": "data:video/mp4;base64," + encoded}}]}])

        def validate(payload):
            returned = str(payload.get("request_token", "")).strip()
            prefix = len(returned) >= 12 and (returned.startswith(token) or token.startswith(returned))
            if not (returned == token or prefix):
                raise ValueError("request_token mismatch")
            ids = [str(a["respondent_id"]) for a in payload["answers"]]
            if len(ids) != 3 or set(ids) != {key for key, _ in personas}:
                raise ValueError("Panel IDs mismatch")
            for answer in payload["answers"]:
                for name, _, _, _, _ in md.ROUND4_FEATURES:
                    if name not in answer:
                        raise ValueError("Incomplete panel feature response")

        payload = _request(kwargs, folder / f"call_{call_id:02d}.json", video_sha=prepared_sha,
                           api_key=api_key, allow_live=allow_live, validate=validate)
        answers = {str(a["respondent_id"]): a for a in payload["answers"]}
        rows = []
        for respondent_id, description in personas:
            answer = answers[respondent_id]
            row = dict(call_id=call_id, respondent_id=respondent_id, persona=description,
                       main_message_summary=str(answer["main_message_summary"]).strip())
            for name, kind, low, high, _ in md.ROUND4_FEATURES:
                value = answer[name]
                row[name] = int(bool(value)) if kind == "boolean" else int(np.clip(value, low, high))
            rows.append(row)
        return rows

    # Bound memory the same way as the existing cloud pipeline.
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(one, range(1, 11)))
    return sorted([row for group in groups for row in group],
                  key=lambda row: (row["call_id"], row["respondent_id"]))


def _fresh(encoded: str, prepared_sha: str, path: Path, api_key: str, allow_live: bool):
    fields = fresh_contract.FIELDS
    properties = {k: {"type": "integer", "minimum": 0, "maximum": 3} for k in fields}
    properties.update(main_claim={"type": "string"}, evidence={
        "type": "object", "additionalProperties": False,
        "properties": {k: {"type": "string"} for k in fields}, "required": fields})
    schema = {"type": "json_schema", "json_schema": {
        "name": "three_component_replacements_v1", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": properties, "required": [*fields, "main_claim", "evidence"]}}}
    kwargs = dict(model=fresh_contract.MODEL, temperature=0.0, response_format=schema,
                  messages=[{"role": "user", "content": [
                      {"type": "text", "text": fresh_contract.PROMPT.format(brand="Avito")},
                      {"type": "image_url", "image_url": {"url": "data:video/mp4;base64," + encoded}}]}])

    def validate(payload):
        if not all(type(payload.get(k)) is int and 0 <= payload[k] <= 3 for k in fields):
            raise ValueError("Invalid fresh feature scale")
        if not isinstance(payload.get("evidence"), dict) or any(k not in payload["evidence"] for k in fields):
            raise ValueError("Missing fresh feature evidence")

    return _request(kwargs, path, video_sha=prepared_sha, api_key=api_key,
                    allow_live=allow_live, validate=validate)


def _physical(source: Path, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    import imageio_ffmpeg
    # The accepted feature matrix defines physical duration as frame count / 2
    # after fps=2 sampling. Preserve that half-second grid for close-up shares.
    completed = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(source), "-an",
        "-vf", "fps=2,scale=160:90", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True, timeout=180)
    duration = len(completed.stdout) / (160 * 90 * 3) / 2
    values = {**physical_updates(source), "phys__duration": duration}
    _write_json(cache, values)
    return values


def _prepare_inputs(source: Path, root: Path) -> dict[str, Path]:
    """Reuse the exact media bytes on resume, including a renamed upload."""
    manifest_path = root / "prepared" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        paths = {key: source if item["path"] == "source" else root / item["path"]
                 for key, item in manifest.items()}
        if set(paths) != {"aipm1", "aipm2", "panel", "fresh"}:
            raise ValueError("Incomplete prepared-media manifest")
        for key, path in paths.items():
            if not path.is_file() or file_sha256(path) != manifest[key]["sha256"]:
                raise ValueError(f"Prepared video changed or is missing: {key}")
        return paths
    prepared = {
        key: prepare_legacy_video(source, root / "prepared" / key, key)
        for key in ("aipm1", "aipm2")}
    prepared["panel"] = md.prepare_video(source, root / "prepared" / "panel")
    fresh_path = root / "prepared" / "fresh" / "video.mp4"
    if not fresh_path.exists():
        import imageio_ffmpeg
        fresh_path.parent.mkdir(parents=True, exist_ok=True)
        run_video_command([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(source),
            "-vf", "scale='min(640,iw)':-2", "-r", "24", "-c:v", "libx264", "-crf", "27",
            "-preset", "fast", "-threads", "1", "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart", str(fresh_path)])
    prepared["fresh"] = fresh_path
    _write_json(manifest_path, {
        key: {"path": "source" if path.resolve() == source.resolve() else str(path.relative_to(root)),
              "sha256": file_sha256(path)} for key, path in prepared.items()})
    return prepared


def run_latest_analysis(*, source_video: Path, output_root: Path, api_key: str = "",
                        brand: str = "Avito", vertical: str = "Goods", family: str | None = None,
                        record: str | None = None, allow_live: bool = False,
                        progress: Callable[[str], None] | None = None,
                        evidence_collector: Callable | None = None,
                        material_kind: str = "finished", repeat_count: int = ANALYSIS_REPEATS):
    """Resume completed stages and score the requested number of full runs."""
    version = scoring_version_for(material_kind)
    validate_repeat_count(repeat_count)
    if brand != "Avito":
        raise ValueError("This exact application protocol is calibrated for Avito uploads")
    source_video = Path(source_video).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    with analysis_slot():
        sha = file_sha256(source_video)
        family = family or family_for_video(sha)
        root = Path(output_root).resolve() / PROTOCOL_VERSION
        if material_kind == "neuromatics":
            # A new measurement contract must not reuse observations from
            # either finished creatives or the previous neuromatics prompts.
            root = root / neuro_contract.NEUROMATICS_PROTOCOL
        root = root / sha
        root.mkdir(parents=True, exist_ok=True)
        physical = _physical(source_video, root / "physical.json")
        prepared = _prepare_inputs(source_video, root)
        if material_kind == "neuromatics":
            prepared["fresh"] = prepared["panel"]
        prepared_hash = {key: file_sha256(path) for key, path in prepared.items()}
        measurements = []
        for repeat in range(1, repeat_count + 1):
            if progress:
                progress(f"Повтор {repeat}/{repeat_count}: проверяем свойства ролика")
            folder = root / f"repeat_{repeat:02d}"
            objective, objective_runs = {}, {}
            for kind in ("aipm1", "aipm2"):
                encoded = base64.b64encode(prepared[kind].read_bytes()).decode()
                objective[kind], objective_runs[kind] = _objective(
                    kind, encoded, prepared_hash[kind], folder / kind, api_key, allow_live,
                    material_kind=material_kind)
                del encoded
            encoded = base64.b64encode(prepared["panel"].read_bytes()).decode()
            panel = _panel(encoded, prepared_hash["panel"], folder / "panel", api_key, allow_live,
                           material_kind=material_kind)
            del encoded
            encoded = base64.b64encode(prepared["fresh"].read_bytes()).decode()
            fresh = _fresh(encoded, prepared_hash["fresh"], folder / "fresh.json", api_key, allow_live)
            del encoded
            measured = dict(repeat=repeat, source_sha=sha, objective_features=objective,
                            objective_runs=objective_runs, diagnostic_panel=panel, fresh=fresh)
            _write_json(folder / "measurements.json", measured)
            measurements.append(measured)
        if file_sha256(source_video) != sha:
            raise ValueError("Source video changed during analysis")
        rows = rows_from_measurements(measurements, source_sha=sha, physical=physical,
                                      duration=physical["phys__duration"], brand=brand,
                                      vertical=vertical, family=family, record=record,
                                      material_kind=material_kind, repeat_count=repeat_count)
        result = score_feature_rows(rows, metadata=dict(
            source_sha=sha, source_name=source_video.name, brand=brand, vertical=vertical,
            family=family, duration_seconds=md.video_duration(source_video),
            prepared_video_sha256=prepared_hash,
            evidence=dict(fresh=[m["fresh"] for m in measurements],
                          panel=[m["diagnostic_panel"] for m in measurements],
                          objective_runs={k: [m["objective_runs"][k] for m in measurements]
                                          for k in ("aipm1", "aipm2")}),
            extraction=dict(repeats=repeat_count, calls_per_repeat=16, independent_stage_caches=True,
                            transcript_and_recovery_used=False),
        ), material_kind=material_kind, repeat_count=repeat_count)
        identity = _fingerprint(dict(scoring=version, material_kind=material_kind, brand=brand, vertical=vertical,
                                    family=family, record=record, repeat_count=repeat_count))[:16]
        _write_json(root / f"result_{identity}.json", result)
        if evidence_collector is not None:
            try:
                result["independent_evidence"] = evidence_collector(
                    source_video=source_video, output_root=root / "independent_evidence",
                    api_key=api_key, allow_live=allow_live, progress=progress)
                result["evidence_status"] = "complete"
            except Exception as exc:
                # Keep completed scores usable when an optional review fails.
                result["independent_evidence"] = []
                result["evidence_status"] = "incomplete"
                result["evidence_error_type"] = type(exc).__name__
            _write_json(root / f"result_{identity}.json", result)
        return result
