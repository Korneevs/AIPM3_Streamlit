from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import message_delivery_runtime as md_runtime
from .interpretation import build_interpretation
from .creative_review import manager_readout
from .models import SCORING_VERSION, FrozenModels, aipm3_score, score_aipm1, score_aipm2, score_message_delivery
from .objective_features import PROTOCOL_VERSION, extract_component, prepare_legacy_video


def _legacy_extract(component: str, source: Path, output_dir: Path, api_key: str):
    prepared = prepare_legacy_video(source, output_dir / component, component)
    encoded, video_sha = md_runtime.encode_video(prepared)
    features, runs = extract_component(
        component=component, video_base64=encoded, api_key=api_key,
    )
    return features, runs, video_sha


def _message_delivery_extract(
    prepared: Path,
    video_base64: str,
    video_sha: str,
    duration: float,
    output_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    md_runtime.configure_api_key(api_key)
    placeholder_secrets = output_dir / "unused-secrets.toml"
    with ThreadPoolExecutor(max_workers=3) as executor:
        panel_future = executor.submit(
            md_runtime.extract_panel30,
            video_base64,
            video_sha,
            output_dir / "panel30",
            placeholder_secrets,
            False,
        )
        transcript_future = executor.submit(
            md_runtime.extract_transcript,
            video_base64,
            video_sha,
            duration,
            output_dir / "transcript",
            placeholder_secrets,
            False,
        )
        recovery_future = executor.submit(
            md_runtime.extract_recovery,
            prepared,
            video_base64,
            video_sha,
            output_dir / "recovery",
            placeholder_secrets,
            False,
        )
        panel_frame, panel = panel_future.result()
        transcript_frame, words_per_second = transcript_future.result()
        recovery_frame, recovery = recovery_future.result()
    return {
        "panel_frame": panel_frame,
        "panel": panel,
        "transcript_frame": transcript_frame,
        "words_per_second": words_per_second,
        "recovery_frame": recovery_frame,
        "recovery": recovery,
    }


def run_analysis(
    *,
    source_video: Path,
    output_root: Path,
    api_key: str,
    models: FrozenModels,
) -> dict[str, Any]:
    source_hash = hashlib.sha256(source_video.read_bytes()).hexdigest()
    # Separate the restored protocol from cached results of the merged prompt.
    output_dir = output_root / PROTOCOL_VERSION / source_hash[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = md_runtime.prepare_video(source_video, output_dir / "prepared_media")
    video_base64, video_sha = md_runtime.encode_video(prepared)
    duration = md_runtime.video_duration(prepared)

    with ThreadPoolExecutor(max_workers=3) as executor:
        aipm1_future = executor.submit(
            _legacy_extract, "aipm1", source_video, output_dir, api_key,
        )
        aipm2_future = executor.submit(
            _legacy_extract, "aipm2", source_video, output_dir, api_key,
        )
        md_future = executor.submit(
            _message_delivery_extract,
            prepared,
            video_base64,
            video_sha,
            duration,
            output_dir / "message_delivery",
            api_key,
        )
        a1_features, a1_runs, a1_video_sha = aipm1_future.result()
        a2_features, a2_runs, a2_video_sha = aipm2_future.result()
        md_extracted = md_future.result()

    panel = md_extracted["panel"]
    recovery = md_extracted["recovery"]
    words_per_second = float(md_extracted["words_per_second"])
    technical = md_runtime.technical_features(
        panel,
        words_per_second,
        recovery,
        models.message_delivery_bundle,
    )

    aipm1 = score_aipm1(a1_features, models.aipm1)
    aipm2 = score_aipm2(a2_features, models.aipm2)
    message_delivery = score_message_delivery(
        technical,
        models.message_delivery_bundle,
    )
    aipm3 = aipm3_score(aipm1, aipm2, message_delivery)

    signed_train = np.asarray(
        models.message_delivery_bundle["signed_mean_train"], dtype=float
    )
    completeness_index = int(round(100.0 * float(np.mean(
        signed_train <= technical["eng__gemini_round4__signed_mean8"]
    ))))
    business_features = {
        "completeness_index": completeness_index,
        "message_specificity": float(panel["message_specificity_level__r4_mean3"]),
        "recovery_consistency": float(recovery["p12__cluster_valid_mask_consistency"]),
        "recovery_mask_gap": float(recovery["p12__cluster_valid_mask_mae_smoothed"]),
        "audio_completeness": int(panel["audio_only_message_completeness__r4_value"]),
        "words_per_second": words_per_second,
        "cta_clarity": bool(panel["cta_clarity__r4_value"]),
        "offer_condition_count": float(panel["offer_condition_count__r4_mean3"]),
        "semantic_agreement": float(panel["main_idea_entity_load__r4_agreement3"]),
    }
    interpretation = build_interpretation(
        aipm1,
        aipm2,
        message_delivery,
        business_features,
    )

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "scoring_version": SCORING_VERSION,
        "model_sha256": models.artifact_sha256,
        "component_video_sha": {
            "aipm1": a1_video_sha, "aipm2": a2_video_sha, "message_delivery": video_sha,
        },
        "video_sha": video_sha,
        "source_sha": source_hash,
        "duration_seconds": duration,
        "prepared_video": str(prepared),
        "objective_features": {"aipm1": a1_features, "aipm2": a2_features},
        "objective_runs": {"aipm1": a1_runs, "aipm2": a2_runs},
        "aipm1": aipm1,
        "aipm2": aipm2,
        "message_delivery": message_delivery,
        "aipm3": aipm3,
        "message_delivery_business": business_features,
        "main_idea": recovery["recovery_candidate"],
        "recovery_curve": recovery["condition_valid_posterior"],
        "interpretation": interpretation,
        "transcripts": md_extracted["transcript_frame"].to_dict(orient="records"),
        "blind_answers": [
            {"respondent_id": str(row["respondent_id"]), "answer": str(row["main_message_summary"])}
            for row in md_extracted["panel_frame"].to_dict(orient="records")
        ] if "panel_frame" in md_extracted else [],
        # Read-only diagnostic copy, taken AFTER prediction; never model inputs.
        "diagnostic_panel": md_extracted["panel_frame"].to_dict(orient="records")
        if "panel_frame" in md_extracted else [],
        "diagnostic_recovery": md_extracted["recovery_frame"].to_dict(orient="records")
        if "recovery_frame" in md_extracted else [],
    }
    result["manager_readout"] = manager_readout(result)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result
