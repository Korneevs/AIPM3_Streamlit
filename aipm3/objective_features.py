"""Verbatim, independent extraction protocols from the local AIPM apps.

Retries repeat a failed request; they do not add a vote. No shared history,
persona, request token or rewritten prompt is added to AIPM1/AIPM2.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from .legacy_contracts import (
    AIPM1_ANALYSIS_PROMPT, AIPM1_FEATURES_JSON_SCHEMA, AIPM1_GEMINI_MODEL,
    AIPM1_MAX_BINARY_MB, AIPM1_N_ITERATIONS, AIPM1_PROD_FEATURES,
    AIPM2_BOOL_FEATURES, AIPM2_DECOMP_PROMPT, AIPM2_DECOMP_SCHEMA,
    AIPM2_ENUM_FEATURES, AIPM2_GEMINI_MODEL, AIPM2_INT_FEATURES,
    AIPM2_MAX_BINARY_MB, AIPM2_N_ITERATIONS,
)

BASE_URL = "https://litellm.data-light.ru/v1"
PROTOCOL_VERSION = "local-contracts-20260908"


def request_kwargs(component: str, video_base64: str) -> dict[str, Any]:
    """Exactly the kwargs passed to chat.completions.create by each local app."""
    if component == "aipm1":
        model, prompt, schema = (
            AIPM1_GEMINI_MODEL, AIPM1_ANALYSIS_PROMPT, AIPM1_FEATURES_JSON_SCHEMA,
        )
    elif component == "aipm2":
        model, prompt, schema = (
            AIPM2_GEMINI_MODEL, AIPM2_DECOMP_PROMPT.format(brand="Avito"),
            AIPM2_DECOMP_SCHEMA,
        )
    else:
        raise ValueError(f"Unknown component: {component}")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:video/mp4;base64,{video_base64}",
            }},
        ]}],
        "temperature": 0.0,
        "response_format": schema,
    }


def prepare_legacy_video(source: Path, output_dir: Path, component: str) -> Path:
    """Keep each local app's size gate and FFmpeg arguments, not MD's encoding.

    Safe private output names replace local AIPM1's .replace('.mp4', ...),
    which could otherwise overwrite a MOV input. Media settings are unchanged.
    """
    if component not in {"aipm1", "aipm2"}:
        raise ValueError(f"Unknown component: {component}")
    max_mb = AIPM1_MAX_BINARY_MB if component == "aipm1" else AIPM2_MAX_BINARY_MB
    within_limit = source.stat().st_size / 1024 / 1024 <= max_mb
    if within_limit and (component == "aipm1" or source.suffix.lower() == ".mp4"):
        return source
    # Both local AIPM apps prefer bundled FFmpeg over the system binary.
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"{component}: FFmpeg is required to prepare this video")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "prepared.mp4"
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=180)
    return output


def aggregate_aipm1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("AIPM1: no successful iterations")
    result = {
        field: Counter([int(str(row[field]).strip()) if field == "message_focus_seconds"
                        else int(row[field]) for row in rows]).most_common(1)[0][0]
        for field in AIPM1_PROD_FEATURES if field != "unique_offer_count"
    }
    result["unique_offer_count"] = int(round(np.mean([
        int(row["unique_offer_count"]) for row in rows
    ])))
    transcripts = [row["voiceover_text"] for row in rows if row.get("voiceover_text")]
    result["voiceover_text"] = max(transcripts, key=len) if transcripts else "Не удалось извлечь"
    return result


def aggregate_aipm2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("AIPM2: no successful iterations")
    out = {}
    for field in AIPM2_INT_FEATURES:
        out[field] = int(round(np.mean([int(row[field]) for row in rows])))
    for field in AIPM2_BOOL_FEATURES:
        out[field] = bool(Counter([bool(row[field]) for row in rows]).most_common(1)[0][0])
    for field in AIPM2_ENUM_FEATURES:
        out[field] = Counter([str(row[field]) for row in rows]).most_common(1)[0][0]
    return out


def extract_component(
    *, component: str, video_base64: str, api_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Serial iterations, as in the local apps: first vote wins a mode tie.
    kwargs = request_kwargs(component, video_base64)
    iterations = AIPM1_N_ITERATIONS if component == "aipm1" else AIPM2_N_ITERATIONS
    required = kwargs["response_format"]["json_schema"]["schema"]["required"]
    rows = []
    with OpenAI(api_key=api_key, base_url=BASE_URL, timeout=300, max_retries=0) as client:
        for iteration in range(iterations):
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(**kwargs)
                    row = json.loads(response.choices[0].message.content)
                    if not isinstance(row, dict) or any(field not in row for field in required):
                        raise ValueError("Incomplete feature response")
                    rows.append(row)
                    break
                except Exception as exc:
                    if attempt == 2:
                        # Never silently score a partial panel as a complete analysis.
                        raise RuntimeError(
                            f"{component}: iteration {iteration + 1}/{iterations} failed"
                        ) from exc
                    time.sleep(20 if "429" in str(exc) else 5)
    aggregate = aggregate_aipm1(rows) if component == "aipm1" else aggregate_aipm2(rows)
    return aggregate, rows
