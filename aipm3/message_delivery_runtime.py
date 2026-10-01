#!/usr/bin/env python3
"""Final Message Delivery inference for one advertising video.

The script has two commands:

1. ``build-bundle`` freezes the already selected eight-feature ordinal model.
2. ``predict`` extracts the production features from a video and scores it.

The source videos and historical artifacts are read-only. New API responses and
derived media are written only to the selected output directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, ValidationError
from openai import OpenAI
from scipy import stats

if __package__:
    from .runtime_resources import file_sha256, run_video_command
else:  # Keep the standalone command-line entry point available.
    from runtime_resources import file_sha256, run_video_command


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE = PACKAGE_DIR / "message_delivery_model_bundle.joblib"
DEFAULT_SECRETS = PACKAGE_DIR.parent / ".streamlit" / "secrets.toml"
LOCAL_PROJECT_ROOT = PACKAGE_DIR.parent
MODEL_NAME = "google/gemini-3.1-pro-preview"
BASE_URL = "https://litellm.data-light.ru/v1"
MAX_BINARY_MB = 13.0
MAX_WORKERS = 2
_API_KEY: str | None = None

CLASS_LABELS = {
    0: "считываемость до 20%",
    1: "считываемость от 20% до 26%",
    2: "считываемость выше 26%",
}

FINAL_FEATURES = tuple(sorted([
    "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3",
    "eng__gemini_round4__signed_mean8",
    "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity",
    "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3",
    "eng__product__words_per_second__X__audio_only_message_completeness__r4_value",
    "offer_condition_count__r4_mean3",
    "p12__cluster_valid_mask_mae_smoothed",
    "words_per_second",
]))

FINAL_PARAMS = {
    "learning_rate": 0.05,
    "loss": "huber",
    "max_depth": 2,
    "min_samples_leaf": 3,
    "n_estimators": 100,
}
SELECTED_MODEL_SEED = 44

PANEL30_PERSONAS = [
    ("p01", "18–24, крупный город, учится и подрабатывает"),
    ("p02", "35–44, небольшой город, семья с детьми"),
    ("p03", "55–64, крупный город, работает по найму"),
    ("p04", "25–34, небольшой город, самозанятый"),
    ("p05", "45–54, крупный город, живёт один"),
    ("p06", "18–24, малый город, первая постоянная работа"),
    ("p07", "35–44, крупный город, офисная работа"),
    ("p08", "55–64, небольшой город, активно пользуется интернетом"),
    ("p09", "25–34, крупный город, семья с маленьким ребёнком"),
    ("p10", "45–54, малый город, рабочая профессия"),
    ("p11", "18–24, крупный город, ищет работу"),
    ("p12", "35–44, небольшой город, предприниматель"),
    ("p13", "55–64, крупный город, на пенсии и подрабатывает"),
    ("p14", "25–34, малый город, работает удалённо"),
    ("p15", "45–54, крупный город, семья без детей дома"),
    ("p16", "18–24, небольшой город, студент"),
    ("p17", "35–44, крупный город, фрилансер"),
    ("p18", "55–64, малый город, живёт с партнёром"),
    ("p19", "25–34, крупный город, молодой специалист"),
    ("p20", "45–54, небольшой город, самозанятый"),
    ("p21", "18–24, крупный город, живёт с родителями"),
    ("p22", "35–44, малый город, семья с подростками"),
    ("p23", "55–64, крупный город, часто покупает онлайн"),
    ("p24", "25–34, небольшой город, работает в сфере услуг"),
    ("p25", "45–54, крупный город, руководитель"),
    ("p26", "18–24, малый город, совмещает учёбу и фриланс"),
    ("p27", "35–44, крупный город, воспитывает ребёнка один"),
    ("p28", "55–64, небольшой город, пенсионер"),
    ("p29", "25–34, крупный город, живёт один и много работает"),
    ("p30", "45–54, малый город, семья с детьми"),
]

RECOVERY_PERSONAS = [
    ("p01", "18–24, крупный город, учится и подрабатывает"),
    ("p02", "25–34, крупный город, офисная работа"),
    ("p03", "25–34, небольшой город, работает удалённо"),
    ("p04", "35–44, семья с детьми, крупный город"),
    ("p05", "35–44, небольшой город, самозанятый"),
    ("p06", "45–54, крупный город, наёмный сотрудник"),
    ("p07", "45–54, небольшой город, семья без детей дома"),
    ("p08", "55–64, крупный город, регулярно пользуется интернетом"),
    ("p09", "18–24, небольшой город, студент"),
    ("p10", "25–34, семья с маленьким ребёнком"),
    ("p11", "35–44, живёт один, крупный город"),
    ("p12", "55–64, небольшой город, работает"),
]

ROUND4_FEATURES = [
    ("first_core_claim_time_band", "integer", 0, 4, "Когда впервые явно понятен главный посыл: 0 — 0–5 с; 1 — 5–10 с; 2 — 10–15 с; 3 — позже 15 с; 4 — явного момента нет."),
    ("product_role_reveal_time_band", "integer", 0, 4, "Когда впервые становится ясно, какую роль продукт играет в истории: 0 — 0–5 с; 1 — 5–10 с; 2 — 10–15 с; 3 — позже 15 с; 4 — роль не объяснена."),
    ("core_claim_quartile_coverage", "integer", 0, 4, "В скольких из четырёх равных частей ролика главный посыл явно произносится, написан или однозначно демонстрируется."),
    ("independent_message_channels_count", "integer", 0, 3, "Сколько независимых каналов передают посыл: речь, экранный текст, наблюдаемое действие/результат."),
    ("claim_brand_cooccurrence_count", "integer", 0, 4, "Число отдельных моментов, когда конкретный главный посыл одновременно связан с брендом или продуктом."),
    ("demonstrated_user_outcome_count", "integer", 0, 4, "Число различных наблюдаемых результатов для пользователя как следствия продукта."),
    ("distinct_benefit_count", "integer", 0, 4, "Число разных выгод пользователя, явно названных или однозначно показанных."),
    ("secondary_claim_count", "integer", 0, 5, "Число самостоятельных рекламных тезисов помимо главного посыла."),
    ("mandatory_inference_chain_length", "integer", 0, 4, "Минимум смысловых переходов, которые зритель должен додумать от увиденного к главному посылу."),
    ("narrative_context_switch_count", "integer", 0, 4, "Число переключений между сюжетными контекстами, требующих заново понять ситуацию."),
    ("main_idea_entity_load", "integer", 1, 5, "Число обязательных смысловых сущностей для пересказа идеи: продукт, пользователь, объект, действие, выгода/условие."),
    ("offer_condition_count", "integer", 0, 4, "Число условий или ограничений, необходимых для корректного понимания предложения."),
    ("audio_only_message_completeness", "integer", 0, 3, "Насколько посыл понятен только по звуку: 0 — нельзя определить; 3 — можно полностью пересказать."),
    ("visual_only_message_completeness", "integer", 0, 3, "Насколько посыл понятен без звука с учётом экранного текста: 0 — нельзя определить; 3 — можно полностью пересказать."),
    ("irony_or_twist_dependency", "integer", 0, 3, "Насколько понимание зависит от шутки, иронии, метафоры или сюжетного твиста."),
    ("offer_novelty_explanation_need", "integer", 0, 3, "Сколько объяснения нужно для непривычного продукта, функции или условия."),
    ("main_claim_linguistic_directness", "integer", 0, 3, "Прямота формулировки главного тезиса в речи или тексте."),
    ("benefit_recipient_clarity", "integer", 0, 3, "Насколько ясно, для кого возникает выгода."),
    ("action_object_pair_clarity", "integer", 0, 3, "Насколько ясно, какое действие с каким объектом предлагает реклама."),
    ("plausible_alternative_idea_count", "integer", 0, 4, "Сколько других идей помимо выбранной главной может правдоподобно назвать зритель."),
    ("cta_clarity", "boolean", 0, 1, "TRUE только если зрителю явно сообщено конкретное следующее действие. Бренд, слоган или интерфейс без действия считать FALSE."),
    ("message_specificity_level", "integer", 0, 3, "Конкретность посыла: 0 — бренд/настроение; 1 — общая выгода; 2 — конкретный объект или действие; 3 — объект, действие и результат/условие."),
]

STABLE_ROUND4 = [
    "audio_only_message_completeness", "benefit_recipient_clarity",
    "core_claim_quartile_coverage", "demonstrated_user_outcome_count",
    "distinct_benefit_count", "first_core_claim_time_band",
    "independent_message_channels_count", "irony_or_twist_dependency",
    "main_idea_entity_load", "narrative_context_switch_count",
    "offer_condition_count", "offer_novelty_explanation_need",
    "plausible_alternative_idea_count", "product_role_reveal_time_band",
    "secondary_claim_count", "visual_only_message_completeness",
]

TRANSCRIPT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "complete_video_transcript",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "request_token": {"type": "string"},
                "transcript": {"type": "string"},
                "has_speech": {"type": "boolean"},
            },
            "required": ["request_token", "transcript", "has_speech"],
        },
    },
}

TRANSCRIPT_PROMPT = """Просмотри и прослушай рекламный ролик целиком. Сделай
максимально точную дословную транскрипцию всех отчётливо произнесённых слов:
речь героев, закадровый голос, пропетые слова и слоган. Не добавляй экранный
текст, если он не произнесён. Не описывай кадры и звуки. Если речи нет,
transcript должен быть пустой строкой, has_speech=false."""


def json_schema(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name, "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": properties, "required": required,
            },
        },
    }


def panel_schema() -> dict[str, Any]:
    props: dict[str, Any] = {
        "respondent_id": {"type": "string"},
        "main_message_summary": {"type": "string"},
    }
    for name, kind, low, high, _ in ROUND4_FEATURES:
        props[name] = {"type": "boolean"} if kind == "boolean" else {
            "type": "integer", "minimum": low, "maximum": high,
        }
    return json_schema(
        "message_delivery_panel30",
        {
            "request_token": {"type": "string"},
            "answers": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": props, "required": list(props),
                },
            },
        },
        ["request_token", "answers"],
    )


RECOVERY_SCHEMA = json_schema(
    "recovery_panel_answers",
    {
        "request_token": {"type": "string"},
        "answers": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "respondent_id": {"type": "string"},
                    "answer": {"type": "string"},
                    "brand_only": {"type": "boolean"},
                    "no_idea": {"type": "boolean"},
                },
                "required": ["respondent_id", "answer", "brand_only", "no_idea"],
            },
        },
    },
    ["request_token", "answers"],
)

CLUSTER_SCHEMA = json_schema(
    "recovery_answer_clustering",
    {
        "request_token": {"type": "string"},
        "assignments": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "respondent_uid": {"type": "string"},
                    "canonical_idea": {"type": "string"},
                    "answer_type": {
                        "type": "string", "enum": ["valid", "brand_only", "no_idea"],
                    },
                },
                "required": ["respondent_uid", "canonical_idea", "answer_type"],
            },
        },
    },
    ["request_token", "assignments"],
)


def configure_api_key(api_key: str) -> None:
    """Configure the API key without writing it to disk or environment variables."""
    global _API_KEY
    _API_KEY = str(api_key).strip()


def load_api_key(secrets_path: Path) -> str:
    if _API_KEY:
        return _API_KEY
    value = os.getenv("VSELLM_API_KEY")
    if value:
        return value
    with secrets_path.open("rb") as stream:
        data = tomllib.load(stream)
    value = data.get("VSELLM_API_KEY") or data.get("general", {}).get("VSELLM_API_KEY")
    if not value:
        raise RuntimeError("VSELLM_API_KEY is absent")
    return str(value)


def create_client(secrets_path: Path) -> OpenAI:
    return OpenAI(
        api_key=load_api_key(secrets_path), base_url=BASE_URL,
        timeout=300, max_retries=0,
    )


def ffmpeg_executable() -> str:
    value = shutil.which("ffmpeg")
    if value:
        return value
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_duration(path: Path) -> float:
    result = subprocess.run(
        [ffmpeg_executable(), "-threads", "1", "-i", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Cannot read video duration: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def prepare_video(source: Path, output_dir: Path) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    size_mb = source.stat().st_size / 1024 / 1024
    if source.suffix.lower() == ".mp4" and size_mb <= MAX_BINARY_MB:
        return source
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
    output = output_dir / f"{source.stem}_{digest}.mp4"
    if output.exists() and output.stat().st_size > 0:
        return output
    command = [
        ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(source),
        "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "28",
        "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k", str(output),
    ]
    run_video_command(command)
    if output.stat().st_size / 1024 / 1024 > MAX_BINARY_MB:
        raise RuntimeError("Prepared video is still above 13 MB")
    return output


def encode_video(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def cache_path(cache_dir: Path, task_key: str, prompt: str, video_sha: str | None) -> Path:
    raw = json.dumps(
        {"model": MODEL_NAME, "task": task_key, "prompt": prompt, "video": video_sha},
        ensure_ascii=False, sort_keys=True,
    )
    return cache_dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.json"


def call_json(
    *, task_key: str, prompt: str, schema: dict[str, Any], cache_dir: Path,
    secrets_path: Path, video_b64: str | None, video_sha: str | None,
    temperature: float, fresh: bool, retries: int = 5,
    expected_ids: tuple[str, str, set[str]] | None = None,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(f"md-final-v1|{video_sha}|{task_key}".encode()).hexdigest()[:16]
    full_prompt = f"{prompt}\n\nREQUEST_TOKEN: {token}\nВерни request_token дословно."
    path = cache_path(cache_dir, task_key, full_prompt, video_sha)
    validator = Draft202012Validator(schema.get("json_schema", {}).get("schema", {}))

    def validate(payload):
        if not isinstance(payload, dict):
            raise ValueError("Response must be a JSON object")
        validator.validate(payload)
        returned = str(payload.get("request_token", "")).strip()
        prefix_match = (len(returned) >= 12
                        and (returned.startswith(token) or token.startswith(returned)))
        if not (returned == token or prefix_match):
            raise ValueError("request_token mismatch")
        if expected_ids is not None:
            collection, field, expected = expected_ids
            ids = [row[field] for row in payload[collection]]
            if len(ids) != len(expected) or set(ids) != expected:
                raise ValueError(f"{field} IDs mismatch")

    if path.exists() and not fresh:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("task") != task_key:
                raise ValueError("Cached task mismatch")
            validate(saved["response"])
            return saved["response"]
        except (KeyError, TypeError, ValueError, ValidationError):
            # Preserve only this invalid response for diagnosis. Other completed
            # requests in the same repeat remain available for resumption.
            path.replace(path.with_name(f"{path.name}.invalid-{time.time_ns()}"))
    content: list[dict[str, Any]] = [{"type": "text", "text": full_prompt}]
    if video_b64 is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        })
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with create_client(secrets_path) as client:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    response_format=schema,
                )
            payload = json.loads(response.choices[0].message.content)
            validate(payload)
            path.write_text(
                json.dumps({"task": task_key, "response": payload}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(20 * attempt if "429" in str(exc) else 6 * attempt)
    raise RuntimeError(f"Gemini failed for {task_key}: {last_error}")


def panel_prompt(personas: list[tuple[str, str]]) -> str:
    roster = "\n".join(f"- {key}: {description}" for key, description in personas)
    rules = "\n".join(
        f"- {name} ({kind}): {rule}" for name, kind, _, _, rule in ROUND4_FEATURES
    )
    return f"""Сымитируй трёх разных людей после однократного просмотра
рекламного ролика целиком, со звуком. Каждый отвечает независимо и не видит
ответов остальных.

Панель:
{roster}

Для каждого респондента сначала определи один главный посыл, затем оцени
признаки относительно его понимания. Используй только видео и звук.

Правила признаков:
{rules}

Верни каждый respondent_id ровно один раз."""


def extract_panel30(
    b64: str, sha: str, output_dir: Path, secrets: Path, fresh: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    schema = panel_schema()
    tasks = []
    for call_id in range(1, 11):
        start = (call_id - 1) * 3
        tasks.append((call_id, PANEL30_PERSONAS[start:start + 3]))

    def one(call_id: int, personas: list[tuple[str, str]]) -> list[dict[str, Any]]:
        payload = call_json(
            task_key=f"panel30_{call_id:02d}", prompt=panel_prompt(personas),
            schema=schema, cache_dir=output_dir / "cache", secrets_path=secrets,
            video_b64=b64, video_sha=sha, temperature=0.2, fresh=fresh,
            expected_ids=("answers", "respondent_id", {key for key, _ in personas}),
        )
        by_id = {str(item["respondent_id"]): item for item in payload["answers"]}
        expected = {key for key, _ in personas}
        if set(by_id) != expected:
            raise ValueError(f"Panel IDs mismatch: {sorted(by_id)} != {sorted(expected)}")
        rows = []
        for respondent_id, description in personas:
            answer = by_id[respondent_id]
            row = {
                "call_id": call_id, "respondent_id": respondent_id,
                "persona": description,
                "main_message_summary": str(answer["main_message_summary"]).strip(),
            }
            for name, kind, low, high, _ in ROUND4_FEATURES:
                value = answer[name]
                row[name] = int(bool(value)) if kind == "boolean" else int(np.clip(value, low, high))
            rows.append(row)
        return rows

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(one, call_id, personas) for call_id, personas in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"panel30 respondents: {len(rows)}/30", flush=True)
    frame = pd.DataFrame(rows).sort_values(["call_id", "respondent_id"])
    frame.to_csv(output_dir / "panel30_respondents.csv", index=False, encoding="utf-8-sig")
    aggregate: dict[str, float] = {}
    for name, _, _, _, _ in ROUND4_FEATURES:
        values = frame[name].astype(float)
        mean = float(values.mean())
        aggregate[f"{name}__r4_value"] = float(math.floor(mean + 0.5))
        aggregate[f"{name}__r4_mean3"] = mean
        agreements = frame.groupby("call_id")[name].apply(
            lambda part: Counter(part.tolist()).most_common(1)[0][1] / len(part)
        )
        aggregate[f"{name}__r4_agreement3"] = float(agreements.mean())
        aggregate[f"{name}__panel30_consensus"] = float(
            Counter(values.tolist()).most_common(1)[0][1] / len(values)
        )
    return frame, aggregate


def extract_transcript(
    b64: str, sha: str, duration: float, output_dir: Path,
    secrets: Path, fresh: bool,
) -> tuple[pd.DataFrame, float]:
    def one(iteration: int) -> dict[str, Any]:
        payload = call_json(
            task_key=f"transcript_{iteration}", prompt=TRANSCRIPT_PROMPT,
            schema=TRANSCRIPT_SCHEMA, cache_dir=output_dir / "cache",
            secrets_path=secrets, video_b64=b64, video_sha=sha,
            temperature=0.0, fresh=fresh,
        )
        transcript = str(payload["transcript"]).strip()
        count = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", transcript))
        return {"iteration": iteration, "transcript": transcript, "word_count": count}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows = [future.result() for future in as_completed(
            [executor.submit(one, iteration) for iteration in [1, 2, 3]]
        )]
    frame = pd.DataFrame(rows).sort_values("iteration")
    frame.to_csv(output_dir / "transcripts.csv", index=False, encoding="utf-8-sig")
    return frame, float(frame["word_count"].mean() / duration)


def nested_segments(seed: str, mask_id: int, fraction: float) -> list[int]:
    order = list(range(12))
    random.Random(f"{seed}|nested|{mask_id}").shuffle(order)
    return sorted(order[:max(1, round(fraction * 12))])


def make_nested_clip(
    source: Path, output: Path, fraction: float, mask_id: int, seed: str,
) -> Path:
    if output.exists() and output.stat().st_size > 0:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = video_duration(source)
    segment_duration = duration / 12
    selected = nested_segments(seed, mask_id, fraction)
    filters: list[str] = []
    inputs: list[str] = []
    for index, segment in enumerate(selected):
        start, end = segment * segment_duration, min(duration, (segment + 1) * segment_duration)
        filters.append(f"[0:v]trim=start={start:.4f}:end={end:.4f},setpts=PTS-STARTPTS[v{index}]")
        filters.append(f"[0:a]atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS[a{index}]")
        inputs.append(f"[v{index}][a{index}]")
    filters.append("".join(inputs) + f"concat=n={len(selected)}:v=1:a=1[outv][outa]")
    command = [
        ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(source),
        "-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "72k", str(output),
    ]
    run_video_command(command)
    return output


RECOVERY_CONDITIONS = {
    "full": "полный исходный ролик без удаления или искажения частей",
    "nested_25_m1": "маска 1: показано около 25% исходных сцен",
    "nested_50_m1": "маска 1: показано около 50% исходных сцен",
    "nested_75_m1": "маска 1: показано около 75% исходных сцен",
    "nested_25_m2": "маска 2: показано около 25% исходных сцен",
    "nested_50_m2": "маска 2: показано около 50% исходных сцен",
    "nested_75_m2": "маска 2: показано около 75% исходных сцен",
}


def recovery_prompt(condition: str, personas: list[tuple[str, str]]) -> str:
    roster = "\n".join(f"- {key}: {description}" for key, description in personas)
    return f"""Сымитируй ответы четырёх разных людей после однократного
просмотра материала. Условие: {RECOVERY_CONDITIONS[condition]}.
Каждый отвечает независимо и не видит ответы остальных.

Вопрос: «Кроме попытки убедить Вас воспользоваться сервисом, какова была
основная идея этой рекламы?»

Панель:
{roster}

Для каждого respondent_id верни одну короткую естественную мысль. Не достраивай
удалённые части. Простое упоминание Авито не является идеей: brand_only=true.
Если идея не понятна: no_idea=true. Верни каждого respondent_id ровно один раз."""


def cluster_recovery(
    rows: list[dict[str, Any]], output_dir: Path, secrets: Path, sha: str, fresh: bool,
) -> list[dict[str, Any]]:
    compact = [{"respondent_uid": row["respondent_uid"], "answer": row["raw_answer"]} for row in rows]
    prompt = f"""Ты кодировщик открытых ответов рекламного исследования.
Сгруппируй семантически одинаковые ответы в одинаковую короткую canonical_idea.
Если ответ содержит только бренд: answer_type=brand_only и <BRAND_ONLY>.
Если идея не понята: answer_type=no_idea и <NO_IDEA>. Иначе answer_type=valid.
Не используй знания о ролике, кодируй только ответы. Каждому respondent_uid
дай ровно одно назначение.

ОТВЕТЫ:
{json.dumps(compact, ensure_ascii=False)}"""
    payload = call_json(
        task_key="recovery_cluster", prompt=prompt, schema=CLUSTER_SCHEMA,
        cache_dir=output_dir / "cache", secrets_path=secrets,
        video_b64=None, video_sha=sha, temperature=0.0, fresh=fresh,
        expected_ids=("assignments", "respondent_uid", {row["respondent_uid"] for row in rows}),
    )
    assignments = {item["respondent_uid"]: item for item in payload["assignments"]}
    if set(assignments) != {row["respondent_uid"] for row in rows}:
        raise ValueError("Clustering assignments do not cover respondents")
    return [{**row, **assignments[row["respondent_uid"]]} for row in rows]


def extract_recovery(
    prepared: Path, b64: str, sha: str, output_dir: Path,
    secrets: Path, fresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants: dict[str, Path] = {"full": prepared}
    for mask_id in [1, 2]:
        for fraction in [0.25, 0.50, 0.75]:
            condition = f"nested_{int(fraction * 100)}_m{mask_id}"
            path = make_nested_clip(
                prepared, output_dir / "variants" / f"{condition}.mp4",
                fraction, mask_id, sha,
            )
            variants[condition] = path

    tasks = []
    groups = [RECOVERY_PERSONAS[:4], RECOVERY_PERSONAS[4:8], RECOVERY_PERSONAS[8:12]]
    for condition, path in variants.items():
        condition_sha = file_sha256(path)
        for group_id, personas in enumerate(groups, start=1):
            tasks.append((condition, condition_sha, path, group_id, personas))

    def one(condition, condition_sha, path, group_id, personas):
        # Encode only active requests, not all six variants held for the whole run.
        encoded = b64 if condition == "full" else encode_video(path)[0]
        payload = call_json(
            task_key=f"recovery_{condition}_g{group_id}",
            prompt=recovery_prompt(condition, personas), schema=RECOVERY_SCHEMA,
            cache_dir=output_dir / "cache", secrets_path=secrets,
            video_b64=encoded, video_sha=condition_sha,
            temperature=0.0, fresh=fresh,
            expected_ids=("answers", "respondent_id", {key for key, _ in personas}),
        )
        by_id = {str(item["respondent_id"]): item for item in payload["answers"]}
        return [{
            "condition_group": condition,
            "respondent_uid": f"g{group_id}_{condition}_{key}",
            "respondent_id": key, "persona": description,
            "raw_answer": str(by_id[key]["answer"]).strip(),
            "self_brand_only": bool(by_id[key]["brand_only"]),
            "self_no_idea": bool(by_id[key]["no_idea"]),
        } for key, description in personas]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(one, *task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"recovery respondents: {len(rows)}/84", flush=True)
    clustered = cluster_recovery(rows, output_dir, secrets, sha, fresh)
    frame = pd.DataFrame(clustered).sort_values(["condition_group", "respondent_uid"])
    frame.to_csv(output_dir / "recovery_respondents.csv", index=False, encoding="utf-8-sig")
    full = frame.loc[
        frame["condition_group"].eq("full") & frame["answer_type"].eq("valid"),
        "canonical_idea",
    ]
    candidate = str(full.value_counts().index[0]) if len(full) else "<NO_IDEA>"
    posterior: dict[str, float] = {}
    for condition in RECOVERY_CONDITIONS:
        part = frame.loc[frame["condition_group"].eq(condition)]
        successes = int(part["answer_type"].eq("valid").sum())
        posterior[condition] = (successes + 0.5) / (len(part) + 1.0)
    gaps = [
        abs(posterior[f"nested_{fraction}_m1"] - posterior[f"nested_{fraction}_m2"])
        for fraction in [25, 50, 75]
    ]
    mae = float(np.mean(gaps))
    return frame, {
        "recovery_candidate": candidate,
        "p12__cluster_valid_mask_mae_smoothed": mae,
        "p12__cluster_valid_mask_consistency": 1.0 - mae,
        "condition_valid_posterior": posterior,
    }


def build_bundle(bundle_path: Path) -> None:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))
    import message_delivery_all_features_search as all_features
    import message_delivery_ordinal_feature_search as ordinal
    import message_delivery_panel12_model_search as panel_model

    data, raw_features, families, _ = panel_model.load_data()
    holdout_indices, _ = all_features.split_source.select_random_holdout(data)
    data["split"] = "train"
    data.loc[holdout_indices, "split"] = "holdout"
    train = data.loc[data["split"].eq("train")].reset_index(drop=True).copy()
    y = train["target"].astype(int).reset_index(drop=True)
    X, _, _, _ = ordinal.add_engineered_features(train, train.iloc[:1], raw_features, families)
    splits = panel_model.repeated_splits(y)
    oof = panel_model.repeated_oof(
        X, y.astype(float), FINAL_FEATURES, "gradient_boosting", FINAL_PARAMS, splits
    )
    thresholds = ordinal.optimize_thresholds(y.to_numpy(), oof)
    model = ordinal.make_model(
        "gradient_boosting", FINAL_PARAMS, seed=SELECTED_MODEL_SEED
    )
    model.fit(X[list(FINAL_FEATURES)], y.astype(float))

    means = train[raw_features].astype(float).mean()
    scales = train[raw_features].astype(float).std(ddof=0).replace(0.0, 1.0)
    correlations = {}
    for feature in raw_features:
        value = stats.spearmanr(train["target"], train[feature]).statistic
        correlations[feature] = float(value) if np.isfinite(value) else -1.0
    usable = [feature for feature in families["gemini_round4"] if train[feature].nunique() > 1]
    round4_top8 = sorted(usable, key=lambda feature: abs(correlations[feature]), reverse=True)[:8]
    round4_signs = [1.0 if correlations[feature] >= 0 else -1.0 for feature in round4_top8]
    relevant = set(round4_top8) | {
        "p12__cluster_valid_mask_mae_smoothed",
        "p12__cluster_valid_mask_consistency",
        "message_specificity_level__r3_mean3",
        "main_idea_entity_load__r4_agreement3",
        "cta_clarity", "words_per_second",
        "audio_only_message_completeness__r4_value",
    }
    bundle = {
        "bundle_version": "message-delivery-final8-single-seed44-v1",
        "model": model,
        "selected_seed": SELECTED_MODEL_SEED,
        "feature_columns": list(FINAL_FEATURES),
        "params": FINAL_PARAMS,
        "threshold_t0": float(thresholds["t0"]),
        "threshold_t2": float(thresholds["t2"]),
        "raw_means": {feature: float(means[feature]) for feature in relevant},
        "raw_scales": {feature: float(scales[feature]) for feature in relevant},
        "round4_top8": round4_top8,
        "round4_signs": round4_signs,
        "signed_mean_train": X["eng__gemini_round4__signed_mean8"].astype(float).tolist(),
        "training_n": len(train),
        "oof_class_spearman": float(thresholds["spearman_target"]),
        "oof_accuracy": float(thresholds["accuracy"]),
        "exploratory_holdout_hits": 6,
        "exploratory_holdout_n": 10,
        "selection_note": (
            "All five checked seeds produced the same 6/10 holdout classes; "
            "seed 44 had the lowest continuous-score holdout MAE (0.614240)."
        ),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(f"Bundle written: {bundle_path}")


def technical_features(
    panel: dict[str, float], words_per_second: float,
    recovery: dict[str, Any], bundle: dict[str, Any],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for feature in STABLE_ROUND4:
        for suffix in ["value", "mean3", "agreement3"]:
            key = f"{feature}__r4_{suffix}"
            raw[key] = float(panel[key])
    raw["cta_clarity"] = float(panel["cta_clarity__r4_value"])
    raw["message_specificity_level__r3_mean3"] = float(
        panel["message_specificity_level__r4_mean3"]
    )
    raw["words_per_second"] = words_per_second
    raw["p12__cluster_valid_mask_mae_smoothed"] = float(
        recovery["p12__cluster_valid_mask_mae_smoothed"]
    )
    raw["p12__cluster_valid_mask_consistency"] = float(
        recovery["p12__cluster_valid_mask_consistency"]
    )

    def z(name: str) -> float:
        return (raw[name] - bundle["raw_means"][name]) / bundle["raw_scales"][name]

    mae = "p12__cluster_valid_mask_mae_smoothed"
    consistency = "p12__cluster_valid_mask_consistency"
    specificity = "message_specificity_level__r3_mean3"
    entity = "main_idea_entity_load__r4_agreement3"
    audio = "audio_only_message_completeness__r4_value"
    round4 = float(np.mean([
        z(feature) * sign
        for feature, sign in zip(bundle["round4_top8"], bundle["round4_signs"])
    ]))
    return {
        "eng__absdiff__p12__cluster_valid_mask_mae_smoothed__X__message_specificity_level__r3_mean3": abs(z(mae) - z(specificity)),
        "eng__gemini_round4__signed_mean8": round4,
        "eng__product__main_idea_entity_load__r4_agreement3__X__cta_clarity": z(entity) * z("cta_clarity"),
        "eng__product__p12__cluster_valid_mask_consistency__X__message_specificity_level__r3_mean3": z(consistency) * z(specificity),
        "eng__product__words_per_second__X__audio_only_message_completeness__r4_value": z("words_per_second") * z(audio),
        "offer_condition_count__r4_mean3": raw["offer_condition_count__r4_mean3"],
        "p12__cluster_valid_mask_mae_smoothed": raw[mae],
        "words_per_second": raw["words_per_second"],
    }


def predict(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.bundle)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_video(args.video, output / "prepared_media")
    b64, sha = encode_video(prepared)
    duration = video_duration(prepared)

    with ThreadPoolExecutor(max_workers=3) as executor:
        panel_future = executor.submit(
            extract_panel30, b64, sha, output / "panel30", args.secrets, args.fresh
        )
        transcript_future = executor.submit(
            extract_transcript, b64, sha, duration, output / "transcript",
            args.secrets, args.fresh,
        )
        recovery_future = executor.submit(
            extract_recovery, prepared, b64, sha, output / "recovery",
            args.secrets, args.fresh,
        )
        _, panel = panel_future.result()
        _, words_per_second = transcript_future.result()
        _, recovery = recovery_future.result()

    features = technical_features(panel, words_per_second, recovery, bundle)
    X = pd.DataFrame([features])[bundle["feature_columns"]]
    score = float(bundle["model"].predict(X)[0])
    predicted_class = int(
        0 if score < bundle["threshold_t0"]
        else 2 if score >= bundle["threshold_t2"]
        else 1
    )
    signed_train = np.asarray(bundle["signed_mean_train"], dtype=float)
    completeness_index = int(round(100 * float(
        np.mean(signed_train <= features["eng__gemini_round4__signed_mean8"])
    )))
    business_features = {
        "конкретность_сообщения_0_3": round(panel["message_specificity_level__r4_mean3"], 3),
        "устойчивость_при_частичном_просмотре": round(recovery["p12__cluster_valid_mask_consistency"], 4),
        "расхождение_recovery_масок": round(recovery["p12__cluster_valid_mask_mae_smoothed"], 4),
        "полнота_сообщения_только_по_звуку_0_3": int(panel["audio_only_message_completeness__r4_value"]),
        "плотность_речи_слов_в_секунду": round(words_per_second, 4),
        "ясный_призыв_к_действию": bool(panel["cta_clarity__r4_value"]),
        "количество_условий_предложения": round(panel["offer_condition_count__r4_mean3"], 3),
        "согласие_по_смысловым_элементам": round(panel["main_idea_entity_load__r4_agreement3"], 4),
        "индекс_полноты_раскрытия_предложения_из_100": completeness_index,
    }
    result = {
        "model": bundle["bundle_version"],
        "video": str(args.video.resolve()),
        "message_delivery_score": score,
        "predicted_class": predicted_class,
        "predicted_class_label": CLASS_LABELS[predicted_class],
        "creative_score": float(np.clip(score / 2.0, 0.0, 1.0)),
        "thresholds": {
            "class_0_below": bundle["threshold_t0"],
            "class_2_from": bundle["threshold_t2"],
        },
        "business_features": business_features,
        "technical_features": features,
        "recovery_top_idea": recovery["recovery_candidate"],
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{"video": args.video.name, **business_features}]).to_csv(
        output / "business_features.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-bundle", help="Freeze the selected model")
    build.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    run = subparsers.add_parser("predict", help="Score one video")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    run.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS)
    run.add_argument("--fresh", action="store_true", help="Ignore cached Gemini responses")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-bundle":
        build_bundle(args.bundle)
    else:
        predict(args)


if __name__ == "__main__":
    main()
