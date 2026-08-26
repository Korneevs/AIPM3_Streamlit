from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
from openai import OpenAI


MODEL_NAME = "google/gemini-3.1-pro-preview"
BASE_URL = "https://litellm.data-light.ru/v1"
N_ITERATIONS = 5

AIPM1_FEATURES = [
    "main_character",
    "promo",
    "humor",
    "state_transformation",
    "message_focus_seconds",
    "has_screen_offer_text",
    "unique_offer_count",
]

AIPM2_VIDEO_FEATURES = [
    "brand_logo_screen_seconds",
    "pack_shot_duration_seconds",
    "monologue_to_camera",
    "jingle_present",
    "ends_with_brand_logo_alone",
    "scene_pace_high",
    "vocal_song_present",
    "silence_or_music_only_seconds",
]

BOOL_FEATURES = {
    "main_character",
    "promo",
    "humor",
    "state_transformation",
    "has_screen_offer_text",
    "monologue_to_camera",
    "jingle_present",
    "ends_with_brand_logo_alone",
    "scene_pace_high",
    "vocal_song_present",
}

INT_FEATURES = {
    "message_focus_seconds",
    "unique_offer_count",
    "brand_logo_screen_seconds",
    "pack_shot_duration_seconds",
    "silence_or_music_only_seconds",
}


PROMPT = """Ты — рекламный аналитик. Просмотри и прослушай рекламный ролик
целиком. Заполни только наблюдаемые признаки. Бренд ролика — Avito.

Общие правила:
- Не оценивай общее качество и не угадывай намерения автора.
- Для бинарных полей при сомнении ставь false.
- Для длительностей считай суммарное время в секундах и округляй до целого.
- Каждый ответ является новым независимым просмотром: не используй знания из
  других запросов и не пытайся согласовать результат с ними.

Поля AIPM 1.0:
1. main_character — один и тот же герой явно доминирует в ролике.
2. promo — есть конкретная акция, скидка, бонус, промокод или ограниченный оффер.
3. humor — юмор является заметной частью коммуникации, а не просто позитивным тоном.
4. state_transformation — показан контраст «до/после» благодаря продукту.
5. message_focus_seconds: 3 — средний зритель сформулирует оффер мгновенно;
   2 — сформулирует с усилием; 1 — не сможет назвать содержательную идею.
6. has_screen_offer_text — на экране есть конкретный числовой оффер, цена,
   процент, дедлайн или «бесплатно» с предметом.
7. unique_offer_count — число разных конкретных офферов; повтор не считать дважды.

Поля AIPM 2.0:
8. brand_logo_screen_seconds — суммарное время заметного логотипа или маскота Avito.
9. pack_shot_duration_seconds — длительность финального кадра, где доминируют
   логотип, слоган или CTA; если его нет, 0.
10. monologue_to_camera — герой обращается прямо в камеру.
11. jingle_present — есть короткий запоминающийся брендовый музыкальный мотив.
12. ends_with_brand_logo_alone — финал занят только лого/CTA без героев и сюжета.
13. scene_pace_high — в среднем не меньше одной монтажной склейки в секунду.
14. vocal_song_present — в ролике есть вокальная мелодия, пение или рэп.
15. silence_or_music_only_seconds — суммарное время без речи.

Верни строго JSON по схеме и повтори request_token дословно."""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "aipm3_objective_features",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "request_token": {"type": "string"},
                "main_character": {"type": "boolean"},
                "promo": {"type": "boolean"},
                "humor": {"type": "boolean"},
                "state_transformation": {"type": "boolean"},
                "message_focus_seconds": {"type": "integer", "minimum": 1, "maximum": 3},
                "has_screen_offer_text": {"type": "boolean"},
                "unique_offer_count": {"type": "integer", "minimum": 0, "maximum": 20},
                "brand_logo_screen_seconds": {"type": "integer", "minimum": 0, "maximum": 600},
                "pack_shot_duration_seconds": {"type": "integer", "minimum": 0, "maximum": 600},
                "monologue_to_camera": {"type": "boolean"},
                "jingle_present": {"type": "boolean"},
                "ends_with_brand_logo_alone": {"type": "boolean"},
                "scene_pace_high": {"type": "boolean"},
                "vocal_song_present": {"type": "boolean"},
                "silence_or_music_only_seconds": {"type": "integer", "minimum": 0, "maximum": 600},
            },
            "required": [
                "request_token",
                *AIPM1_FEATURES,
                *AIPM2_VIDEO_FEATURES,
            ],
        },
    },
}


def create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=300, max_retries=0)


def _video_block(video_base64: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:video/mp4;base64,{video_base64}"},
    }


def _one_call(
    client: OpenAI,
    video_base64: str,
    video_sha: str,
    iteration: int,
    retries: int = 4,
) -> dict[str, Any]:
    token = hashlib.sha256(f"aipm3-objective-v1|{video_sha}|{iteration}".encode()).hexdigest()[:16]
    prompt = f"{PROMPT}\n\nrequest_token: {token}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        _video_block(video_base64),
                    ],
                }],
                temperature=0.0,
                response_format=SCHEMA,
            )
            payload = json.loads(response.choices[0].message.content)
            returned = str(payload.get("request_token", "")).strip()
            if not (returned == token or returned.startswith(token) or token.startswith(returned)):
                raise ValueError("request_token mismatch")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(15 * attempt if "429" in str(exc) else 5 * attempt)
    raise RuntimeError(f"Gemini failed on objective iteration {iteration}: {last_error}")


def _mode(values: list[Any]) -> Any:
    return Counter(values).most_common(1)[0][0]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No objective feature rows")
    result: dict[str, Any] = {}
    for feature in BOOL_FEATURES:
        result[feature] = bool(_mode([bool(row[feature]) for row in rows]))
    for feature in INT_FEATURES:
        values = [int(row[feature]) for row in rows]
        if feature == "message_focus_seconds":
            result[feature] = int(_mode(values))
        else:
            result[feature] = int(math.floor(float(np.mean(values)) + 0.5))
    return result


def extract_objective_features(
    *,
    video_base64: str,
    video_sha: str,
    api_key: str,
    iterations: int = N_ITERATIONS,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = create_client(api_key)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_one_call, client, video_base64, video_sha, iteration): iteration
            for iteration in range(1, iterations + 1)
        }
        for future in as_completed(futures):
            rows.append(future.result())
            if on_progress:
                on_progress(len(rows), iterations)
    rows.sort(key=lambda row: str(row.get("request_token", "")))
    return aggregate(rows), rows
