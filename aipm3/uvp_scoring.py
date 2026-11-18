"""Run the blind, five-judge UVP protocol for a single uploaded creative."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from urllib.parse import urlsplit

import httpx

from . import uvp_scoring_prompts as prompts
from .uvp_strength import MODEL, UNUSABLE_ARGUMENT, score_image
from .runtime_resources import request_timeout

PANEL = tuple(prompts.MODELS)
CHECKER = PANEL[1]
BATCH_SIZE = 6
API_WORKERS = 3
_GATE = threading.BoundedSemaphore(API_WORKERS)
_ERROR_TYPES = UNUSABLE_ARGUMENT | {"literal_penalty"}
_EXTRA = {PANEL[3]: {"enable_thinking": False}, PANEL[4]: {"reasoning_effort": "low"}}
_BLIND = prompts.ENCODE + """
Добавь в JSON main_idea: одним коротким предложением, что после просмотра запомнилось
именно об Авито; если роль бренда неясна, скажи это. Ответы на русском."""
_AUDIT = prompts.CHECK + """
Дополнительно верни main_idea_judge: индекс 0..4 того свободного пересказа,
который точнее всего передаёт основную мысль ролика. Выбирай по материалу.
Не добавляй ни в пересказ, ни в факты детали со своей стороны."""
_FORMAT = """
Ответь только JSON, по-русски. Идентификаторы имиджей копируй точно. Идентификаторы
кадров и реплик перечисляй отдельно, например ["V01","V02"], без диапазонов.
Текст полей для менеджеров: без служебных ID и обсуждения процесса оценки."""


class UVPServiceError(RuntimeError):
    """Safe to present without including credentials, frames or provider response."""


def _rows(payload, field, expected, key="id"):
    rows = payload.get(field)
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"{field}: требуется ровно {len(expected)} элементов")
    if {r[key] for r in rows} != set(expected):
        raise ValueError(f"{field}: используй точные входные {key} без повторений")
    return rows


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _parse(content):
    if not isinstance(content, str):
        raise ValueError("Отсутствует JSON")
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.I)
    obj = json.loads(clean)
    if not isinstance(obj, dict):
        raise ValueError("Нужен JSON-объект")
    return obj


def _format_correction(exc, prompt, payload):
    correction = "Исправь только формат, верни полный JSON по заданной схеме. " + str(exc)[:150]
    if prompt in (prompts.EVALUATE, prompts.REPAIR):
        ids = payload.get("repair_ids") or [q["id"] for q in payload.get("questions", [])]
        if ids:
            correction += (
                f" Массив images должен содержать ровно {len(ids)} объектов, по одному для каждого ID: "
                + json.dumps(ids, ensure_ascii=False)
                + ". Включи каждый имидж, в том числе без поддержки в ролике. Не пропускай и не повторяй ID. "
                "Сохрани все обязательные поля исходной схемы. takeaway и reason_to_believe должны быть "
                "непустыми: если опоры нет, объясни её отсутствие, не придумывай подтверждений. "
                "Критерии и правила оценки не меняются."
            )
    return correction


def _call(client, model, prompt, payload, sheets, validate, tokens):
    correction = ""
    content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    content.extend({"type": "image_url", "image_url": {"url": s}} for s in sheets)
    for attempt in range(3):
        messages = [{"role": "system", "content": prompt + _FORMAT},
                    {"role": "user", "content": content}]
        if correction:
            messages.append({"role": "user", "content": correction})
        try:
            with _GATE:
                timeout = request_timeout(300)
                request_limits = httpx.Timeout(timeout, connect=min(30, timeout), write=min(90, timeout))
                response = client.post("chat/completions", timeout=request_limits, json={
                    "model": model, "messages": messages, "max_completion_tokens": tokens,
                    **_EXTRA.get(model, {}),
                })
            if response.status_code >= 400:
                if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise UVPServiceError(f"Сервис оценки временно недоступен (HTTP {response.status_code}).")
                time.sleep(attempt + 1)
                continue
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Ответ неполный; сократи текст, верни полный JSON")
            obj = _parse(choice["message"]["content"])
            validate(obj)
            return obj
        except (httpx.TransportError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            if attempt == 2:
                raise UVPServiceError("Не удалось получить полный проверенный ответ оценщика. Повторите попытку.") from exc
            if isinstance(exc, httpx.TransportError):
                time.sleep(attempt + 1)
            else:
                correction = _format_correction(exc, prompt, payload)


def _validate_blind(obj):
    if not _nonempty(obj.get("main_idea")) or not isinstance(obj.get("recall"), list) or not isinstance(obj.get("impression"), list):
        raise ValueError("Нужны main_idea, recall и impression")


def _validate_meanings(obj, ids):
    for r in _rows(obj, "items", ids, key="key"):
        if not all(_nonempty(r.get(k)) for k in ("belief", "object", "stage", "scope_and_parts", "not_equivalent")) or not r.get("valid_routes"):
            raise ValueError("Опиши все части определения вопроса")


def _validate_eval(obj, ids, sources, *, repair_batch=None):
    rows = obj.get("images")
    if isinstance(rows, list):
        known = set(repair_batch or ids)
        for item in rows:
            key = item.get("id", "")
            if isinstance(key, str) and ":" in key and key.rsplit(":", 1)[1] in known:
                item["id"] = key.rsplit(":", 1)[1]
        # A repair may echo the complete batch. Never overwrite the unflagged estimates.
        if repair_batch and len(rows) == len(repair_batch) and {r.get("id") for r in rows} == known:
            obj["images"] = [r for r in rows if r["id"] in ids]
    if not _nonempty(obj.get("main_idea")):
        raise ValueError("Отсутствует main_idea")
    for r in _rows(obj, "images", ids):
        clarity, strength, direction = r["clarity"], r["strength"], r["direction"]
        if type(clarity) is not int or type(strength) is not int or min(clarity, strength) < 0 or max(clarity, strength) > 3:
            raise ValueError("clarity/strength – целые от 0 до 3")
        if direction not in ("positive", "negative", "neutral") or ((strength == 0) != (direction == "neutral")):
            raise ValueError("neutral только со strength 0, positive/negative только 1..3")
        if not all(_nonempty(r.get(k)) for k in ("takeaway", "reason_to_believe")):
            raise ValueError("Нужны вывод и его основание")
        for field, label in (("evidence", "observation"), ("limitations", "reason")):
            if r.get(field) is None and field == "limitations":
                r[field] = []
            items = r[field]
            if not isinstance(items, list) or len(items) > 2:
                raise ValueError(f"{field}: до двух наблюдений")
            for entry in items:
                if not _nonempty(entry[label]) or not isinstance(entry["source_ids"], list) or not set(entry["source_ids"]) <= sources:
                    raise ValueError("Неизвестная ссылка на кадр или реплику либо отсутствует обоснование")


def _validate_audit(obj, ids):
    if obj.get("checked_images") != len(PANEL) * len(ids):
        raise ValueError("checked_images должен равняться числу оценщиков × число имиджей")
    if type(obj.get("main_idea_judge")) is not int or obj["main_idea_judge"] not in range(len(PANEL)):
        raise ValueError("Нужен индекс main_idea_judge от 0 до 4")
    if not isinstance(obj.get("issues"), list):
        raise ValueError("Нужен массив issues")
    for issue in obj["issues"]:
        if (type(issue.get("model_index")) is not int or issue["model_index"] not in range(len(PANEL))
                or issue["image_id"] not in ids or issue["type"] not in _ERROR_TYPES
                or not _nonempty(issue.get("reason")) or not _nonempty(issue.get("quote"))):
            raise ValueError("Замечанию нужны корректный model_index, image_id, type, reason, quote")


# Only cache definitions of public-to-the-model survey questions. No user materials, sessions or keys.
_MEANING_CACHE = {}
_MEANING_LOCK = threading.Lock()


def _meaning(client, model, vertical, questions):
    key = (str(client.base_url), model, vertical, json.dumps(questions, ensure_ascii=False, sort_keys=True))
    with _MEANING_LOCK:
        if key in _MEANING_CACHE:
            return _MEANING_CACHE[key]
    result = []
    for start in range(0, len(questions), BATCH_SIZE):
        group = questions[start:start+BATCH_SIZE]
        ids = [q["id"] for q in group]
        packet = {"category": vertical, "items": [dict(key=q["id"], **q) for q in group]}
        result.extend(_call(client, model, prompts.MEANINGS, packet, [], lambda v: _validate_meanings(v, ids), 4500)["items"])
    with _MEANING_LOCK:
        if len(_MEANING_CACHE) >= 64:
            _MEANING_CACHE.pop(next(iter(_MEANING_CACHE)))
        _MEANING_CACHE[key] = result
    return result


def _for_judges(fn):
    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = [pool.submit(copy_context().run, fn, i) for i in range(len(PANEL))]
        return [f.result() for f in futures]


def _evidence_time(refs, frames):
    for ref in refs:
        if ref in frames:
            return f"{float(frames[ref]):.1f}".replace(".", ",") + " с"
    return "Озвучка" if any(ref.startswith("T") for ref in refs) else ""


def _manager_text(text):
    # Evidence references are already presented with their actual reading time.
    text = re.sub(r"\b(?:V\d{2}|T\d+)\b(?:\s*[-–—,/]\s*\b(?:V\d{2}|T\d+)\b)*", "", text)
    text = re.sub(r"(?:\([,;:\s]*\)|\[[,;:\s]*\])", "", text)
    return re.sub(r" +([.,;:!?])", r"\1", re.sub(r" {2,}", " ", text)).strip()


def _display_row(q, drafts, flags, sources, frame_times):
    stats = score_image(drafts, flags, sources)
    target = stats["after"]
    # Never use an argument the checker rejected as the supporting explanation.
    flagged = {issue["model_index"] for issue in flags}
    candidates = [i for i in range(len(PANEL)) if stats["usable"][i] and i not in flagged
                  and ((target == 0 and stats["votes"][i] == 0) or stats["votes"][i] * target > 0)]
    candidates.sort(key=lambda i: abs(stats["votes"][i] - target))
    evidence, explanation, limitation = [], "", ""
    if candidates:
        chosen = drafts[candidates[0]]
        explanation = _manager_text(chosen["reason_to_believe"])
        evidence = [dict(observation=_manager_text(e["observation"]), source_ids=e["source_ids"],
                         time=_evidence_time(e["source_ids"], frame_times)) for e in chosen["evidence"]]
        limits = chosen["limitations"]
        if limits:
            limitation = _manager_text(limits[0]["reason"])
    if target == 0 and stats["before"] != 0:
        explanation = "Отдельные детали допускают такую ассоциацию, но подтверждений этой оценки недостаточно."
    if not explanation:
        explanation = "Для оценки нет достаточного обоснования по ролику."
    return dict(id=q["id"], label=q["statement"], group=q["group"], secondary_groups=q.get("secondary_groups", []), score=target,
                clarity=stats["clarity"], review=stats["review"], explanation=explanation,
                limitation=limitation, evidence=evidence)


def score_video(*, vertical, questions, evidence, api_base, api_key, on_progress=None):
    """Return a complete verified report, or raise; callbacks run only on the calling thread."""
    notify = on_progress or (lambda message: None)
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Список имиджей пуст или в нём есть повторы.")
    public_questions = [{k: q[k] for k in ("id", "statement", "variants") if k in q} for q in questions]
    sheets, frame_times, speech = evidence["sheets"], evidence["frame_times"], evidence["speech"]
    n_frames = len(frame_times)
    if not sheets or not 1 <= n_frames <= 40 or set(frame_times) != {f"V{i:02d}" for i in range(1, n_frames + 1)}:
        raise ValueError("Не удалось подготовить полный набор кадров.")
    if not all(isinstance(s, str) and s.startswith("data:image/") for s in sheets):
        raise ValueError("Неверный формат кадров.")
    if not isinstance(speech, list) or any(not _nonempty(t.get("id")) or not _nonempty(t.get("text")) for t in speech):
        raise ValueError("Неверный формат расшифровки.")
    if not api_key or urlsplit(api_base).scheme != "https":
        raise UVPServiceError("Подключение к сервису оценки не настроено.")
    sources = set(frame_times) | {r["id"] for r in speech}
    material = {"brand": "Авито", "category": vertical, "speech": speech,
                "material": f"Кадры в хронологическом порядке. В этом ролике доступны только V01..V{n_frames:02d} ({n_frames} шт.).",
                "transcript_limitations": evidence.get("transcript_limitations", "")}
    result = []
    with httpx.Client(base_url=api_base.rstrip("/") + "/", headers={"Authorization": "Bearer " + api_key},
                      timeout=httpx.Timeout(300, connect=30, write=90), follow_redirects=False) as client:
        notify("Смотрим ролик без подсказки о посыле.")
        blind = _for_judges(lambda i: _call(client, PANEL[i], _BLIND, material, sheets, _validate_blind, 1800))
        notify("Готовим вопросы бренд-замера для выбранной вертикали.")
        meanings = _for_judges(lambda i: _meaning(client, PANEL[i], vertical, public_questions))
        main_ideas = [p["main_idea"] for p in blind]
        leads = []
        for start in range(0, len(questions), BATCH_SIZE):
            group = questions[start:start+BATCH_SIZE]
            ids = [q["id"] for q in group]
            contracts = [[r for r in m if r["key"] in ids] for m in meanings]
            packets = [{**material, "questions": public_questions[start:start+BATCH_SIZE],
                        "blind_first_impression": blind[i], "perception_contract": contracts[i]} for i in range(len(PANEL))]
            notify(f"Оцениваем имиджи {start + 1}–{start + len(ids)} из {len(questions)}.")
            drafts = _for_judges(lambda i: _call(client, PANEL[i], prompts.EVALUATE, packets[i], sheets,
                                                 lambda v: _validate_eval(v, ids, sources), 6500))

            def audit():
                analyses = [dict(model_index=i, perception_contract=contracts[i], **d) for i, d in enumerate(drafts)]
                packet = dict(**material, questions=packets[0]["questions"], analyses=analyses, blind_main_ideas=main_ideas)
                return _call(client, CHECKER, _AUDIT, packet, sheets, lambda v: _validate_audit(v, ids), 6500)

            notify("Проверяем, подтверждает ли ролик выводы оценщиков.")
            review = audit()
            if review["issues"]:
                def repair(i):
                    issues = [x for x in review["issues"] if x["model_index"] == i]
                    if not issues:
                        return drafts[i]
                    subset = sorted({issue["image_id"] for issue in issues})
                    packet = dict(**packets[i], own_draft=drafts[i], logical_review=issues, repair_ids=subset)
                    revised = _call(client, PANEL[i], prompts.REPAIR, packet, sheets,
                                    lambda v: _validate_eval(v, subset, sources, repair_batch=ids), 6000)
                    fixes = {r["id"]: r for r in revised["images"]}
                    return dict(main_idea=drafts[i]["main_idea"], images=[fixes.get(r["id"], r) for r in drafts[i]["images"]])
                drafts = _for_judges(repair)
                review = audit()
            leads.append(review["main_idea_judge"])
            by_judge = [{r["id"]: r for r in draft["images"]} for draft in drafts]
            for q in group:
                rows = [j[q["id"]] for j in by_judge]
                flags = [f for f in review["issues"] if f["image_id"] == q["id"]]
                result.append(_display_row(q, rows, flags, sources, frame_times))
    lead = max(dict.fromkeys(leads), key=leads.count)
    return {"model": MODEL, "vertical": vertical, "main_idea": _manager_text(main_ideas[lead]), "images": result}
