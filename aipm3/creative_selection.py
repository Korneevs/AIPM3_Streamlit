"""Two independent judges and one bounded argument-exchange round.

No shared cache contains briefs, scripts or judgments. A provider failure aborts
the run rather than being counted as an opposing vote or an empty selection.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timezone
import copy
import hashlib
import json
import re
import time
from urllib.parse import urlsplit

import httpx

from . import creative_prompts as prompts
from .runtime_resources import request_timeout


PROTOCOL_VERSION = "creative-exceptions-two-judges-v1"
DEFAULT_MODELS = ("openai/gpt-6.1-sol", "anthropic/claude-opus-5.5")
MAX_SCENARIOS = 60
MAX_SCENARIO_CHARS = 12000
MAX_TOTAL_CHARS = 180000
MAX_BRIEF_CHARS = 15000
BATCH_SIZE = 6
_FIELDS = ("idea", "quote", "mechanism", "brief_link", "risk", "minimal_fix", "reason", "decision_change_reason")
_VERDICTS = {"nominate", "reject", "insufficient"}


class CreativeServiceError(RuntimeError):
    """Safe error text; never includes a provider body, key or user material."""


def _string(value, *, required=False, limit=1200):
    return isinstance(value, str) and len(value) <= limit and (not required or bool(value.strip()))


def validate_inputs(brief, scenarios, quota, models):
    if not _string(brief, required=True, limit=MAX_BRIEF_CHARS):
        raise ValueError(f"Добавьте бриф длиной до {MAX_BRIEF_CHARS} символов.")
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise ValueError(f"Нужно от 1 до {MAX_SCENARIOS} сценариев.")
    if type(quota) is not int or not 1 <= quota <= MAX_SCENARIOS:
        raise ValueError(f"Ориентир просмотра должен быть от 1 до {MAX_SCENARIOS}.")
    ids = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Каждому сценарию нужны id, title и text.")
        if not all(_string(scenario.get(k), required=True, limit=limit) for k, limit in
                   (("id", 100), ("title", 300), ("text", MAX_SCENARIO_CHARS))):
            raise ValueError("Проверьте названия, идентификаторы и длину сценариев.")
        ids.append(scenario["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Идентификаторы сценариев должны быть уникальными.")
    if sum(len(s["text"]) + len(s["title"]) for s in scenarios) > MAX_TOTAL_CHARS:
        raise ValueError(f"Общий объём сценариев должен быть до {MAX_TOTAL_CHARS} символов.")
    if not isinstance(models, (tuple, list)) or len(models) != 2 or not all(
        _string(m, required=True, limit=200) and not re.search(r"\s", m) for m in models
    ):
        raise ValueError("Нужны две доступные модели разных семейств.")
    if not models[0].startswith("openai/") or not models[1].startswith(("anthropic/", "qwen/")):
        raise ValueError("Выберите GPT и отдельную модель Claude или Qwen.")


def _exact_rows(payload, field, expected_ids):
    rows = payload.get(field) if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("Верни ровно один объект для каждого запрошенного ID.")
    if any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows):
        raise ValueError("Каждая запись должна содержать строковый id.")
    if {r["id"] for r in rows} != set(expected_ids):
        raise ValueError("ID должны точно совпасть с запрошенными, без пропусков и повторов.")
    return rows


def _validate_opinions(payload, expected_ids, source, *, revision=False):
    rows = _exact_rows(payload, "items", expected_ids)
    for row in rows:
        if row.get("verdict") not in _VERDICTS:
            raise ValueError("verdict: только nominate, reject или insufficient.")
        for field in _FIELDS:
            required = field in ("idea", "reason") or (row["verdict"] == "nominate" and field != "decision_change_reason")
            if field == "decision_change_reason":
                required = revision
            if not _string(row.get(field), required=required, limit=900 if field == "quote" else 1200):
                raise ValueError("Все поля должны быть короткими строками; для номинации нужны все основания.")
        if row["quote"] and row["quote"] not in source[row["id"]]["text"]:
            raise ValueError("quote должна быть точной непрерывной цитатой из text данного сценария.")
        if not revision and row["decision_change_reason"]:
            raise ValueError("В независимом проходе decision_change_reason должна быть пустой.")
    return {row["id"]: {"id": row["id"], "verdict": row["verdict"], **{k: row[k] for k in _FIELDS}}
            for row in rows}


def _parse(content):
    if not isinstance(content, str):
        raise ValueError("Нужен полный JSON-объект.")
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip(), flags=re.I)
    try:
        value = json.loads(clean)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Нужен корректный JSON без текста вокруг.") from exc
    if not isinstance(value, dict):
        raise ValueError("Нужен JSON-объект.")
    return value


def _call(client, *, model, prompt, payload, validator, tokens=10000):
    """At most three requests, including transport and structural repairs."""
    correction = ""
    # The repair is structural only; never include raw provider errors or partial votes.
    for attempt in range(3):
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        if correction:
            messages.append({"role": "user", "content": correction})
        body = {"model": model, "messages": messages, "max_completion_tokens": tokens}
        if model.startswith("qwen/"):
            body["enable_thinking"] = False
        try:
            timeout = request_timeout(300)
            response = client.post("chat/completions", json=body,
                                   timeout=httpx.Timeout(timeout, connect=min(30, timeout), write=min(60, timeout)))
            if response.status_code >= 400:
                if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise CreativeServiceError("Одна из моделей недоступна. Повторите запуск позже; отбор не завершён.")
                time.sleep(attempt + 1)
                continue
            data = response.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Ответ неполный. Сократи объяснения и верни полный JSON.")
            parsed = _parse(choice["message"]["content"])
            return validator(parsed)
        except CreativeServiceError:
            raise
        except httpx.TransportError as exc:
            if attempt == 2:
                raise CreativeServiceError("Связь с одной из моделей прервалась. Отбор не завершён.") from exc
            time.sleep(attempt + 1)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            if attempt == 2:
                raise CreativeServiceError("Не удалось получить полный проверенный ответ двух моделей. Повторите запуск.") from exc
            # Only own schema messages are returned as guidance, not arbitrary exceptions.
            if prompt == prompts.PORTFOLIO:
                correction = ("Исправь только структуру JSON. groups — полное разбиение всех входных кандидатов без потерь и повторов; "
                              "details содержит каждый входной id один раз. Представители должны входить в свои группы. "
                              "Для дополнительных представителей нужны overflow_reason. Соблюдай ограничения длины полей; цитаты копируй точно из текста.")
            else:
                correction = "Исправь только структуру JSON, сохрани независимое решение. Верни все evaluate_ids и обязательные поля. Цитаты копируй точно из текста."
    raise CreativeServiceError("Отбор не завершён.")


def _parallel(fn):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(copy_context().run, fn, index) for index in range(2)]
        return [future.result() for future in futures]


def _status(initial, final):
    votes = [v["verdict"] for v in final]
    if votes == ["nominate", "nominate"]:
        return "agreed_initial" if all(v["verdict"] == "nominate" for v in initial) else "agreed_after_discussion"
    if votes == ["reject", "reject"]:
        return "rejected"
    if "insufficient" in votes:
        return "insufficient"
    return "disputed"


def _validate_portfolio(payload, ids, source, quota):
    groups = payload.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= len(ids):
        raise ValueError("Нужны непустые группы всех кандидатов.")
    seen = []
    cleaned_groups = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("Каждая группа должна быть объектом.")
        members = group.get("member_ids")
        if not isinstance(members, list) or not members or any(not isinstance(i, str) for i in members):
            raise ValueError("member_ids должен содержать исходные ID.")
        if group.get("representative_id") not in members:
            raise ValueError("Представитель должен принадлежать своей группе.")
        if not all(_string(group.get(k), required=True, limit=200) for k in ("shared_mechanism", "representative_reason")):
            raise ValueError("Объясни группировку и выбор представителя.")
        seen.extend(members)
        cleaned_groups.append({"id": f"G{index+1:03d}", **{k: group[k] for k in
                              ("member_ids", "representative_id", "shared_mechanism", "representative_reason")}})
    if len(seen) != len(ids) or set(seen) != set(ids):
        raise ValueError("Группы должны включать каждый кандидат ровно один раз.")
    details = _exact_rows(payload, "details", ids)
    extra = {g["representative_id"] for g in cleaned_groups[quota:]}
    cleaned_details = {}
    for row in details:
        if not _string(row.get("distinctiveness"), required=True, limit=350):
            raise ValueError("Для каждого кандидата объясни отличие или близость к представителю.")
        if not _string(row.get("overflow_reason"), required=row["id"] in extra, limit=350):
            raise ValueError("Для каждой дополнительной идеи обоснуй превышение ориентира.")
        quote = row.get("evidence_quote")
        if not _string(quote, required=True, limit=200) or quote not in source[row["id"]]["text"]:
            raise ValueError("Дай точную цитату из текста каждого кандидата.")
        cleaned_details[row["id"]] = {k: row[k] for k in ("distinctiveness", "overflow_reason", "evidence_quote")}
    return cleaned_groups, cleaned_details


def run_selection(*, brief, scenarios, quota, api_key, api_base, models=DEFAULT_MODELS, on_progress=None):
    """Return the complete audited selection, or raise without making up votes."""
    validate_inputs(brief, scenarios, quota, models)
    if not api_key or urlsplit(api_base).scheme != "https":
        raise CreativeServiceError("Подключение к моделям не настроено.")
    # Drop caller's additional fields (such as a readability score) before any model request.
    pack = [{k: s[k] for k in ("id", "title", "text")} for s in scenarios]
    source = {s["id"]: s for s in pack}
    ids = [s["id"] for s in pack]
    notify = on_progress or (lambda message: None)
    initial = [{}, {}]
    with httpx.Client(base_url=api_base.rstrip("/") + "/", headers={"Authorization": "Bearer " + api_key},
                      timeout=httpx.Timeout(300, connect=30, write=60), follow_redirects=False) as client:
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start+BATCH_SIZE]
            notify(f"Независимая оценка: сценарии {start+1}–{start+len(batch)} из {len(ids)}")
            payload = {"brief": brief, "scenarios": pack, "evaluate_ids": batch}
            outcomes = _parallel(lambda i: _call(client, model=models[i], prompt=prompts.INDEPENDENT,
                payload=payload, validator=lambda p: _validate_opinions(p, batch, source)))
            for index, outcome in enumerate(outcomes):
                initial[index].update(outcome)
        final = copy.deepcopy(initial)
        disagreements = [i for i in ids if initial[0][i]["verdict"] != initial[1][i]["verdict"]]
        for start in range(0, len(disagreements), BATCH_SIZE):
            batch = disagreements[start:start+BATCH_SIZE]
            notify(f"Сопоставляем аргументы: {start+1}–{start+len(batch)} из {len(disagreements)}")

            def reconsider(index):
                payload = {"brief": brief, "scenarios": pack, "evaluate_ids": batch,
                           "own_initial": [initial[index][i] for i in batch],
                           "other_initial": [initial[1-index][i] for i in batch]}
                return _call(client, model=models[index], prompt=prompts.RECONSIDER, payload=payload,
                             validator=lambda p: _validate_opinions(p, batch, source, revision=True))

            outcomes = _parallel(reconsider)
            for index, outcome in enumerate(outcomes):
                final[index].update(outcome)
        rows = []
        for scenario in pack:
            sid = scenario["id"]
            first = [initial[i][sid] for i in range(2)]
            last = [final[i][sid] for i in range(2)]
            status = _status(first, last)
            # The concise card keeps a nominating argument visible even in a dispute.
            summary = next((r for r in last if r["verdict"] == "nominate"), last[0])
            rows.append({**scenario, "status": status, **{k: summary[k] for k in _FIELDS},
                         "initial": first, "final": last, "discussion_happened": sid in disagreements,
                         "distinctiveness": "", "overflow_reason": "", "group_id": None,
                         "is_representative": False})
        confirmed = [r for r in rows if r["status"] in ("agreed_initial", "agreed_after_discussion")]
        groups, details = [], {}
        if confirmed:
            notify("Группируем близкие идеи и объясняем дополнительные кандидаты")
            confirmed_ids = [r["id"] for r in confirmed]
            portfolio_payload = {"brief": brief, "quota": quota,
                                 "candidates": [{"id": r["id"], "title": r["title"], "text": r["text"],
                                                 "judgments": r["final"]} for r in confirmed]}
            groups, details = _call(client, model=models[0], prompt=prompts.PORTFOLIO,
                payload=portfolio_payload, tokens=min(32000, 2500 + 650 * len(confirmed)),
                validator=lambda p: _validate_portfolio(p, confirmed_ids, source, quota))
        by_id = {row["id"]: row for row in rows}
        for group in groups:
            for sid in group["member_ids"]:
                by_id[sid].update(details[sid], group_id=group["id"],
                                 is_representative=sid == group["representative_id"])
    counts = {status: sum(row["status"] == status for row in rows) for status in
              ("agreed_initial", "agreed_after_discussion", "rejected", "disputed", "insufficient")}
    counts.update(total=len(rows), confirmed=len(confirmed), independent_ideas=len(groups),
                  additional=max(0, len(groups)-quota))
    identity = {"brief": brief, "scenarios": pack, "models": list(models), "protocol": PROTOCOL_VERSION}
    return {"protocol_version": PROTOCOL_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "input_sha256": hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "models": list(models), "quota": quota, "counts": counts, "items": rows, "groups": groups,
            "primary_ids": [g["representative_id"] for g in groups[:quota]],
            "additional_ids": [g["representative_id"] for g in groups[quota:]],
            "disputed_ids": [r["id"] for r in rows if r["status"] == "disputed"],
            "insufficient_ids": [r["id"] for r in rows if r["status"] == "insufficient"]}
