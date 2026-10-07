"""Connect the archived UVP interface to existing current-model observations.

This diagnostic reads blind panel answers, never changes scoring, and never
generates new answers conditioned on the selected target.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .vertical_uvp import _answers, evaluate_uvp, make_target
from .latest_interpretation import has_person_name


def target_for_vertical(vertical: str, goods: str | None = None) -> dict | None:
    aliases = {"Goods": "Товары", "Auto": "Авто · C2C транзакция",
               "Авто": "Авто · C2C транзакция", "Services": "Услуги",
               "Realty": "Недвижимость", "Travel": "Путешествия", "Jobs": "Работа"}
    vertical = aliases.get(vertical, vertical)
    if vertical == "Товары" and goods is None:
        return None
    return make_target(vertical, goods)


def with_blind_answers(result: dict) -> dict:
    """Keep the first complete blind panel, as in the reference interface.

    Repeats are repeated measurements of the same personas, not new people.
    Export this small panel so saved results can still run UVP without video.
    """
    out = deepcopy(result)
    if out.get("blind_answers"):
        _answers(out)
        return out
    evidence = out.get("evidence") or out.get("metadata", {}).get("evidence") or {}
    for panel in evidence.get("panel", []):
        rows = [{"respondent_id": str(row["respondent_id"]),
                 "answer": str(row.get("main_message_summary", "")).strip()}
                for row in panel if isinstance(row, dict) and row.get("respondent_id")]
        if len(rows) == 30 and all(row["answer"] for row in rows):
            out["blind_answers"] = _answers({"blind_answers": rows})
            break
    return out


def main_idea(result: dict) -> str:
    candidates = [result.get("main_idea", "")]
    for run in result.get("independent_evidence", []):
        candidates.append(run.get("values", {}).get("main_claim", ""))
    evidence = result.get("evidence") or result.get("metadata", {}).get("evidence") or {}
    candidates.extend(row.get("main_claim", "") for row in evidence.get("fresh", []))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() and not has_person_name(candidate):
            return candidate.strip()
    return "Не удалось однозначно определить основную идею по сохранённому разбору."


def with_uvp(result: dict, target: dict, api_key: str, cache_root: Path) -> dict:
    out = with_blind_answers(result)
    try:
        out["vertical_uvp"] = evaluate_uvp(out, target, api_key, cache_root)
    except Exception:
        # Optional UVP failure must not discard the expensive completed scores.
        out["vertical_uvp"] = {"target": dict(target), "status": "error",
                               "source_sha": out.get("source_sha")}
    return out
