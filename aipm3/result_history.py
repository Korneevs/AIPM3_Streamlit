"""Compact per-session comparison; no video bytes or independent ranking claims."""
from __future__ import annotations

from pathlib import Path

from .manager_explanation import build_manager_explanation


def remember_result(history: list[dict], result: dict, name: str, *, limit: int = 10) -> list[dict]:
    source_sha = result.get("source_sha")
    scoring_version = result.get("scoring_version")
    protocol_version = result.get("protocol_version")
    if not source_sha or not scoring_version or not protocol_version:
        return list(history)
    explanation = build_manager_explanation(result)
    labels = {row["key"]: row["label"] for row in explanation["components"]}
    entry = {
        "source_sha": source_sha, "scoring_version": scoring_version,
        "protocol_version": protocol_version, "name": Path(name).name,
        "index_100": float(result["aipm3"]["index_100"]),
        "components": {key: float(result[key]["percentile"])
                       for key in ("aipm1", "aipm2", "message_delivery")},
        "lowest": ", ".join(labels[key] for key in explanation["weakest_keys"]),
    }
    identity = (source_sha, scoring_version, protocol_version)
    retained = [row for row in history if
                (row["source_sha"], row["scoring_version"], row["protocol_version"]) != identity]
    return (retained + [entry])[-limit:]
