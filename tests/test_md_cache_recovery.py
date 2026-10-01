"""Bad model responses must not permanently poison a resumable MD cache."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from aipm3 import message_delivery_runtime as md


def panel_answers(ids):
    return [{"respondent_id": rid, "main_message_summary": "Главная идея",
             **{name: False if kind == "boolean" else low
                for name, kind, low, _, _ in md.ROUND4_FEATURES}} for rid in ids]


def install_client(monkeypatch, handler):
    class Client:
        def __enter__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"][0]["text"]
            payload = handler(prompt, kwargs)
            payload["request_token"] = prompt.split("REQUEST_TOKEN: ")[1].splitlines()[0]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    monkeypatch.setattr(md, "create_client", lambda _: Client())
    monkeypatch.setattr(md.time, "sleep", lambda _: None)


def request(tmp_path, task, schema, expected_ids=None):
    kwargs = dict(task_key=task, prompt="Original prompt", schema=schema,
                  cache_dir=tmp_path / "cache", secrets_path=tmp_path / "unused",
                  video_b64=None, video_sha="same-video", temperature=.2, fresh=False,
                  retries=1, expected_ids=expected_ids)
    token = hashlib.sha256(f"md-final-v1|same-video|{task}".encode()).hexdigest()[:16]
    prompt = f"Original prompt\n\nREQUEST_TOKEN: {token}\nВерни request_token дословно."
    path = md.cache_path(kwargs["cache_dir"], task, prompt, "same-video")
    return kwargs, path, token


def test_first_bad_panel_ids_error_second_valid_succeeds_without_losing_other_caches(tmp_path, monkeypatch):
    good_ids = ["p01", "p02", "p03"]
    bad = {"answers": panel_answers(["p01", "p02", "wrong-person"])}
    valid = {"answers": panel_answers(good_ids)}
    replies = iter([bad, valid])
    calls = []

    def handler(prompt, _):
        calls.append(prompt)
        return deepcopy(next(replies))

    install_client(monkeypatch, handler)
    kwargs, path, _ = request(tmp_path, "panel30_01", md.panel_schema(),
                              ("answers", "respondent_id", set(good_ids)))
    neighbor = tmp_path / "cache" / "other-valid-response.json"
    neighbor.parent.mkdir(parents=True)
    neighbor.write_text('{"task":"other","response":{"known":"good"}}')
    before = neighbor.read_bytes()
    with pytest.raises(RuntimeError, match="IDs mismatch"):
        md.call_json(**kwargs)
    assert not path.exists()
    result = md.call_json(**kwargs)
    assert [row["respondent_id"] for row in result["answers"]] == good_ids
    assert md.call_json(**kwargs) == result
    assert len(calls) == 2 and neighbor.read_bytes() == before


@pytest.mark.parametrize("stage", ["panel", "transcript", "recovery", "cluster"])
def test_old_poisoned_entry_is_quarantined_then_replaced_once(stage, tmp_path, monkeypatch):
    if stage == "panel":
        schema, ids = md.panel_schema(), ("answers", "respondent_id", {"p01", "p02", "p03"})
        good = {"answers": panel_answers(["p01", "p02", "p03"])}
        bad = deepcopy(good)
        bad["answers"][2]["respondent_id"] = "p02"
    elif stage == "transcript":
        schema, ids = md.TRANSCRIPT_SCHEMA, None
        good = {"transcript": "Авито", "has_speech": True}
        bad = {"transcript": None, "has_speech": True}
    elif stage == "recovery":
        schema, ids = md.RECOVERY_SCHEMA, ("answers", "respondent_id", {"p01"})
        good = {"answers": [{"respondent_id": "p01", "answer": "Идея", "brand_only": False, "no_idea": False}]}
        bad = deepcopy(good)
        bad["answers"][0]["respondent_id"] = "unknown"
    else:
        schema, ids = md.CLUSTER_SCHEMA, ("assignments", "respondent_uid", {"u01"})
        good = {"assignments": [{"respondent_uid": "u01", "answer_type": "valid", "canonical_idea": "Идея"}]}
        bad = deepcopy(good)
        bad["assignments"].append(deepcopy(bad["assignments"][0]))
    kwargs, path, token = request(tmp_path, stage, schema, ids)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task": stage, "response": {"request_token": token, **bad}}))
    original = path.read_bytes()
    calls = []

    def handler(prompt, _):
        calls.append(prompt)
        return deepcopy(good)

    install_client(monkeypatch, handler)
    result = md.call_json(**kwargs)
    assert result == {"request_token": token, **good}
    quarantined = list(path.parent.glob(path.name + ".invalid-*"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == original
    assert json.loads(path.read_text())["response"] == result
    assert md.call_json(**kwargs) == result and len(calls) == 1


def test_full_panel_resume_requests_only_failed_group(tmp_path, monkeypatch):
    calls = Counter()
    failing = True

    def handler(prompt, _):
        ids = re.findall(r"^- (p\d+):", prompt, flags=re.MULTILINE)
        assert len(ids) == 3
        calls[ids[0]] += 1
        payload = {"answers": panel_answers(ids)}
        if failing and ids[0] == "p01":
            payload["answers"][0]["respondent_id"] = "bad-id"
        return payload

    install_client(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="IDs mismatch"):
        md.extract_panel30("video", "sha", tmp_path / "panel", tmp_path / "unused", False)
    valid_paths = list((tmp_path / "panel" / "cache").glob("*.json"))
    assert len(valid_paths) == 9
    saved = {path: path.read_bytes() for path in valid_paths}
    assert calls["p01"] == 5 and all(count == 1 for key, count in calls.items() if key != "p01")
    failing = False
    frame, _ = md.extract_panel30("video", "sha", tmp_path / "panel", tmp_path / "unused", False)
    assert len(frame) == 30 and frame.respondent_id.nunique() == 30
    assert calls["p01"] == 6 and all(count == 1 for key, count in calls.items() if key != "p01")
    assert all(path.read_bytes() == raw for path, raw in saved.items())
