"""Frozen finished requests and isolated neuro measurements, without network calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from aipm3 import latest_pipeline as pipeline
from aipm3 import latest_runtime as runtime


# Captured from HEAD 1e783a25 before the neuromatics extraction change, with
# encoded="fake-video" and prepared_sha="fixed-prepared-sha". These cover the
# entire API contract: prompt, model, temperature, schema and media reference.
FINISHED_REQUEST_HASHES = {
    **{f"aipm1/call_{i:02d}.json": "db1a70fea529aa4b9859872c623acb7916dd4a4bf9a658c60fa1153b2b56cbcc"
       for i in range(1, 4)},
    **{f"aipm2/call_{i:02d}.json": "2f241aa163c723fd7f0414feef3015b12ad279ec9b9cbb45947dc5f3724f2eba"
       for i in range(1, 3)},
    "fresh.json": "6ecd070e153a1d5dbc3053432af05ee4f66c64f117852535715fdb6b1cc391b7",
    "panel/call_01.json": "1d1e3d129fd13822a17932b00106715255d14fd9dc7045252d9352ed6a6ea6bf",
    "panel/call_02.json": "9c16af28a18ee7acbc012c5be12fe9cb57f3ca91a85dad109088a8ae2c2648b0",
    "panel/call_03.json": "acc7511dfb328e509f6e5c8e8c5d32bff1fe04e2a2e6aa969ad1753b3e5ba84d",
    "panel/call_04.json": "a4bd2aef400663ef13b60870323f3f98711d5d4651a0d0d0d9f44bbf7fac71ef",
    "panel/call_05.json": "e5aad517443c4170a08db869797e5ca87ff9771c9bfea8f942caa1630de2d3d5",
    "panel/call_06.json": "c86c1c964d82d213315990ee3ac75062aa48e2ddbec7e999550f53826c9316b9",
    "panel/call_07.json": "2e6fc02ccf2393f61e0fa4f96039b57bcd308da2bb524b80d3a7eec506df83d3",
    "panel/call_08.json": "f2b7077a9b40483b3377ea7d26da5cc7270eeabc1d9e73903da32882b89b4691",
    "panel/call_09.json": "15da800834b9bd202b9eb34f42b3779808928cbe6e7f3b18238f92557315b5b3",
    "panel/call_10.json": "ee7e713ea31eadb5f608fe262cb5b2061d22584ebf342585e3a6676d6bba19b8",
}

# Calculated directly from the supplied prompt literals, independently of the
# application module, using the same fixed prepared-media identity as above.
NEURO_PROMPT_HASHES = {
    **{f"aipm2/call_{i:02d}.json": "2cd26893309d485da5a7422f6e9ca3344f5d20505f4ae51fa647fe960c9225e5"
       for i in range(1, 3)},
    "panel/call_01.json": "7bdf3e0a110602ec992b96f2dd129edbf3df47dafcb10a80c3f96e281cd41d78",
    "panel/call_02.json": "39595d75b8112db79f4a187ee12f6551ab9365d979024da7b84b80aad92d6eb2",
    "panel/call_03.json": "d9eec09d645bc3929695e0919ed86ce20846ee6e0037f6d521c6b26905c540a5",
    "panel/call_04.json": "45d4637bee271e304f5565a5c7352bc8fa3cc6b6d1c9366f9dcd39f9478d516a",
    "panel/call_05.json": "154957d7e545dc0492825a3db1e676c21a66488ceef41c7e1ef3afcf3344c3b6",
    "panel/call_06.json": "1420ee4bd8e63250ff16dcc5130aa8e32d0c2e8065e600ce757e26577863464f",
    "panel/call_07.json": "73b61d1e3d4733ade12225414c16950c396ad2ea855c9e773983ba036289cdcd",
    "panel/call_08.json": "7ca759c6a14bcaf5f7767d29ad303533f078c74407ca8dcecc395072e4a1fa56",
    "panel/call_09.json": "53e5ce320a3335cd05a6949bd04bd50259bb5b38fa5cc33e99acf19b5786908a",
    "panel/call_10.json": "ff40f59368d22cecd30f001fd0c700dcf3827e1b1cbe273f11af6fc603ff7b4b",
}

# Golden scores after the eight-input M refit and neuromatics-only mean policy.
# Finished scores and the complete extraction requests remain unchanged.
BASELINE_SCORES = {
    "finished": {
        "OPM": 0.044241491929948584, "Q": 0.005485168114534491,
        "message_delivery": 0.11613391631611504,
        "norm_ad_recall": 0.12398243990549947, "noticeability": 0.38095238095238093,
    },
    "neuromatics": {
        "OPM": 0.03716285322115681, "Q": 0.0034463431782574914,
        "message_delivery": 0.11613391631611504,
        "norm_ad_recall": 0.09273623738597897, "noticeability": 0.32,
    },
}


def _prompt(request):
    return request["messages"][0]["content"][0]["text"]


def _payload(request):
    """One complete, deterministic observation for each existing API schema."""
    properties = request["response_format"]["json_schema"]["schema"]["properties"]
    if "answers" in properties:
        prompt = _prompt(request)
        answers = []
        for ident in re.findall(r"^- (p\d{2}):", prompt, re.MULTILINE):
            answer = {"respondent_id": ident, "main_message_summary": "main claim"}
            answer.update({name: False if kind == "boolean" else low
                           for name, kind, low, _, _ in pipeline.md.ROUND4_FEATURES})
            answers.append(answer)
        return {"answers": answers,
                "request_token": prompt.split("REQUEST_TOKEN: ")[1].splitlines()[0]}
    if "main_claim" in properties:
        values = {name: 1 for name in pipeline.fresh_contract.FIELDS}
        values.update(main_claim="main claim",
                      evidence={name: "0–5 seconds" for name in pipeline.fresh_contract.FIELDS})
        return values
    values = {
        name: False if spec["type"] == "boolean" else max(0, spec.get("minimum", 0))
        if spec["type"] in {"integer", "number"} else spec.get("enum", ["text"])[0]
        for name, spec in properties.items()
    }
    if "total_video_duration_sec" in values:
        values["total_video_duration_sec"] = 15
    return values


def _capture_requests(monkeypatch, material_kind):
    captured = {}

    def request(kwargs, path, **options):
        assert options["allow_live"] is False
        captured[path.as_posix()] = deepcopy(kwargs)
        payload = _payload(kwargs)
        options["validate"](payload)
        return payload

    monkeypatch.setattr(pipeline, "_request", request)
    for component in ("aipm1", "aipm2"):
        pipeline._objective(component, "fake-video", "fixed-prepared-sha", Path(component),
                            "", False, material_kind=material_kind)
    pipeline._panel("fake-video", "fixed-prepared-sha", Path("panel"), "", False,
                    material_kind=material_kind)
    pipeline._fresh("fake-video", "fixed-prepared-sha", Path("fresh.json"), "", False)
    return captured


def test_finished_requests_match_complete_prechange_snapshot(monkeypatch):
    captured = _capture_requests(monkeypatch, "finished")
    fingerprints = {
        name: hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for name, request in captured.items()
    }
    assert fingerprints == FINISHED_REQUEST_HASHES


def test_neuromatics_changes_only_the_supplied_aipm2_and_panel_prompts(monkeypatch):
    finished = _capture_requests(monkeypatch, "finished")
    neuro = _capture_requests(monkeypatch, "neuromatics")
    assert set(neuro) == set(finished) == set(FINISHED_REQUEST_HASHES)
    for name, request in neuro.items():
        if name in NEURO_PROMPT_HASHES:
            assert hashlib.sha256(_prompt(request).encode()).hexdigest() == NEURO_PROMPT_HASHES[name]
            prompt_only = deepcopy(request)
            prompt_only["messages"][0]["content"][0]["text"] = _prompt(finished[name])
            assert prompt_only == finished[name]
        else:
            assert request == finished[name]


@pytest.mark.parametrize("first_kind", ["finished", "neuromatics"])
def test_same_video_has_separate_resumable_measurements_and_original_output_contract(
        tmp_path, monkeypatch, first_kind):
    import openai

    source = tmp_path / "source.mp4"
    source.write_bytes(b"one source used for both kinds")
    media = {}
    for stage in ("aipm1", "aipm2", "panel", "fresh"):
        media[stage] = tmp_path / f"prepared_{stage}.mp4"
        media[stage].write_bytes(f"different prepared bytes for {stage}".encode())
    monkeypatch.setattr(pipeline, "_physical", lambda *_: {
        "phys__audio_dynamic_range_db": 10., "phys__motion_mean": .04, "phys__duration": 15.})
    monkeypatch.setattr(pipeline, "_prepare_inputs", lambda *_: dict(media))
    monkeypatch.setattr(pipeline.md, "video_duration", lambda _: 15.)
    api_requests = []

    class LocalClient:
        def __init__(self, **_):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            api_requests.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(_payload(kwargs))))],
                model_dump=lambda **_: {"id": "local-response"},
            )

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(openai, "OpenAI", LocalClient)
    collector_calls = []

    # Deliberately has the original strict signature: new extraction must not
    # send material_kind or silently turn a valid collector into incomplete.
    def collector(*, source_video, output_root, api_key, allow_live, progress):
        collector_calls.append(output_root)
        return []

    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    roots = {
        "finished": cache / runtime.PROTOCOL_VERSION / sha,
        "neuromatics": cache / runtime.PROTOCOL_VERSION / "neuromatics-measurement-v3.7-20261005" / sha,
    }
    kinds = [first_kind, "neuromatics" if first_kind == "finished" else "finished"]
    results, previous_files = {}, {}

    def analyze(kind, allow_live):
        return pipeline.run_latest_analysis(
            source_video=source, output_root=cache, api_key="local-test" if allow_live else "",
            allow_live=allow_live, family="unseen-test-family", material_kind=kind,
            repeat_count=3, evidence_collector=collector,
        )

    for index, kind in enumerate(kinds, start=1):
        result = results[kind] = analyze(kind, True)
        assert len(api_requests) == index * 48
        assert collector_calls[-1] == roots[kind] / "independent_evidence"
        assert result["evidence_status"] == "complete"
        assert result["scores"] == pytest.approx(BASELINE_SCORES[kind], rel=1e-12, abs=1e-14)
        assert result["extraction"] == {
            "repeats": 3, "calls_per_repeat": 16, "independent_stage_caches": True,
            "transcript_and_recovery_used": False,
        }
        assert all(path.read_bytes() == content for path, content in previous_files.items())
        previous_files.update({path: path.read_bytes() for path in roots[kind].rglob("*.json")})
        stage_records = [path for path in roots[kind].rglob("*.json")
                         if "request_fingerprint" in json.loads(path.read_text())]
        assert len(stage_records) == 48
        for path in stage_records:
            stage = "fresh" if path.name == "fresh.json" else path.parent.name
            expected_media = media["panel"] if kind == "neuromatics" and stage == "fresh" else media[stage]
            record = json.loads(path.read_text())
            assert record["prepared_sha"] == hashlib.sha256(expected_media.read_bytes()).hexdigest()

    assert set(results["finished"]) == set(results["neuromatics"])
    assert set(results["finished"]["metadata"]) == set(results["neuromatics"]["metadata"])
    stage_snapshots = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in cache.rglob("*.json") if "request_fingerprint" in json.loads(path.read_text())
    }
    monkeypatch.setattr(openai, "OpenAI", lambda **_: pytest.fail("Offline resume attempted a new request"))
    for kind in kinds:
        resumed = analyze(kind, False)
        assert resumed == results[kind]
        assert runtime.validate_cached_result(resumed, material_kind=kind)["scores"] == results[kind]["scores"]
    assert len(api_requests) == 96
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == snapshot
               for path, snapshot in stage_snapshots.items())
