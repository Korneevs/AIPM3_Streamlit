"""Exact frozen-model parity and extraction/cache safety, with no live calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_pipeline as pipeline
from aipm3 import latest_runtime as runtime
from aipm3.objective_features import request_kwargs


RECOMMENDED = Path(__file__).resolve().parents[3] / "outputs/AIPM_effect15_20261001/recommended"


@pytest.fixture(scope="module")
def reference():
    if not RECOMMENDED.exists():
        pytest.skip("Private original recommended bundle is not installed")
    spec = importlib.util.spec_from_file_location("_latest_original_reference", RECOMMENDED / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AIPM3(RECOMMENDED / "models")


@pytest.fixture(scope="module")
def rows():
    if not RECOMMENDED.exists():
        pytest.skip("Private original campaign inputs are not installed")
    frames = {t: pd.read_csv(RECOMMENDED / "data" / f"campaign_inputs_{t}.csv",
                             float_precision="round_trip") for t in "nmr"}
    shared = set.intersection(*(set(d.record) for d in frames.values()))
    record = next(r for r in sorted(shared)
                  if all(len(d[d.record.eq(r)]) == 10 for d in frames.values()))
    return {t: d[d.record.eq(record)].sort_values("repeat").to_dict("records")
            for t, d in frames.items()}


def test_bundle_identity_matches_original(reference):
    for name, digest in runtime.artifact_hashes().items():
        assert hashlib.sha256((RECOMMENDED / name).read_bytes()).hexdigest() == digest
    assert [len(h.state["columns"]) for h in runtime.load_latest_models().heads.values()] == [9, 8, 7]


@pytest.mark.parametrize("dataset", ["human_inputs", "campaign_inputs", "alternative_inputs"])
def test_every_frozen_input_frame_matches_original(reference, dataset):
    model = runtime.load_latest_models()
    for task in "nr":
        frame = pd.read_csv(RECOMMENDED / "data" / f"{dataset}_{task}.csv", float_precision="round_trip")
        np.testing.assert_array_equal(model.heads[task].predict(frame), reference.heads[task].predict(frame))


def test_ten_aligned_repeats_and_mean_products_match_public_score(reference, rows):
    frames = {t: pd.DataFrame(v).sample(frac=1, random_state=42) for t, v in rows.items()}
    actual = runtime.score_feature_rows(frames, metadata={"source_name": "test.mp4"})
    current = runtime.load_latest_models()
    expected = current.score(*(frames[t] for t in "nmr")).iloc[0]
    for key in actual["scores"]:
        assert actual["scores"][key] == expected[key]
    repeats = pd.DataFrame(actual["per_repeat"])
    assert repeats.repeat.tolist() == list(range(1, 11))
    np.testing.assert_array_equal(repeats.Q, repeats.noticeability * repeats.message_delivery * repeats.norm_ad_recall)
    assert actual["metadata"]["source_name"] == "test.mp4"
    assert runtime.validate_cached_result(actual)["scores"] == actual["scores"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "misaligned", "family", "sha"])
def test_rejects_incomplete_or_mixed_repeat_rows(rows, bad):
    changed = deepcopy(rows)
    if bad == "missing":
        changed["n"].pop()
    elif bad == "duplicate":
        changed["n"][9]["repeat"] = 9
    elif bad == "misaligned":
        changed["m"][9]["repeat"] = 11
    elif bad == "family":
        for row in changed["r"]:
            row["family"] = "another-family"
    else:
        for task in "nmr":
            for row in changed[task]:
                row["sha"] = "different-video" if task == "r" else "original-video"
    with pytest.raises(ValueError):
        runtime.score_feature_rows(changed)


def test_cached_result_rejects_old_or_tampered_scores(rows):
    result = runtime.score_feature_rows(rows)
    result["scores"]["Q"] += .1
    with pytest.raises(ValueError, match="оценки"):
        runtime.validate_cached_result(result)
    result["scoring_version"] = "old-model"
    with pytest.raises(ValueError, match="версией"):
        runtime.validate_cached_result(result)


def test_known_video_uses_its_frozen_family_and_unknown_gets_new_family():
    mapping = runtime.known_video_families()
    sha, families = next((sha, families) for sha, families in mapping.items() if len(families) == 1)
    assert runtime.family_for_video(sha) == families[0]
    assert runtime.family_for_video("unseen-sha") == "upload_unseen-sha"


def test_interpretation_keeps_all_three_model_scores_and_exact_additivity(rows):
    from aipm3.latest_interpretation import build_latest_interpretation
    result = runtime.score_feature_rows(rows)
    interpreted = build_latest_interpretation(result)
    for card in interpreted["cards"]:
        score = result["scores"][runtime.SCORE_NAMES[card["task"]]]
        assert card["score"] == pytest.approx(score, abs=1e-12)
        assert interpreted["details"][card["task"]]["additivity_error"] < 1e-9


def test_bundle_loads_in_an_isolated_python_process(tmp_path):
    bundle = tmp_path / "portable"
    shutil.copytree(runtime.BUNDLE_DIR, bundle)
    script = """
import importlib.util, sys
from pathlib import Path
import pandas as pd
p=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('portable_frozen_model',p/'model.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
model=m.AIPM3(p/'models')
for task in 'nmr':
    frame=pd.read_csv(p/'data'/('fit_'+task+'.csv'),float_precision='round_trip')
    assert len(model.heads[task].predict(frame))==len(frame)
print('portable ok')
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(bundle)],
                               cwd=tmp_path, text=True, capture_output=True, timeout=90, check=True)
    assert completed.stdout.strip() == "portable ok"


def _sample(schema):
    props = schema["json_schema"]["schema"]["properties"]
    return {name: False if spec["type"] == "boolean" else max(0, spec.get("minimum", 0))
            if spec["type"] in {"integer", "number"} else spec.get("enum", ["text"])[0]
            for name, spec in props.items()}


def test_request_cache_never_dispatches_without_live_permission(tmp_path, monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **_: pytest.fail("Live request attempted"))
    kwargs = request_kwargs("aipm1", "video")
    with pytest.raises(pipeline.MissingMeasurement):
        pipeline._request(kwargs, tmp_path / "missing.json", video_sha="sha", api_key="",
                          allow_live=False)


def test_request_cache_reuses_exact_contract_and_rejects_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import openai
    payload = _sample(request_kwargs("aipm1", "video")["response_format"])
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                               model_dump=lambda **_: {"id": "mock"})

    class Client:
        def __init__(self, **_):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            return response
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    monkeypatch.setattr(openai, "OpenAI", Client)
    kwargs = request_kwargs("aipm1", "video")
    path = tmp_path / "observation.json"
    expected = pipeline._request(kwargs, path, video_sha="sha", api_key="mock", allow_live=True)
    before = path.read_bytes()
    assert pipeline._request(kwargs, path, video_sha="sha", api_key="", allow_live=False) == expected
    assert path.read_bytes() == before
    assert len(calls) == 1
    with pytest.raises(ValueError, match="contract mismatch"):
        pipeline._request(kwargs, path, video_sha="other", api_key="", allow_live=False)


@pytest.mark.parametrize("material_kind", ["finished", "neuromatics"])
@pytest.mark.parametrize("repeat_count", [3, 5])
def test_short_runs_reuse_ten_run_stage_paths_without_overwriting_results(tmp_path, monkeypatch, material_kind, repeat_count):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"test video")
    monkeypatch.setattr(pipeline, "_physical", lambda *_: {
        "phys__audio_dynamic_range_db": 10., "phys__motion_mean": .04, "phys__duration": 15.})
    monkeypatch.setattr(pipeline, "_prepare_inputs", lambda *_: {k: source for k in ["aipm1", "aipm2", "panel", "fresh"]})
    monkeypatch.setattr(pipeline.md, "video_duration", lambda _: 15.)
    requests = []

    def request(kwargs, path, **options):
        assert options["allow_live"] is False
        requests.append(path)
        if path.parent.name in {"aipm1", "aipm2"}:
            payload = _sample(kwargs["response_format"])
            if "total_video_duration_sec" in payload:
                payload["total_video_duration_sec"] = 15
        elif path.parent.name == "panel":
            call = int(path.stem.split("_")[1])
            answers = []
            for ident, _ in pipeline.md.PANEL30_PERSONAS[(call - 1) * 3:call * 3]:
                answer = {"respondent_id": ident, "main_message_summary": "main claim"}
                answer.update({name: False if kind == "boolean" else low
                               for name, kind, low, high, rule in pipeline.md.ROUND4_FEATURES})
                answers.append(answer)
            payload = {"answers": answers, "request_token": kwargs["messages"][0]["content"][0]["text"].split("REQUEST_TOKEN: ")[1].splitlines()[0]}
        else:
            payload = {key: 1 for key in pipeline.fresh_contract.FIELDS}
            payload.update(main_claim="main claim", evidence={key: "0–5 seconds" for key in pipeline.fresh_contract.FIELDS})
        options["validate"](payload)
        return payload

    monkeypatch.setattr(pipeline, "_request", request)
    progress = []
    result = pipeline.run_latest_analysis(source_video=source, output_root=tmp_path / "cache",
                                         material_kind=material_kind, progress=progress.append,
                                         repeat_count=repeat_count)
    assert len(requests) == len(set(requests)) == 16 * repeat_count
    assert len(result["per_repeat"]) == result["repeat_count"] == result["extraction"]["repeats"] == repeat_count
    assert any(f"{repeat_count}/{repeat_count}" in message for message in progress)
    assert result["extraction"]["transcript_and_recovery_used"] is False
    for repeat in range(1, repeat_count + 1):
        paths = [p for p in requests if f"repeat_{repeat:02d}" in p.parts]
        assert len(paths) == 16
        assert sum(p.parent.name == "aipm1" for p in paths) == 3
        assert sum(p.parent.name == "aipm2" for p in paths) == 2
        assert sum(p.parent.name == "panel" for p in paths) == 10
    short_paths = set(requests)
    short_results = {p: p.read_bytes() for p in (tmp_path / "cache").rglob("result_*.json")}
    assert len(short_results) == 1
    requests.clear()
    long_result = pipeline.run_latest_analysis(source_video=source, output_root=tmp_path / "cache",
                                              material_kind=material_kind, repeat_count=10)
    assert len(requests) == len(set(requests)) == 160
    assert short_paths.issubset(set(requests))
    assert len(long_result["per_repeat"]) == long_result["repeat_count"] == 10
    assert len(list((tmp_path / "cache").rglob("result_*.json"))) == 2
    assert all(path.read_bytes() == data for path, data in short_results.items())
    assert runtime.validate_cached_result(result)["scores"] == result["scores"]
    assert runtime.validate_cached_result(long_result)["scores"] == long_result["scores"]


@pytest.mark.parametrize("material_kind", ["finished", "neuromatics"])
def test_parallel_measurements_preserve_sequential_aggregation_and_bound_payloads(tmp_path, monkeypatch, material_kind):
    from aipm3 import runtime_resources
    source = tmp_path / "video.mp4"
    source.write_bytes(b"prepared video")
    prepared = {k: source for k in ("aipm1", "aipm2", "panel", "fresh")}
    hashes = {k: "sha" for k in prepared}
    owner = threading.get_ident()
    counts = {"active": 0, "peak": 0}
    lock = threading.Lock()
    parallel = [False]

    def request(kwargs, path, **options):
        if parallel[0]:
            assert threading.get_ident() != owner
            assert runtime_resources.request_timeout() == pytest.approx(120)
            with lock:
                counts["active"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
            time.sleep(.01)
        number = 1 if path.name == "fresh.json" else int(path.stem.rsplit("_", 1)[1])
        if path.parent.name in {"aipm1", "aipm2"}:
            payload = _sample(kwargs["response_format"])
            # The order of equal votes must remain the call order, even if
            # completions arrive in a different order.
            for key, spec in kwargs["response_format"]["json_schema"]["schema"]["properties"].items():
                if spec.get("type") == "boolean":
                    payload[key] = number == 1
            if path.parent.name == "aipm1":
                payload["voiceover_text"] = "claim " * number
                payload["unique_offer_count"] = number
        elif path.parent.name == "panel":
            token = kwargs["messages"][0]["content"][0]["text"].split("REQUEST_TOKEN: ")[1].splitlines()[0]
            answers = []
            for ident, _ in reversed(pipeline.md.PANEL30_PERSONAS[(number - 1) * 3:number * 3]):
                answer = dict(respondent_id=ident, main_message_summary=f"claim {number}")
                answer.update({name: False if kind == "boolean" else low
                               for name, kind, low, high, rule in pipeline.md.ROUND4_FEATURES})
                answers.append(answer)
            payload = dict(request_token=token, answers=answers)
        else:
            payload = {key: 1 for key in pipeline.fresh_contract.FIELDS}
            payload.update(main_claim="claim", evidence={key: "observation" for key in pipeline.fresh_contract.FIELDS})
        options["validate"](payload)
        if parallel[0]:
            with lock:
                counts["active"] -= 1
        return payload

    monkeypatch.setattr(pipeline, "_request", request)
    objective, runs = {}, {}
    for kind in ("aipm1", "aipm2"):
        objective[kind], runs[kind] = pipeline._objective(kind, "video", "sha", tmp_path / kind,
            "", False, material_kind=material_kind)
    panel = pipeline._panel("video", "sha", tmp_path / "panel", "", False, material_kind=material_kind)
    fresh = pipeline._fresh("video", "sha", tmp_path / "fresh.json", "", False)
    original_read = Path.read_bytes

    def read_in_worker(path):
        assert threading.get_ident() != owner, "Do not encode videos before submitting bounded work"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_in_worker)
    parallel[0] = True
    progress = []
    def observe(message):
        assert threading.get_ident() == owner, "UI progress must run on the Streamlit thread"
        progress.append(message)
    with runtime_resources.analysis_deadline():
        actual = pipeline._measure_repeat(prepared, hashes, tmp_path, "", False, material_kind, 1, 5, observe)
    assert actual == (objective, runs, panel, fresh)
    assert counts == {"active": 0, "peak": 2}
    assert len(progress) == 16 and progress[-1] == "Повтор 1/5: выполнено проверок 16/16"


@pytest.mark.parametrize("first_fails", [False, True])
def test_shared_queue_keeps_multiple_users_in_order_through_final_checks(tmp_path, monkeypatch, first_fails):
    from concurrent.futures import ThreadPoolExecutor
    from aipm3 import runtime_resources
    sources = [tmp_path / f"user_{i}.mp4" for i in range(4)]
    for i, source in enumerate(sources):
        source.write_bytes(f"video {i}".encode())
    events = []
    queued = [threading.Event() for _ in sources]
    final_checks = [threading.Event() for _ in sources]
    released = [threading.Event() for _ in sources]
    advanced = threading.Event()
    positions = [[] for _ in sources]

    def physical(source, _):
        assert runtime_resources._ANALYSIS_LOCK.locked()
        events.append(("analysis", int(source.stem.rsplit("_", 1)[1])))
        return {"phys__duration": 15.}

    monkeypatch.setattr(pipeline, "_physical", physical)
    monkeypatch.setattr(pipeline, "_prepare_inputs", lambda source, _: {
        key: source for key in ("aipm1", "aipm2", "panel", "fresh")})
    monkeypatch.setattr(pipeline, "_measure_repeat", lambda *args: ({}, {"aipm1": [], "aipm2": []}, [], {}))
    monkeypatch.setattr(pipeline, "rows_from_measurements", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "score_feature_rows", lambda *args, **kwargs: {"scores": {"Q": .1}})
    monkeypatch.setattr(pipeline.md, "video_duration", lambda _: 15.)

    def job(number):
        def waiting(position):
            positions[number].append(position)
            if position:
                queued[number].set()
            if number == 2 and position == 1:
                advanced.set()
        def finish(result):
            assert runtime_resources._ANALYSIS_LOCK.locked()
            final_checks[number].set()
            if number < 2:
                assert released[number].wait(5)
            events.append(("complete", number))
            if number == 0 and first_fails:
                raise ValueError("final check failed")
            return dict(result, checked=True)
        return pipeline.run_latest_analysis(source_video=sources[number], output_root=tmp_path / "cache",
            family=f"user_{number}", record=f"user_{number}",
            material_kind="finished" if number % 2 == 0 else "neuromatics",
            on_queue=waiting, postprocess=finish)

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(job, 0)]
        try:
            if not final_checks[0].wait(2):
                jobs[0].result(timeout=1)
                pytest.fail("First job did not reach its final checks")
            for number in range(1, 4):
                jobs.append(pool.submit(job, number))
                assert queued[number].wait(2)
                assert positions[number][0] == number
            assert events == [("analysis", 0)]
            released[0].set()
            assert final_checks[1].wait(2)
            assert advanced.wait(2)
            assert not final_checks[2].is_set()
            released[1].set()
            for number, future in enumerate(jobs):
                if number == 0 and first_fails:
                    with pytest.raises(ValueError, match="final check failed"):
                        future.result(timeout=3)
                else:
                    assert future.result(timeout=3)["checked"] is True
        finally:
            for event in released:
                event.set()
    assert events == [(phase, number) for number in range(4) for phase in ("analysis", "complete")]
    assert all(p[-1] == 0 for p in positions)
    assert not runtime_resources._ANALYSIS_QUEUE and not runtime_resources._ANALYSIS_LOCK.locked()


def test_prepared_media_resume_uses_saved_bytes_even_for_renamed_upload(tmp_path, monkeypatch):
    first, second = tmp_path / "one.mp4", tmp_path / "renamed.mov"
    first.write_bytes(b"source")
    second.write_bytes(first.read_bytes())
    root = tmp_path / "cache"
    fresh = root / "prepared/fresh/video.mp4"
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"prepared")
    monkeypatch.setattr(pipeline, "prepare_legacy_video", lambda source, *_: source)
    monkeypatch.setattr(pipeline.md, "prepare_video", lambda source, *_: source)
    pipeline._prepare_inputs(first, root)
    monkeypatch.setattr(pipeline, "prepare_legacy_video", lambda *_: pytest.fail("Reencoded cached media"))
    monkeypatch.setattr(pipeline.md, "prepare_video", lambda *_: pytest.fail("Reencoded cached media"))
    actual = pipeline._prepare_inputs(second, root)
    assert actual["aipm1"] == second
    assert actual["fresh"] == fresh
    fresh.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed or is missing"):
        pipeline._prepare_inputs(second, root)
