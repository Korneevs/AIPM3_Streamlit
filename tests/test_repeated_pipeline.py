"""Offline contracts for complete repeats, resumption and averaging order."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aipm3 import repeated_pipeline as repeat
from aipm3 import message_delivery_runtime as md
from aipm3.feature_profile import GROUPS
from aipm3.manual_celebrity import apply_celebrity
from aipm3.models import (
    AIPM1_INDEX_BY_CLASS, AIPM1_FEATURES, AIPM2_FEATURES,
    EXPECTED_ARTIFACT_SHA256, MD_REFERENCE, SCORING_VERSION,
    aipm2_reference_index, aipm3_score, percentile_index,
)
from aipm3.objective_features import PROTOCOL_VERSION
from aipm3.result_export import export_result


def child(number, source_sha="test-video", model_sha=None):
    """Different valid model outputs, with the original per-run product formula."""
    result = {
        "source_sha": source_sha, "protocol_version": PROTOCOL_VERSION,
        "scoring_version": SCORING_VERSION,
        "model_sha256": deepcopy(model_sha or EXPECTED_ARTIFACT_SHA256),
        "duration_seconds": 20, "main_idea": "Продать вещь на Авито",
        "prepared_video": "/private/prepared.mp4",
        "objective_runs": {"aipm1": [{"vote": i} for i in range(3)],
                           "aipm2": [{"vote": i} for i in range(2)]},
        "transcripts": [{"iteration": i, "transcript": "Авито", "word_count": 1} for i in range(3)],
        "message_delivery_business": {"audio_completeness": number % 4, "cta_clarity": bool(number % 2),
                                      "words_per_second": 2 + number / 10, "offer_condition_count": 1.0},
        "recovery_curve": {key: .7 for key in
                           ["full"] + [f"nested_{f}_m{m}" for f in (25, 50, 75) for m in (1, 2)]},
    }
    class1 = number % 3
    ref1 = AIPM1_INDEX_BY_CLASS[class1]
    raw2 = .05 + number / 100
    ref2 = aipm2_reference_index(raw2)
    rawmd = .2 + number / 8
    refmd = percentile_index(rawmd, MD_REFERENCE, 49)
    result["aipm1"] = dict(reference_index=ref1, percentile=100 * (ref1 - .5),
                           raw_class=class1, creative_score=class1 / 2,
                           probability=[float(i == class1) for i in range(3)],
                           feature_values={key: number % 2 for key in AIPM1_FEATURES})
    result["aipm2"] = dict(reference_index=ref2, percentile=100 * (ref2 - .5),
                           raw_score=raw2, reference_score=round(raw2, 4),
                           feature_values={key: number % 2 for key in AIPM2_FEATURES})
    result["message_delivery"] = dict(reference_index=refmd, percentile=100 * (refmd - .5),
                                      raw_score=rawmd, raw_class=0 if rawmd < .5 else 1,
                                      creative_score=rawmd / 2, thresholds={"class_0_below": .5, "class_2_from": 1.5},
                                      feature_values={"words_per_second": 2 + number / 10})
    for name in ("aipm1", "aipm2", "message_delivery"):
        result[name]["feature_effects"] = {
            key: (number - 5) / 10
            for fields in GROUPS[name][1].values() for key in fields
        }
    result["aipm3"] = aipm3_score(result["aipm1"], result["aipm2"], result["message_delivery"])
    result["blind_answers"] = [{"respondent_id": str(i), "answer": f"Продать вещь {number}"} for i in range(30)]
    result["diagnostic_panel"] = [{"respondent_id": str(i), "main_message_summary": f"Продать вещь {number}",
                                    "offer_novelty_explanation_need": 1, "irony_or_twist_dependency": 0}
                                   for i in range(30)]
    result["diagnostic_recovery"] = [{"respondent_id": str(i), "respondent_uid": f"{key}:{i}",
                                       "condition_group": key, "answer_type": "valid", "raw_answer": "Продать вещь"}
                                      for key in result["recovery_curve"] for i in range(12)]
    return result


@pytest.fixture
def results():
    return [child(i) for i in range(1, 11)]


@pytest.fixture
def execution(tmp_path, monkeypatch):
    source = tmp_path / "upload.mp4"
    source.write_bytes(b"the same original uploaded video")
    model = SimpleNamespace(artifact_sha256=deepcopy(EXPECTED_ARTIFACT_SHA256))
    calls = []

    def fake(**kwargs):
        root = kwargs["output_root"]
        number = int(root.name.split("_")[-1])
        calls.append(root)
        assert root.name == f"repeat_{number:02d}"
        return child(number, hashlib.sha256(kwargs["source_video"].read_bytes()).hexdigest(),
                     kwargs["models"].artifact_sha256)

    monkeypatch.setattr(repeat, "run_analysis", fake)
    return dict(source_video=source, output_root=tmp_path / "cache", api_key="never-used",
                models=model), calls, fake


def test_means_original_outputs_and_products_not_mean_features_or_classes(results):
    before = deepcopy(results)
    actual = repeat.aggregate_repeats(results)
    assert results == before
    assert actual["aipm3"]["index"] == pytest.approx(sum(r["aipm3"]["index"] for r in results) / 10)
    wrong = aipm3_score(actual["aipm1"], actual["aipm2"], actual["message_delivery"])["index"]
    assert abs(actual["aipm3"]["index"] - wrong) > .001
    for name in ("aipm1", "aipm2", "message_delivery"):
        assert actual[name]["reference_index"] == pytest.approx(sum(r[name]["reference_index"] for r in results) / 10)
        assert "raw_class" not in actual[name]
        for key in actual[name]["feature_effects"]:
            assert actual[name]["feature_effects"][key] == pytest.approx(sum(r[name]["feature_effects"][key] for r in results) / 10)
    assert sum(actual["aipm1"]["class_counts"].values()) == 10
    assert len(actual["diagnostic_panel"]) == 300
    assert len(actual["diagnostic_recovery"]) == 840
    assert len({r["respondent_id"] for r in actual["blind_answers"]}) == 300
    assert repeat.is_repeated_result(actual)


def test_identical_repeats_preserve_frozen_model_scores_exactly(results):
    original = results[0]
    actual = repeat.aggregate_repeats([deepcopy(original) for _ in range(10)])
    for name in ("aipm1", "aipm2", "message_delivery"):
        assert actual[name]["reference_index"] == pytest.approx(original[name]["reference_index"], abs=1e-15)
    assert actual["aipm3"]["index"] == pytest.approx(original["aipm3"]["index"], abs=1e-15)


@pytest.mark.parametrize("selection", ["fomenko", "zhuravlyov", "kurkova"])
def test_celebrity_applies_inside_each_repeat_before_averaging(results, selection):
    base = repeat.aggregate_repeats(results)
    actual = repeat.apply_repeated_celebrity(base, selection)
    expected = sum(apply_celebrity(r, selection)["aipm3"]["index"] for r in results) / 10
    assert actual["aipm3"]["index"] == pytest.approx(expected)
    assert actual["aipm3"]["index"] == pytest.approx(sum(row["aipm3"] for row in actual["repeat_scores"]) / 10)
    assert actual["aipm3"]["index"] == pytest.approx(base["aipm3"]["index"] * 1.3)
    assert repeat.apply_repeated_celebrity(actual, selection) == actual
    assert repeat.apply_repeated_celebrity(actual, "none") == base
    assert actual["repeat_results"] == results


@pytest.mark.parametrize("count", [0, 1, 9, 11])
def test_partial_or_extra_repeats_never_return_an_average(results, count):
    with pytest.raises(ValueError, match="10"):
        repeat.aggregate_repeats((results + results)[:count])


@pytest.mark.parametrize("damage", ["panel", "model", "source", "nan", "scoring"])
def test_invalid_repeat_rejects_entire_result(results, damage):
    bad = results[4]
    if damage == "panel":
        bad["diagnostic_panel"].pop()
    elif damage == "model":
        bad["model_sha256"] = {"other": "model"}
    elif damage == "source":
        bad["source_sha"] = "different-video"
    elif damage == "nan":
        bad["aipm2"]["feature_effects"]["jingle_present"] = float("nan")
    else:
        bad["scoring_version"] = "old"
    with pytest.raises((ValueError, TypeError)):
        repeat.aggregate_repeats(results)


def test_exactly_ten_distinct_caches_then_zero_paid_calls_on_resume(execution):
    kwargs, calls, _ = execution
    progress = []
    actual = repeat.run_repeated_analysis(**kwargs, progress=progress.append)
    assert len(calls) == len(set(calls)) == 10
    assert [p.name for p in calls] == [f"repeat_{i:02d}" for i in range(1, 11)]
    assert len(actual["repeat_cache"]) == 10
    assert all(Path(p).exists() for p in actual["repeat_cache"])
    assert any("10/10" in msg for msg in progress)
    assert repeat.run_repeated_analysis(**kwargs) == actual
    assert len(calls) == 10
    assert len(list(kwargs["output_root"].rglob("result.json"))) == 1


def test_failure_preserves_completed_and_partial_md_cache_and_resumes(execution, monkeypatch):
    kwargs, calls, fake = execution
    first = True
    partial_path = None

    def failing(**arguments):
        nonlocal first, partial_path
        root = arguments["output_root"]
        if root.name == "repeat_04":
            partial_path = root / "message_delivery" / "cache" / "finished-request.json"
            if first:
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text('{"response": "saved fourth-repeat stage"}')
                first = False
                raise RuntimeError("temporary provider failure")
            assert partial_path.read_text() == '{"response": "saved fourth-repeat stage"}'
        return fake(**arguments)

    monkeypatch.setattr(repeat, "run_analysis", failing)
    with pytest.raises(RuntimeError, match="4/10"):
        repeat.run_repeated_analysis(**kwargs)
    assert len(calls) == 3
    assert len(list(kwargs["output_root"].rglob("complete.json"))) == 3
    assert not list(kwargs["output_root"].rglob("result.json"))
    # A fresh upload path with identical content must resume the same cache.
    new_source = kwargs["source_video"].with_name("another-upload.mp4")
    new_source.write_bytes(kwargs["source_video"].read_bytes())
    actual = repeat.run_repeated_analysis(**{**kwargs, "source_video": new_source})
    assert actual["repeat_count"] == 10
    assert len(calls) == 10
    assert partial_path.exists()


def test_old_single_cache_is_not_accepted_and_corrupt_child_is_retried(execution):
    kwargs, calls, _ = execution
    sha = hashlib.sha256(kwargs["source_video"].read_bytes()).hexdigest()
    old = kwargs["output_root"] / PROTOCOL_VERSION / sha[:16] / "result.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(child(1, sha)))
    actual = repeat.run_repeated_analysis(**kwargs)
    assert len(calls) == 10
    broken = Path(actual["repeat_cache"][4])
    saved = json.loads(broken.read_text())
    saved["repeat"] = 99
    broken.write_text(json.dumps(saved))
    repeat.run_repeated_analysis(**kwargs)
    assert len(calls) == 11 and calls[-1].name == "repeat_05"
    assert json.loads(old.read_text()) == child(1, sha)


def test_model_contract_changes_cannot_reuse_previous_cache(execution):
    kwargs, calls, _ = execution
    repeat.run_repeated_analysis(**kwargs)
    kwargs["models"].artifact_sha256 = {**EXPECTED_ARTIFACT_SHA256, "aipm1": "new-model-hash"}
    repeat.run_repeated_analysis(**kwargs)
    assert len(calls) == 20


def test_uvp_keeps_ten_validated_panels_and_combines_300_unique_answers(results, monkeypatch, tmp_path):
    calls = []

    def fake(result, target, api_key, cache_root):
        assert len(result["blind_answers"]) == 30
        calls.append(result)
        return {"total": 30, "target": target, "answers": [
            {**row, "status": "matched", "quote": row["answer"]} for row in result["blind_answers"]]}

    monkeypatch.setattr(repeat, "evaluate_uvp", fake)
    actual = repeat.evaluate_repeated_uvp(repeat.aggregate_repeats(results), {"key": "test"}, "unused", tmp_path)
    assert len(calls) == 10 and actual["total"] == 300
    assert actual["counts"]["matched"] == 300
    assert len({r["respondent_id"] for r in actual["answers"]}) == 300


def test_actual_md_cache_is_isolated_between_repeats_and_resumable_within_one(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        def __enter__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][0]["content"][0]["text"]
            token = prompt.split("REQUEST_TOKEN: ")[1].splitlines()[0]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"request_token": token, "value": len(calls)})))])

    monkeypatch.setattr(md, "create_client", lambda _: FakeClient())
    values = []
    for n in range(1, 11):
        kwargs = dict(task_key="panel30_01", prompt="unchanged prompt", schema={},
                      cache_dir=tmp_path / f"repeat_{n:02d}" / "md" / "cache",
                      secrets_path=tmp_path / "unused", video_b64=None, video_sha="same-source",
                      temperature=.2, fresh=False)
        first = md.call_json(**kwargs)
        assert md.call_json(**kwargs) == first
        values.append(first["value"])
    assert values == list(range(1, 11)) and len(calls) == 10


def test_export_redacts_child_media_but_keeps_ten_scores(results):
    aggregated = repeat.aggregate_repeats(results)
    exported = export_result(aggregated)
    assert len(exported["repeat_results"]) == 10
    assert all("prepared_video" not in r and "objective_runs" not in r and "transcripts" not in r
               for r in exported["repeat_results"])
    assert exported["repeat_results"][0]["aipm3"] == results[0]["aipm3"]
    assert "prepared_video" in aggregated["repeat_results"][0]


def test_stale_single_or_wrong_repeat_version_cannot_be_rescored(results):
    assert not repeat.is_repeated_result(results[0])
    with pytest.raises(ValueError):
        repeat.apply_repeated_celebrity(results[0], "fomenko")
    result = repeat.aggregate_repeats(results)
    result["repetition_version"] = "old"
    assert not repeat.is_repeated_result(result)


def test_result_page_identifies_ten_runs_without_model_or_provider_calls(results):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    prefix = (root / "app_pages/video_pretest.py").read_text().split('\nst.title(')[0]
    app = AppTest.from_string(prefix + '\nshow_result(st.session_state["result"])\n')
    app.session_state["result"] = repeat.aggregate_repeats(results)
    app.run(timeout=30)
    assert not app.exception
    assert any("10 полным прогонам" in item.value for item in app.caption)


def test_app_rejects_old_single_session_result_without_provider_calls(results):
    from streamlit.testing.v1 import AppTest
    source = (Path(__file__).resolve().parents[1] / "app_pages/video_pretest.py").read_text()
    prefix = source.split('\nst.title(')[0]
    tail = 'if "aipm3_result" in st.session_state:' + source.split('if "aipm3_result" in st.session_state:')[1]
    app = AppTest.from_string(prefix + '\n' + tail)
    app.session_state["aipm3_result"] = results[0]
    app.run(timeout=30)
    assert not app.exception
    assert any("не содержит десяти прогонов" in item.value for item in app.warning)
