"""Protocol and replay checks; no Gemini requests and no proprietary fixtures in git."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from aipm3 import legacy_contracts as contracts
from aipm3 import objective_features as objective

LOCAL = Path.home() / "Documents/Work projects/Код/mesSage_streamlit"
RAW = Path.home() / "outputs/aipm_flash_legacy_prompts_20260907/raw_calls"


def local_namespace(filename, names, **extra):
    path = LOCAL / filename
    if not path.exists():
        pytest.skip("Local canonical source not installed (development-only parity check)")
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    env = {"np": np, "json": json, "Counter": Counter, **extra}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def sample_row(component, value=0):
    schema = objective.request_kwargs(component, "VIDEO")["response_format"]
    properties = schema["json_schema"]["schema"]["properties"]
    row = {}
    for name, spec in properties.items():
        if spec["type"] == "boolean":
            row[name] = bool(value)
        elif spec["type"] == "integer":
            row[name] = max(value, spec.get("minimum", 0))
        else:
            row[name] = spec.get("enum", ["sample transcript"])[0]
    return row


def test_contracts_are_verbatim_local_sources():
    a1 = local_namespace("genai_pretest_gemini.py", {
        "ANALYSIS_PROMPT", "FEATURES_JSON_SCHEMA", "GEMINI_MODEL",
        "PROD_FEATURES", "PROD_CAT_FEATURES", "PROD_MODEL_PARAMS", "MAX_BINARY_MB",
    })
    for key in a1.keys() - {"np", "json", "Counter", "__builtins__"}:
        assert getattr(contracts, "AIPM1_" + key) == a1[key], key
    a2 = local_namespace("aipm_model_v1.py", {
        "DECOMP_PROMPT", "INT_FEATURES", "BOOL_FEATURES", "ENUM_FEATURES",
        "ALL_FEATURES", "build_schema", "DECOMP_SCHEMA", "GEMINI_MODEL",
        "MAX_BINARY_MB", "N_ITERATIONS",
    })
    for key in a2.keys() - {"np", "json", "Counter", "__builtins__", "build_schema"}:
        assert getattr(contracts, "AIPM2_" + key) == a2[key], key


@pytest.mark.parametrize("component, count", [("aipm1", 3), ("aipm2", 2)])
def test_independent_requests_and_exact_iteration_count(monkeypatch, component, count):
    client = MagicMock()
    client.__enter__.return_value = client
    row = sample_row(component)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(row)))],
    )
    monkeypatch.setattr(objective, "OpenAI", lambda **_: client)
    _, rows = objective.extract_component(component=component, video_base64="VIDEO", api_key="test")
    assert len(rows) == count
    assert client.chat.completions.create.call_count == count
    expected = objective.request_kwargs(component, "VIDEO")
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs == expected
        assert len(call.kwargs["messages"]) == 1
        assert "request_token" not in json.dumps(call.kwargs)
        assert call.kwargs["model"] == "google/gemini-3.1-pro-preview"


def test_no_scoring_with_incomplete_iterations(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.side_effect = RuntimeError("transport failure")
    monkeypatch.setattr(objective, "OpenAI", lambda **_: client)
    monkeypatch.setattr(objective.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="iteration 1/2 failed"):
        objective.extract_component(component="aipm2", video_base64="VIDEO", api_key="test")


def test_aipm2_bankers_rounding_and_first_vote_tie():
    first, second = sample_row("aipm2", 0), sample_row("aipm2", 1)
    assert objective.aggregate_aipm2([first, second])["brand_logo_screen_seconds"] == 0
    assert objective.aggregate_aipm2([first, second])["jingle_present"] is False
    assert objective.aggregate_aipm2([second, first])["jingle_present"] is True


def test_video_gate_keeps_independent_component_inputs(tmp_path, monkeypatch):
    import imageio_ffmpeg
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test")
    monkeypatch.setattr(objective, "AIPM1_MAX_BINARY_MB", 1)
    monkeypatch.setattr(objective, "AIPM2_MAX_BINARY_MB", 0)
    run = MagicMock()
    monkeypatch.setattr(objective, "run_video_command", run)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    assert objective.prepare_legacy_video(source, tmp_path / "a1", "aipm1") == source
    assert objective.prepare_legacy_video(source, tmp_path / "a2", "aipm2") != source
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("-vf") + 1] == "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert source.read_bytes() == b"test"


def test_58_saved_response_sets_match_local_aggregation_and_requests():
    if not RAW.exists():
        pytest.skip("Private replay fixtures are local only")
    a2 = local_namespace("aipm_model_v1.py", {
        "INT_FEATURES", "BOOL_FEATURES", "ENUM_FEATURES", "ALL_FEATURES",
        "DECOMP_PROMPT", "build_schema", "DECOMP_SCHEMA", "GEMINI_MODEL",
        "video_payload_block", "call_gemini", "aggregate_iterations",
    })
    client = MagicMock()
    a1 = local_namespace("genai_pretest_gemini.py", {
        "ANALYSIS_PROMPT", "FEATURES_JSON_SCHEMA", "FEAT_NAMES_RU", "GEMINI_MODEL",
        "video_payload_block", "get_video_features",
    }, st=MagicMock(), openai_client=client)
    for index in range(58):
        first = [json.loads((RAW / "aipm1" / f"row_{index:02d}__run_{r}.json").read_text()) for r in [1, 2, 3]]
        second = [json.loads((RAW / "aipm2" / f"row_{index:02d}__run_{r}.json").read_text()) for r in [1, 2]]
        client.chat.completions.create.side_effect = [SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(row)))],
        ) for row in first]
        local_features, transcript = a1["get_video_features"]("VIDEO")
        assert objective.aggregate_aipm1(first) == {**local_features, "voiceover_text": transcript}
        assert client.chat.completions.create.call_args.kwargs == objective.request_kwargs("aipm1", "VIDEO")
        assert objective.aggregate_aipm2(second) == a2["aggregate_iterations"](second)
    a2["openai_client"] = client
    client.chat.completions.create.side_effect = [SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(second[0])))],
    )]
    a2["call_gemini"]("VIDEO", "Avito")
    assert client.chat.completions.create.call_args.kwargs == objective.request_kwargs("aipm2", "VIDEO")


def test_message_delivery_computations_are_unchanged():
    canonical = Path.home() / "outputs/model_delivery_package_20260824/message_delivery_model.py"
    if not canonical.exists():
        pytest.skip("Local MD source not installed")
    from aipm3 import message_delivery_runtime as md
    def functions(path):
        class CacheValidationMetadata(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == "call_json":
                    # Domain IDs are now checked before a response enters cache.
                    # All original model computations and request arguments stay exact.
                    node.keywords = [kw for kw in node.keywords if kw.arg != "expected_ids"]
                return node
        tree = CacheValidationMetadata().visit(ast.parse(path.read_text()))
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, ast.FunctionDef)}
    original, deployed = functions(canonical), functions(Path(md.__file__))
    runtime_only = {"load_api_key", "video_duration", "prepare_video", "call_json",
                    "extract_transcript", "make_nested_clip", "extract_recovery"}
    for name in original.keys() - runtime_only:
        assert original[name] == deployed[name], name


def test_frozen_aipm1_matches_local_training_and_scoring():
    import pandas as pd
    from catboost import CatBoostClassifier
    from aipm3.models import score_aipm1
    dataset = Path.home() / "Documents/Work projects/AI-Pretest models/v2_decomposition/final_dataset_v2.parquet"
    artifact = Path(__file__).resolve().parents[1] / "deployment_artifacts/aipm1_model.cbm"
    if not dataset.exists() or not artifact.exists():
        pytest.skip("Local-only model and dataset")
    data = pd.read_parquet(dataset).dropna(subset=["OPM_bucket"])
    frame = data[contracts.AIPM1_PROD_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    for column in contracts.AIPM1_PROD_CAT_FEATURES:
        frame[column] = frame[column].round().astype(int)
    model = CatBoostClassifier(**contracts.AIPM1_PROD_MODEL_PARAMS)
    model.fit(frame, data.OPM_bucket.astype(int), cat_features=contracts.AIPM1_PROD_CAT_FEATURES)
    frozen = CatBoostClassifier().load_model(str(artifact))
    np.testing.assert_array_equal(model.predict_proba(frame), frozen.predict_proba(frame))
    for index in range(58):
        if not RAW.exists():
            break
        rows = [json.loads((RAW / "aipm1" / f"row_{index:02d}__run_{r}.json").read_text()) for r in [1, 2, 3]]
        values = objective.aggregate_aipm1(rows)
        x = pd.DataFrame([{key: values[key] for key in contracts.AIPM1_PROD_FEATURES}])
        probability = model.predict_proba(x)[0]
        score = score_aipm1(values, frozen)
        np.testing.assert_array_equal(probability, score["probability"])
        assert score["creative_score"] == probability[1] * 0.5 + probability[2]


def test_md_bundle_and_predictions_match_local():
    import hashlib
    import joblib
    import pandas as pd
    from aipm3 import message_delivery_runtime as md
    from aipm3.models import score_message_delivery
    original = Path.home() / "outputs/model_delivery_package_20260824/message_delivery_model_bundle.joblib"
    deployed = Path(__file__).resolve().parents[1] / "deployment_artifacts/message_delivery_model_bundle.joblib"
    if not original.exists() or not deployed.exists():
        pytest.skip("Local-only model bundles")
    assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(deployed.read_bytes()).digest()
    bundle = joblib.load(original)
    panel = {}
    for feature, _, low, high, _ in md.ROUND4_FEATURES:
        for suffix, value in [("value", low), ("mean3", (low + high) / 2), ("agreement3", 2 / 3)]:
            panel[f"{feature}__r4_{suffix}"] = value
    recovery = {"p12__cluster_valid_mask_mae_smoothed": 0.1, "p12__cluster_valid_mask_consistency": 0.9}
    features = md.technical_features(panel, 2.0, recovery, bundle)
    x = pd.DataFrame([features])[bundle["feature_columns"]]
    expected = float(bundle["model"].predict(x)[0])
    result = score_message_delivery(features, joblib.load(deployed))
    assert result["raw_score"] == expected
    assert result["raw_class"] == (0 if expected < bundle["threshold_t0"] else 2 if expected >= bundle["threshold_t2"] else 1)


def test_whole_pipeline_routes_independent_features(tmp_path, monkeypatch):
    import pandas as pd
    from aipm3 import pipeline
    from aipm3.models import load_frozen_models
    artifacts_dir = Path(__file__).resolve().parents[1] / "deployment_artifacts"
    if not (artifacts_dir / "aipm1_model.cbm").exists() or not RAW.exists():
        pytest.skip("Local-only replay models")
    models = load_frozen_models(
        str(artifacts_dir / "aipm1_model.cbm"), str(artifacts_dir / "aipm2_model.cbm"),
        str(artifacts_dir / "message_delivery_model_bundle.joblib"),
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    routed = []
    def extract(component, source_video, output_dir, api_key):
        assert source_video == source
        routed.append(component)
        count = 3 if component == "aipm1" else 2
        rows = [json.loads((RAW / component / f"row_00__run_{r}.json").read_text()) for r in range(1, count + 1)]
        agg = objective.aggregate_aipm1 if component == "aipm1" else objective.aggregate_aipm2
        return agg(rows), rows, component + "-sha"
    panel = {}
    for feature, _, low, high, _ in pipeline.md_runtime.ROUND4_FEATURES:
        for suffix, value in [("value", low), ("mean3", (low + high) / 2), ("agreement3", 2 / 3)]:
            panel[f"{feature}__r4_{suffix}"] = value
    recovery = {
        "p12__cluster_valid_mask_mae_smoothed": 0.1,
        "p12__cluster_valid_mask_consistency": 0.9,
        "recovery_candidate": "Example idea",
        "condition_valid_posterior": {key: 0.9 for key in pipeline.md_runtime.RECOVERY_CONDITIONS},
    }
    monkeypatch.setattr(pipeline.md_runtime, "prepare_video", lambda *args: source)
    monkeypatch.setattr(pipeline.md_runtime, "video_duration", lambda *args: 30)
    monkeypatch.setattr(pipeline, "_legacy_extract", extract)
    monkeypatch.setattr(pipeline, "_message_delivery_extract", lambda *args: {
        "panel": panel, "recovery": recovery, "words_per_second": 2.0,
        "transcript_frame": pd.DataFrame([{"transcript": "Example"}]),
        "panel_frame": pd.DataFrame([{"respondent_id": f"p{i:02d}", "main_message_summary": "Example idea"}
                                     for i in range(1, 31)]),
    })
    result = pipeline.run_analysis(source_video=source, output_root=tmp_path / "out", api_key="test", models=models)
    assert sorted(routed) == ["aipm1", "aipm2"]
    assert len(result["objective_runs"]["aipm1"]) == 3
    assert len(result["objective_runs"]["aipm2"]) == 2
    assert result["component_video_sha"]["aipm1"] != result["component_video_sha"]["aipm2"]
    assert result["main_idea"] == "Example idea"
    assert result["interpretation"]["group_rows"]
    assert len(result["blind_answers"]) == 30
    assert result["source_sha"]
    # Replay the actual pipeline entry point from before the diagnostic change.
    import subprocess
    import types
    old_source = subprocess.run(
        ["git", "show", "f93868af0cfe4504f4011db3279c58bca835c405:aipm3/pipeline.py"],
        cwd=artifacts_dir.parent, capture_output=True,
    )
    assert old_source.returncode == 0
    old = types.ModuleType("aipm3._pre_review_pipeline")
    old.__package__ = "aipm3"
    exec(compile(old_source.stdout, "pre_review_pipeline.py", "exec"), old.__dict__)
    old._legacy_extract = pipeline._legacy_extract
    old._message_delivery_extract = pipeline._message_delivery_extract
    previous = old.run_analysis(source_video=source, output_root=tmp_path / "before", api_key="test", models=models)
    for key in ["aipm1", "aipm2", "message_delivery", "aipm3", "objective_features", "objective_runs",
                "message_delivery_business", "recovery_curve", "main_idea"]:
        assert result[key] == previous[key], key
    saved = next((tmp_path / "out").rglob("result.json"))
    assert json.loads(saved.read_text())["aipm3"] == result["aipm3"]
    # The Streamlit page renders exactly the same result and loads pinned models.
    from streamlit.testing.v1 import AppTest
    import artifacts
    monkeypatch.setattr(artifacts, "artifact_path", lambda name: str(artifacts_dir / name))
    page = Path(__file__).resolve().parents[1] / "app_pages/video_pretest.py"
    at = AppTest.from_file(str(page), default_timeout=20)
    at.secrets["VSELLM_API_KEY"] = "test-no-api-calls"
    at.session_state["aipm3_result"] = result
    at.run()
    assert not at.exception
    from aipm3.marketing_profile import build_marketing_profile
    assert len(at.get("plotly_chart")) == len(build_marketing_profile(result)["topics"])
    assert sum(item.value.count("к среднему") for item in at.markdown) == 4
    at.session_state["aipm3_result"] = {**result, "scoring_version": "previous"}
    at.run()
    assert not at.exception
    assert len(at.metric) == 0
    assert any("Обновлена нормировка" in warning.value for warning in at.warning)
