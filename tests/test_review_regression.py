"""Compare against the published version before diagnostics, without any API calls."""
import json
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
import pytest

from aipm3 import models, objective_features

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "f93868af0cfe4504f4011db3279c58bca835c405"
RAW = Path.home() / "outputs/aipm_flash_legacy_prompts_20260907/raw_calls"


def before_source(path):
    result = subprocess.run(["git", "show", f"{BEFORE}:{path}"], cwd=ROOT, capture_output=True)
    if result.returncode:
        pytest.skip("Pre-review git history unavailable")
    return result.stdout


@pytest.fixture(scope="module")
def baseline():
    source = before_source("aipm3/models.py")
    module = types.ModuleType("aipm3._pre_review_models")
    module.__package__ = "aipm3"
    sys.modules[module.__name__] = module
    exec(compile(source, "pre_review_models.py", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def frozen():
    directory = ROOT / "deployment_artifacts"
    if not (directory / "aipm1_model.cbm").exists():
        pytest.skip("Private frozen weights available locally only")
    return models.load_frozen_models(
        str(directory / "aipm1_model.cbm"), str(directory / "aipm2_model.cbm"),
        str(directory / "message_delivery_model_bundle.joblib"),
    )


def test_models_prompts_and_normalization_are_byte_identical():
    for path in ["aipm3/models.py", "aipm3/legacy_contracts.py", "artifacts.py"]:
        assert (ROOT / path).read_bytes() == before_source(path), path


def test_extraction_semantics_unchanged_outside_explicit_memory_fixes():
    import ast
    for path, allowed in {
        "aipm3/objective_features.py": {"prepare_legacy_video"},
        "aipm3/message_delivery_runtime.py": {
            "video_duration", "prepare_video", "call_json", "extract_transcript",
            "make_nested_clip", "extract_recovery",
        },
    }.items():
        def functions(source):
            return {n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        before, after = functions(before_source(path)), functions((ROOT / path).read_bytes())
        assert before.keys() == after.keys()
        for name in before.keys() - allowed:
            assert before[name] == after[name], (path, name)


def test_aipm1_and_aipm2_58_saved_inputs_match_pre_review(baseline, frozen):
    if not RAW.exists():
        pytest.skip("Private saved responses available locally only")
    for index in range(58):
        for component, count, aggregate in [
            ("aipm1", 3, objective_features.aggregate_aipm1),
            ("aipm2", 2, objective_features.aggregate_aipm2),
        ]:
            runs = [json.loads((RAW / component / f"row_{index:02d}__run_{r}.json").read_text())
                    for r in range(1, count + 1)]
            values = aggregate(runs)
            model = getattr(frozen, component)
            before = getattr(baseline, "score_" + component)(values, model)
            after = getattr(models, "score_" + component)(values, model)
            # Entire score dict: features, probabilities, score, rank, class, SHAP.
            assert after == before, (index, component)


def test_md_20_saved_inputs_match_pre_review(baseline, frozen):
    path = Path.home() / "outputs/message_delivery_panel30_repro_20260819/final_model_feature_values_runs.csv"
    if not path.exists():
        pytest.skip("Private saved MD features available locally only")
    data = pd.read_csv(path)
    assert len(data) == 20
    for values in data[frozen.message_delivery_bundle["feature_columns"]].to_dict("records"):
        assert models.score_message_delivery(values, frozen.message_delivery_bundle) == baseline.score_message_delivery(
            values, frozen.message_delivery_bundle
        )


def test_aipm3_47_reference_products_match_pre_review(baseline):
    path = Path.home() / "outputs/message_delivery_meeting_memo_20260825/analysis/combo_input.csv"
    if not path.exists():
        pytest.skip("Private reference values available locally only")
    data = pd.read_csv(path).dropna(subset=["AIPM_v2(adrecall)"])
    assert len(data) == 47
    for row in data.to_dict("records"):
        inputs = [{"reference_index": row[column]} for column in ["aipm1_index", "aipm2_index", "md_index"]]
        assert models.aipm3_score(*inputs) == baseline.aipm3_score(*inputs)
