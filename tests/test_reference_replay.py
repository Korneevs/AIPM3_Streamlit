"""Replay the historical product inputs without Gemini calls or public data copies."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from aipm3.models import (
    AIPM1_INDEX_BY_CLASS, AIPM2_FEATURES, AIPM2_REFERENCE,
    AIPM3_PRODUCT_MEAN, AIPM3_Q33, AIPM3_Q67, MD_REFERENCE,
    aipm2_reference_index, aipm3_score, percentile_index, score_aipm2,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path.home() / "outputs/message_delivery_meeting_memo_20260825/analysis/combo_input.csv"
FEATURES = Path.home() / "Desktop/aipm_v4_validation_results.csv"


def reference():
    if not REFERENCE.exists():
        pytest.skip("Historical campaign data are private and local-only")
    return pd.read_csv(REFERENCE)


def normalized_key(value):
    value = re.sub(r"\.(avi|mp4|mov)$", "", str(value).lower().strip())
    return re.sub(r"[^a-zа-я0-9]", "", value.replace("с", "c").replace("ё", "е"))


def test_all_47_complete_campaigns_match_independent_local_product():
    data = reference()
    assert len(data) == 48
    np.testing.assert_array_equal(
        np.sort(data["AIPM_v2(adrecall)"].dropna()), np.sort(AIPM2_REFERENCE),
    )
    np.testing.assert_allclose(data.md_score_eval, MD_REFERENCE, atol=1e-14, rtol=0)
    # This is the independent historical pandas-rank formula, not an app helper.
    a1 = 0.5 + data["AIPM_v1(OPM)"].rank(method="average") / 49
    a2 = 0.5 + data["AIPM_v2(adrecall)"].rank(method="average") / 49
    md = 0.5 + data.md_score_eval.rank(method="average") / 49
    for actual, saved in [(a1, data.aipm1_index), (a2, data.aipm2_index), (md, data.md_index)]:
        np.testing.assert_allclose(actual, saved, atol=1e-14, rtol=0, equal_nan=True)
    product = a1 * a2 * md
    normalized = product / product.mean()
    assert np.isclose(product.mean(), AIPM3_PRODUCT_MEAN, atol=1e-14, rtol=0)
    np.testing.assert_allclose(normalized.quantile([1 / 3, 2 / 3]), [AIPM3_Q33, AIPM3_Q67], atol=1e-14, rtol=0)
    complete = data.dropna(subset=["AIPM_v2(adrecall)"])
    assert len(complete) == 47
    for index, row in complete.iterrows():
        result = aipm3_score(
            {"reference_index": AIPM1_INDEX_BY_CLASS[int(row["AIPM_v1(OPM)"])]},
            {"reference_index": aipm2_reference_index(row["AIPM_v2(adrecall)"])},
            {"reference_index": percentile_index(row.md_score_eval, MD_REFERENCE, 49)},
        )
        assert abs(result["index"] - normalized.loc[index]) < 1e-14
        expected_class = 0 if normalized.loc[index] < AIPM3_Q33 else 2 if normalized.loc[index] >= AIPM3_Q67 else 1
        assert result["level"] == expected_class


def test_recovered_nine_feature_model_reproduces_all_31_available_campaigns():
    data = reference()
    artifact = ROOT / "deployment_artifacts/aipm2_model.cbm"
    if not FEATURES.exists() or not artifact.exists():
        pytest.skip("Private feature cache and model are local-only")
    cached = pd.read_csv(FEATURES)
    data["join_key"] = data["_key"].map(normalized_key)
    cached["join_key"] = cached.filename.map(normalized_key)
    joined = cached.merge(data, on="join_key", validate="many_to_one").dropna(subset=["AIPM_v2(adrecall)"])
    assert len(joined) == joined.join_key.nunique() == 31
    model = CatBoostRegressor().load_model(str(artifact))
    assert model.feature_names_ == AIPM2_FEATURES
    # The old cache is sparse. Reproduce its missing=0 inference convention;
    # production extraction still requires every field of the full JSON schema.
    features = joined[AIPM2_FEATURES[1:]].fillna(0)
    results = [score_aipm2(row, model) for row in features.to_dict("records")]
    np.testing.assert_allclose([r["reference_score"] for r in results], joined["AIPM_v2(adrecall)"], atol=1e-14, rtol=0)
    np.testing.assert_allclose([r["reference_index"] for r in results], joined.aipm2_index, atol=1e-14, rtol=0)


def test_eleven_feature_alternative_does_not_reproduce_reference():
    data = reference()
    artifact = Path.home() / "Documents/Work projects/AI-Pretest models/v3_adrecall/aipm_v4_model.cbm"
    if not FEATURES.exists() or not artifact.exists():
        pytest.skip("Alternative model is local-only")
    cached = pd.read_csv(FEATURES)
    data["join_key"] = data["_key"].map(normalized_key)
    cached["join_key"] = cached.filename.map(normalized_key)
    joined = cached.merge(data, on="join_key", validate="many_to_one").dropna(subset=["AIPM_v2(adrecall)"])
    model = CatBoostRegressor().load_model(str(artifact))
    assert len(model.feature_names_) == 11
    joined["brand_mean_adrecall"] = 0.1358
    frame = joined[model.feature_names_].fillna(0)
    for index in model.get_cat_feature_indices():
        frame[model.feature_names_[index]] = frame[model.feature_names_[index]].astype(int)
    predictions = np.round(model.predict(frame), 4)
    assert np.isclose(predictions, joined["AIPM_v2(adrecall)"], atol=1e-12, rtol=0).sum() == 0
