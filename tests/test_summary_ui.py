"""Reference-mean display, unchanged classification, and result-page regression."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aipm3.models import SCORING_VERSION, EXPECTED_ARTIFACT_SHA256
from aipm3.feature_profile import GROUPS
from aipm3.summary_ui import REFERENCE_MEANS, metric_summaries


def sample_result():
    result = {
        "aipm3": {"index": 0.5405440947585606, "level": 0},
        "main_idea": "Основная идея тестового ролика",
        "scoring_version": SCORING_VERSION,
        "model_sha256": deepcopy(EXPECTED_ARTIFACT_SHA256),
    }
    for name, index in [("aipm1", 1.0510204081632653), ("aipm2", 0.8775510204081632653),
                        ("message_delivery", 0.5714285714285714)]:
        result[name] = {"reference_index": index, "percentile": 100 * (index - .5), "feature_effects": {feature: 0.0 for features in GROUPS[name][1].values() for feature in features}}
    return result


def test_screenshot_example_uses_relative_indices_not_percentile_ratios():
    result = sample_result()
    before = deepcopy(result)
    cards = metric_summaries(result)
    assert [round(card["delta"]) for card in cards] == [-46, 5, -11, -43]
    assert [card["level"] for card in cards] == [0, 1, 1, 0]
    assert result == before


@pytest.mark.parametrize("percentile,level", [(32.99, 0), (33, 1), (66.99, 1), (67, 2)])
def test_color_preserves_unrounded_existing_class_boundaries(percentile, level):
    result = sample_result()
    for name in REFERENCE_MEANS:
        result[name].update(percentile=percentile, reference_index=.5 + percentile / 100)
    assert [card["level"] for card in metric_summaries(result)[1:]] == [level] * 3


def test_reference_means_match_independent_historical_pandas_ranks():
    path = Path.home() / "outputs/message_delivery_meeting_memo_20260825/analysis/combo_input.csv"
    if not path.exists():
        pytest.skip("Historical reference is private and local-only")
    data = pd.read_csv(path)
    for name, column in [("aipm1", "AIPM_v1(OPM)"), ("aipm2", "AIPM_v2(adrecall)"),
                         ("message_delivery", "md_score_eval")]:
        indices = .5 + data[column].rank(method="average") / 49
        assert REFERENCE_MEANS[name] == pytest.approx(indices.mean(), abs=1e-14)
        assert (100 * (indices / REFERENCE_MEANS[name] - 1)).mean() == pytest.approx(0, abs=1e-12)


def test_result_page_has_overall_card_above_three_components():
    root = Path(__file__).resolve().parents[1]
    source = (root / "app_pages/video_pretest.py").read_text().split('\nst.title(')[0]
    app = AppTest.from_string(source + '\nshow_result(st.session_state["result"])\n')
    result = sample_result()
    before = deepcopy(result)
    app.session_state["result"] = result
    app.run(timeout=30)
    assert not app.exception
    text = " ".join(item.value for item in app.markdown)
    for value in ["−46%", "+5%", "−11%", "−43%", "Ниже нормы", "В норме"]:
        assert value in text
    assert text.count("к среднему") == 4
    assert "Основная идея тестового ролика" in text
    assert len(app.get("plotly_chart")) == 11
    assert text.index('data-metric="overall"') < text.index('data-metric="component"')
    assert text.count('data-metric="component"') == 3
    assert "font-size:60px" in text
    assert len(app.get("download_button")) == 1
    assert app.session_state["result"] == before


def test_reference_mean_is_displayed_as_zero_without_negative_zero():
    result = sample_result()
    result["aipm3"] = {"index": 1, "level": 1}
    for name, mean in REFERENCE_MEANS.items():
        result[name].update(reference_index=mean, percentile=100 * (mean - .5))
    assert all(card["delta"] == 0 for card in metric_summaries(result))
    app = AppTest.from_string('''
import streamlit as st
from aipm3.summary_ui import show_metric_summary
show_metric_summary(st.session_state["result"])
''')
    app.session_state["result"] = result
    app.run()
    assert not app.exception
    text = " ".join(item.value for item in app.markdown)
    assert text.count(">0%</div>") == 4
