from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import latest_runtime as runtime
from aipm3.latest_interpretation import build_latest_interpretation
from aipm3.latest_profile_ui import profile_rows
from aipm3.manager_report import HIDDEN_MANAGER_FEATURES


@pytest.mark.parametrize("kind", ["finished", "neuromatics"])
def test_profile_uses_current_heads_without_losing_features_or_changing_scores(kind):
    frames = {}
    for task in 'nmr':
        train = pd.read_csv(runtime.BUNDLE_DIR / 'data' / f'fit_{task}.csv')
        frame = pd.DataFrame([train.iloc[0].to_dict() for _ in range(3)])
        frame['record'] = frame['family'] = 'interface-fixture'
        frame['sha'] = 'a' * 64
        frame['repeat'] = [1, 2, 3]
        frame['brand'] = 'Avito'
        frame['vertical'] = 'Goods'
        frame['phys__duration'] = 30.
        frames[task] = frame
    result = runtime.score_feature_rows(frames, metadata={'source_sha': 'a' * 64},
                                      material_kind=kind, repeat_count=3)
    before = deepcopy(result)
    explanation = build_latest_interpretation(result)
    profiles = profile_rows(explanation)
    for task, score_name in runtime.SCORE_NAMES.items():
        assert explanation['details'][task]['actual'] == pytest.approx(result['scores'][score_name])
        expected = set(result['model_inputs'][task]) - HIDDEN_MANAGER_FEATURES
        assert {r['feature'] for r in profiles[task]} == expected
        assert len(profiles[task]) == len(expected)
        assert sum(r['importance'] for r in profiles[task]) == pytest.approx(100)
    assert result == before


def test_audio_limited_cards_do_not_show_percentages_or_norms():
    interpretation = {'material_kind': 'neuromatics', 'cards': [
        {'task': task, 'index': 120, 'level': 'Выше типичного уровня', 'assessment_available': False}
        for task in 'nmr'], 'overall': {'index': 120, 'level': 'Выше типичного уровня', 'assessment_available': False}}
    app = AppTest.from_string('from aipm3.latest_profile_ui import show_summary\n'
                             'import streamlit as st\nshow_summary(st.session_state["explanation"])')
    app.session_state['explanation'] = interpretation
    app.run()
    assert not app.exception
    text = ' '.join(x.value for x in app.markdown)
    assert '+20%' not in text and 'Выше нормы' not in text
    assert text.count('Нужна полная озвучка') == 4
