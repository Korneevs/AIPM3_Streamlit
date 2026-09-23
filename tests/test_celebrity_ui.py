from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aipm3.manual_celebrity import apply_celebrity
from aipm3.models import aipm3_score
from aipm3.vertical_uvp import make_target
from test_summary_ui import sample_result


def source():
    result = sample_result()
    result['aipm3'] = aipm3_score(result['aipm1'], result['aipm2'], result['message_delivery'])
    result['source_sha'] = 'same-video'
    result['vertical_uvp'] = {'status': 'error', 'target': make_target('Работа')}
    return result


@pytest.mark.parametrize('choice,name,status,share', [
    ('none', 'Без селебрити из списка.', 'Не влияет', '0%'),
    ('fomenko', 'Николай Фоменко', 'Повышает', '23%'),
    ('zhuravlyov', 'Дмитрий Журавлёв', 'Повышает', '23%'),
    ('kurkova', 'Равшана Куркова', 'Повышает', '23%'),
])
def test_profile_shows_manual_participation_with_separate_weight_basis(choice, name, status, share):
    app = AppTest.from_string('''
import streamlit as st
from aipm3.profile_ui import show_feature_profile
show_feature_profile(st.session_state['result'])
''')
    result = apply_celebrity(source(), choice)
    before = deepcopy(result)
    app.session_state['result'] = result
    app.run(timeout=30)
    assert not app.exception
    text = ' '.join(item.value for item in app.markdown)
    assert name in text and status + ' · запоминаемость' in text
    assert 'Вес в итоговой запоминаемости: <strong>' + share in text
    assert 'Вес среди свойств ролика' in text
    assert 'считаются отдельно' in text
    assert not any(word in text for word in ['1,3', '1.3', 'множител', 'умнож'])
    assert any('Участие указано вручную' == c.value for c in app.caption)
    assert len(app.get('plotly_chart')) == 11
    assert app.session_state['result'] == before


def test_choice_only_changes_saved_scores_on_apply_without_inference_or_uvp(monkeypatch):
    import artifacts
    from aipm3 import models, pipeline, vertical_uvp
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(artifacts, 'artifact_path', lambda name: name)
    monkeypatch.setattr(models, 'load_frozen_models', lambda *args: object())
    def forbidden(*args, **kwargs):
        pytest.fail('Manual choice must not call inference or UVP')
    monkeypatch.setattr(pipeline, 'run_analysis', forbidden)
    monkeypatch.setattr(vertical_uvp, 'evaluate_uvp', forbidden)
    app = AppTest.from_file(str(root / 'app_pages/video_pretest.py'), default_timeout=30)
    app.secrets['VSELLM_API_KEY'] = 'test'
    original = source()
    app.session_state['aipm3_result'] = deepcopy(original)
    app.run()
    assert not app.exception
    app.selectbox(key='selected_celebrity').select('fomenko').run()
    assert app.session_state['aipm3_result'] == original
    assert any('Выбор селебрити изменён' in info.value for info in app.info)
    next(b for b in app.button if b.label == 'Применить выбор селебрити').click().run()
    adjusted = deepcopy(app.session_state['aipm3_result'])
    assert adjusted == apply_celebrity(original, 'fomenko')
    assert adjusted['vertical_uvp'] == original['vertical_uvp']
    app.run()
    assert app.session_state['aipm3_result'] == adjusted
    app.selectbox(key='selected_celebrity').select('none').run()
    assert app.session_state['aipm3_result'] == adjusted
    next(b for b in app.button if b.label == 'Применить выбор селебрити').click().run()
    restored = app.session_state['aipm3_result']
    for key in ('aipm1', 'aipm2', 'aipm3', 'message_delivery', 'vertical_uvp'):
        assert restored[key] == original[key]
    assert not app.exception
