from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from aipm3 import vertical_uvp as uvp
from aipm3.result_export import export_result


def sample():
    return {'source_sha': 'video-one', 'aipm3': {'index': .5}, 'blind_answers': [
        {'respondent_id': str(i), 'answer': 'На сервисе большой выбор нужных вещей.'} for i in range(30)]}


def payload(result, status='matched'):
    return {'answers': [{'respondent_id': row['respondent_id'], 'status': status,
                         'quote': '' if status == 'absent' else 'большой выбор нужных вещей'}
                        for row in result['blind_answers']]}


def test_all_verticals_and_goods_have_exact_requested_uvps():
    expected = {'Авто · C2C транзакция': 'Trust & Safety', 'Путешествия': 'Value',
                'Недвижимость': 'Trust & Safety', 'Услуги': 'Trust', 'Работа': 'Trust', 'Horizont': 'Assortment'}
    for vertical, label in expected.items():
        assert uvp.make_target(vertical)['label'] == label
    assert uvp.make_target('Авто · C2C транзакция')['period'] == 'H2 2027'
    assert uvp.make_target('Товары', 'Ресейл')['label'] == 'Assortment'
    assert uvp.make_target('Товары', 'Распродажа')['label'] == 'Value'
    with pytest.raises(KeyError):
        uvp.make_target('Товары')


@pytest.mark.parametrize('matched,partial,total,status', [(16,0,30,'matched'), (15,0,30,'partial'),
    (10,5,30,'partial'), (0,15,30,'partial'), (0,14,30,'absent'), (29,0,29,'insufficient')])
def test_majority_rule_is_explicit_and_incomplete_panels_not_certified(matched, partial, total, status):
    rows = ([{'status': 'matched'}]*matched + [{'status': 'partial'}]*partial
            + [{'status': 'absent'}]*(total-matched-partial))
    assert uvp.summarize_alignment(rows)['status'] == status


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown', 'invented_quote', 'empty_quote', 'bad_status'])
def test_every_claim_requires_a_real_quote_and_exact_answer_coverage(mutation):
    r = sample(); p = payload(r)
    if mutation == 'missing': p['answers'].pop()
    if mutation == 'duplicate': p['answers'][-1] = p['answers'][0]
    if mutation == 'unknown': p['answers'][0]['respondent_id'] = 'invented'
    if mutation == 'invented_quote': p['answers'][0]['quote'] = 'безопасная сделка'
    if mutation == 'empty_quote': p['answers'][0]['quote'] = ''
    if mutation == 'bad_status': p['answers'][0]['status'] = 'perfect'
    with pytest.raises(ValueError): uvp.validate_alignment(p, r['blind_answers'])


def test_text_only_postcheck_preserves_result_and_caches_by_source_answers_and_target(tmp_path, monkeypatch):
    r = sample(); original = deepcopy(r); calls = []
    def ask(prompt, schema, api_key, video_base64=None):
        calls.append(prompt)
        assert video_base64 is None
        assert schema == uvp.RESPONSE_SCHEMA
        assert 'aipm3' not in prompt and 'index' not in prompt
        return payload(r)
    monkeypatch.setattr(uvp, 'ask_json', ask)
    target = uvp.make_target('Товары', 'Ресейл')
    one = uvp.evaluate_uvp(r, target, 'test', tmp_path)
    two = uvp.evaluate_uvp(r, target, 'test', tmp_path)
    assert one == two and len(calls) == 1
    assert r == original
    uvp.evaluate_uvp(r, uvp.make_target('Товары', 'Распродажа'), 'test', tmp_path)
    r['source_sha'] = 'other-video'
    uvp.evaluate_uvp(r, target, 'test', tmp_path)
    r['blind_answers'][0]['answer'] += ' Новые детали.'
    uvp.evaluate_uvp(r, target, 'test', tmp_path)
    assert len(calls) == 4
    exported = export_result({**original, 'vertical_uvp': one})
    assert exported['vertical_uvp'] == one and exported['aipm3'] == original['aipm3']


def test_failed_or_incomplete_diagnostics_are_not_cached_as_negative(tmp_path, monkeypatch):
    r = sample()
    monkeypatch.setattr(uvp, 'ask_json', lambda *a: {'answers': []})
    with pytest.raises(ValueError): uvp.evaluate_uvp(r, uvp.make_target('Работа'), 'test', tmp_path)
    assert not list(tmp_path.glob('*.json'))
    r['blind_answers'].pop()
    check = uvp.evaluate_uvp(r, uvp.make_target('Работа'), 'test', tmp_path)
    assert check['status'] == 'insufficient' and check['available_answers'] == 29


def test_ui_shows_chosen_target_quotes_and_synthetic_basis():
    r = sample()
    assessment = {'target': uvp.make_target('Товары', 'Ресейл'),
                  **uvp.summarize_alignment(uvp.validate_alignment(payload(r), r['blind_answers']))}
    app = AppTest.from_string('''
import streamlit as st
from aipm3.uvp_ui import show_uvp
show_uvp(st.session_state['result'])
''')
    app.session_state['result'] = {**r, 'vertical_uvp': assessment}
    app.run()
    assert not app.exception
    assert any('Попали' in item.value for item in app.success)
    text = ' '.join(item.value for item in app.markdown)
    assert 'Assortment' in text and 'Ресейл' in text and 'большой выбор нужных вещей' in text
    assert '30 из 30' in text
    assert any('не опрос живых' in c.value for c in app.caption)


def test_ui_wrapper_keeps_scores_after_failed_uvp_call(monkeypatch):
    import types
    root = Path(__file__).resolve().parents[1]
    source = (root / 'app_pages/video_pretest.py').read_text().split('\nst.title(')[0]
    page = types.ModuleType('page_test')
    exec(compile(source, 'page_test', 'exec'), page.__dict__)
    def fail(*a, **kw): raise RuntimeError('network failed')
    monkeypatch.setattr(page, 'evaluate_uvp', fail)
    r = sample(); before = deepcopy(r)
    result = page.with_uvp(r, uvp.make_target('Работа'), 'test')
    assert result['vertical_uvp']['status'] == 'error'
    assert r == before and result['aipm3'] == r['aipm3']


def test_page_selection_never_reruns_models_and_uvp_failure_keeps_previous_scores(monkeypatch):
    import artifacts
    from aipm3 import pipeline
    from test_summary_ui import sample_result
    root = Path(__file__).resolve().parents[1]
    weights = root / 'deployment_artifacts'
    if not weights.exists(): pytest.skip('Private local weights')
    monkeypatch.setattr(artifacts, 'artifact_path', lambda name: str(weights / name))
    def forbidden(*args, **kwargs): pytest.fail('Selector changes must never rescore a video')
    monkeypatch.setattr(pipeline, 'run_analysis', forbidden)
    result = sample_result()
    result['source_sha'] = 'saved-video'
    result['blind_answers'] = sample()['blind_answers']
    result['vertical_uvp'] = {'target': uvp.make_target('Работа'), 'status': 'error'}
    original = deepcopy(result)
    app = AppTest.from_file(str(root / 'app_pages/video_pretest.py'), default_timeout=30)
    app.secrets['VSELLM_API_KEY'] = 'test'
    app.session_state['aipm3_result'] = result
    app.run()
    assert not app.exception
    app.selectbox[0].select('Товары').run()
    assert len(app.selectbox) == 3  # vertical, goods, manual celebrity
    app.selectbox[1].select('Ресейл').run()
    assert not app.exception
    assert app.session_state['aipm3_result'] == original
    assert any('другая вертикаль' in item.value for item in app.info)
    def fail(*args, **kwargs): raise RuntimeError('network failed')
    monkeypatch.setattr(uvp, 'ask_json', fail)
    next(b for b in app.button if b.label == 'Проверить UVP для выбранной вертикали').click().run()
    saved = app.session_state['aipm3_result']
    assert saved['aipm3'] == original['aipm3']
    assert saved['vertical_uvp']['target']['goods'] == 'Ресейл'
    assert saved['vertical_uvp']['status'] == 'error'
    app.selectbox[0].select('Работа').run()
    assert len(app.selectbox) == 2  # vertical and manual celebrity
    assert app.session_state['aipm3_result']['vertical_uvp']['target']['goods'] == 'Ресейл'
