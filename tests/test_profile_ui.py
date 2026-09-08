"""Rendering boundaries for native measurements and saved export previews."""
import copy

from streamlit.testing.v1 import AppTest

from aipm3 import profile_ui


def saved_profile():
    # Exported profiles can retain exact measurements after raw runs are removed.
    return {
        'measurements': [{
            'feature': 'brand_logo_screen_seconds', 'group': 'brand',
            'label': 'Логотип в кадре', 'value': 75, 'low': 50, 'high': 100,
            'calls': 2, 'maximum': 100, 'display': '≈15 сек. · 75%',
        }],
        'withheld': [],
    }


def test_export_preview_keeps_saved_values_without_rebuilding_or_mutation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('An export preview must not reconstruct removed raw data')

    monkeypatch.setattr(profile_ui, 'build_profile', forbidden)
    saved = saved_profile()
    before = copy.deepcopy(saved)
    app = AppTest.from_string('''
import streamlit as st
from aipm3.profile_ui import show_feature_profile
show_feature_profile({}, profile=st.session_state['saved_profile'])
''')
    app.session_state['saved_profile'] = saved
    app.run()
    assert not app.exception
    assert saved == before
    assert len(app.get('plotly_chart')) == 1
    # The exact duration remains visible as regular text at any viewport width.
    assert any(saved['measurements'][0]['display'] in item.value for item in app.markdown)
    assert not app.get('column')


def test_chart_preserves_observed_range_without_values_outside_plot_area():
    row = saved_profile()['measurements'][0]
    figure = profile_ui.group_figure([row], row['maximum'])
    assert not figure.layout.annotations
    assert list(figure.data[1].x) == [row['value']]
    assert list(figure.data[1].error_x.array) == [row['high'] - row['value']]
    assert list(figure.data[1].error_x.arrayminus) == [row['value'] - row['low']]
    assert figure.layout.xaxis.range[0] <= row['low']
    assert figure.layout.xaxis.range[1] >= row['high']


def test_missing_raw_measurements_are_not_rendered_as_zero_bars():
    app = AppTest.from_string('''
from aipm3.profile_ui import show_feature_profile
show_feature_profile({})
''')
    app.run()
    assert not app.exception
    assert not app.get('plotly_chart')
