"""Compact Plotly profile; no generated narrative or implied causal weights."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .feature_profile import build_profile


_GROUPS = {
    'story': ('Сюжетные приёмы', 'Какие приёмы модель обнаружила в ролике.'),
    'brand': ('Бренд в кадре', 'Длительность показа и её доля в хронометраже. '
              'Логотип и финальный кадр могут пересекаться по времени.'),
    'structure': ('Структура сообщения', 'В скольких частях ролика выражен посыл '
                  'и сколько условий предложения нужно учесть.'),
    'channels': ('Звук и изображение по отдельности', 'Шкала полноты сообщения: '
                 '0 — посыл не определить, 3 — возможен полный пересказ. '
                 'Это оценка модели, а не доля понявших людей.'),
}

_LABELS = {
    'pack_shot_duration_seconds': 'Финальный брендовый кадр',
    'core_claim_quartile_coverage': 'Части ролика с основным посылом',
    'offer_condition_count': 'Число условий предложения',
    'audio_only_message_completeness': 'Только по звуку',
    'visual_only_message_completeness': 'Только по изображению, без звука',
}


def _label(row):
    return _LABELS.get(row['feature'], row['label'])


def _display(row):
    """Present existing measurements in their own units without changing values."""
    value = f"{row['value']:.1f}".replace('.', ',')
    if row['feature'] == 'offer_condition_count':
        return f'{value} в среднем · число условий на шкале 0–4'
    if row['feature'] == 'core_claim_quartile_coverage':
        return f'{value} из 4 частей в среднем'
    if row['group'] == 'channels':
        return f'{value} из 3 по шкале полноты'
    return row['display']


def _chart_label(row):
    # Values are rendered as regular text outside Plotly. Short, wrapped axis
    # labels leave room for bars even in a narrow mobile viewport.
    return {
        'pack_shot_duration_seconds': 'Финальный кадр<br>с брендом',
        'core_claim_quartile_coverage': 'Части ролика<br>с посылом',
        'offer_condition_count': 'Число условий',
        'audio_only_message_completeness': 'Только звук',
        'visual_only_message_completeness': 'Без звука',
    }.get(row['feature'], row['label'])


def group_figure(rows, maximum):
    figure = go.Figure()
    labels = [_chart_label(r) for r in rows]
    figure.add_trace(go.Bar(x=[maximum] * len(rows), y=labels, orientation='h',
                            marker_color='#EEF1F5', hoverinfo='skip'))
    figure.add_trace(go.Bar(
        x=[r['value'] for r in rows], y=labels, orientation='h',
        marker_color='#5B7FE8',
        customdata=[[_display(r), r['low'], r['high'], r['calls']] for r in rows],
        hovertemplate='%{y}<br>%{customdata[0]}<br>Диапазон между запросами: '
                      '%{customdata[1]:.2f}–%{customdata[2]:.2f}<br>Запросов: %{customdata[3]}<extra></extra>',
        error_x=dict(type='data', symmetric=False,
                     array=[r['high'] - r['value'] for r in rows],
                     arrayminus=[r['value'] - r['low'] for r in rows],
                     color='#273A69', thickness=1.5, width=5),
    ))
    figure.update_layout(
        height=max(150, len(rows) * 60 + 35), barmode='overlay', bargap=.58,
        margin=dict(l=0, r=12, t=4, b=24), showlegend=False,
        xaxis=dict(range=[0, maximum * 1.04], tickvals=[0, maximum],
                   ticktext=['0', f'{maximum}%' if maximum == 100 else str(maximum)],
                   fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(autorange='reversed', fixedrange=True, title='',
                   automargin=True, tickfont=dict(size=12)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return figure


def show_feature_profile(result, profile=None):
    """Render a live profile or the exact precomputed profile from an export."""
    if profile is None:
        profile = build_profile(result)
    st.subheader('Что обнаружено в ролике')
    st.caption('Эти наблюдения описывают ролик, но не объясняют вклад каждого признака '
               'в итоговый балл. Больше — не всегда лучше.')
    for key, (title, description) in _GROUPS.items():
        with st.container(border=True):
            st.markdown('**' + title + '**')
            st.caption(description)
            rows = [r for r in profile['measurements'] if r['group'] == key]
            withheld = [r for r in profile['withheld'] if r['group'] == key]
            for row in rows:
                st.markdown('**' + _label(row) + ':** ' + _display(row))
            if rows and key != 'story':
                st.plotly_chart(group_figure(rows, rows[0]['maximum']), use_container_width=True,
                                config={'displayModeBar': False}, key='profile_' + key)
            if key == 'structure' and rows:
                st.caption('Ролик делится на четыре равные части. Дробные значения получаются '
                           'при усреднении разметок. Число условий — количество, не балл качества.')
            for row in withheld:
                st.caption(_label(row) + ': ' + row['reason'].lower() + '.')
    st.caption('Штрихи на графиках — диапазон оценок между запросами текущего запуска.')
    with st.expander('Как получены наблюдения и что проверялось'):
        st.write('Проверка на трёх роликах касалась повторяемости этих отдельных признаков. '
                 'Она не подтверждает точность итогового балла, его категории или реакции аудитории.')
        st.write('Логотип и пэкшот — среднее число секунд из двух разметок, делённое на длительность ролика. '
                 'Эти интервалы могут пересекаться, поэтому мы их не складываем. Сюжетные приёмы показываются '
                 'только при совпадении всех трёх разметок AIPM 1.0.')
        st.write('Для структуры и каналов сначала усредняем три оценки внутри запроса, затем десять запросов. '
                 'Покрытие посыла — в скольких из четырёх равных частей он выражен. Условия — среднее число '
                 'ограничений предложения на исходной шкале 0–4. Например, 0,8 означает среднее число условий, '
                 'а не 0,8 балла качества из 4. Полнота по звуку и без звука — оценка 0–3; '
                 'среднее не является процентом понявших людей.')
        st.write('Каждая полоса — отдельный признак. Общего балла группы, ручных весов и причинных вкладов здесь нет. '
                 'Используются исходные признаки, включая диагностические поля вне финальных регрессоров. '
                 'Низкая полоса означает меньше соответствующего свойства, а не автоматически плохой ролик.')
        st.write('Пилот: сохранённый исходный запуск и два новых извлечения на каждом из трёх роликов, с неизменными '
                 'промптами и моделью. Для числовых признаков допуск — 10% длины шкалы, для сюжетных — полное совпадение. '
                 'Это инженерный допуск, не статистическое доказательство эквивалентности. '
                 'Максимальный сдвиг: показ бренда и сюжетные приёмы — 0; покрытие посыла — 0,20 из 4; '
                 'число условий — 0,13 из 4; полнота звука — 0,20 из 3; изображения — 0,03 из 3.')
        st.write('Ясность CTA не включена: различие достигло 20 п.п. Recovery и новые свободные описания не добавлены '
                 'в профиль без отдельной межзапусковой проверки. Для числовых признаков показан фактический разброс '
                 'между запросами. В запросах разные персоны: их разногласия нельзя считать повторными запусками '
                 'или превращать в доказательство устойчивости. Неполные данные не оцениваются. '
                 'Такая проверка не гарантирует устойчивость на любом будущем ролике. Исходные баллы моделей не меняются.')
