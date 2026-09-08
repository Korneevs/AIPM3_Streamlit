"""Compact Plotly profile; no generated narrative or implied causal weights."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .feature_profile import GROUPS, build_profile


def group_figure(rows, maximum):
    figure = go.Figure()
    labels = [r['label'] for r in rows]
    figure.add_trace(go.Bar(x=[maximum] * len(rows), y=labels, orientation='h',
                            marker_color='#EEF1F5', hoverinfo='skip'))
    figure.add_trace(go.Bar(
        x=[r['value'] for r in rows], y=labels, orientation='h',
        marker_color='#5B7FE8',
        customdata=[[r['display'], r['low'], r['high'], r['calls']] for r in rows],
        hovertemplate='%{y}<br>%{customdata[0]}<br>Диапазон между запросами: '
                      '%{customdata[1]:.2f}–%{customdata[2]:.2f}<br>Запросов: %{customdata[3]}<extra></extra>',
        error_x=dict(type='data', symmetric=False,
                     array=[r['high'] - r['value'] for r in rows],
                     arrayminus=[r['value'] - r['low'] for r in rows],
                     color='#273A69', thickness=1.5, width=5),
    ))
    for row in rows:
        figure.add_annotation(x=maximum * 1.02, y=row['label'], text=row['display'],
                              showarrow=False, xanchor='left', font=dict(size=14, color='#353B4A'))
    figure.update_layout(
        height=max(150, len(rows) * 60 + 35), barmode='overlay', bargap=.58,
        margin=dict(l=0, r=0, t=4, b=24), showlegend=False,
        xaxis=dict(range=[0, maximum * 1.43], tickvals=[0, maximum],
                   ticktext=['0', f'{maximum}%' if maximum == 100 else str(maximum)],
                   fixedrange=True, showgrid=False, zeroline=False),
        yaxis=dict(autorange='reversed', fixedrange=True, title='', tickfont=dict(size=14)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return figure


def show_feature_profile(result):
    profile = build_profile(result)
    st.subheader('Профиль ролика по группам признаков')
    st.caption('Измеренные свойства, а не дополнительные баллы качества. Больше — не всегда лучше.')
    for pair in [('story', 'brand'), ('structure', 'channels')]:
        for col, key in zip(st.columns(2), pair):
            with col, st.container(border=True):
                title, description = GROUPS[key]
                st.markdown('**' + title + '**')
                st.caption(description)
                rows = [r for r in profile['measurements'] if r['group'] == key]
                withheld = [r for r in profile['withheld'] if r['group'] == key]
                if key == 'story':
                    for row in rows:
                        left, right = st.columns([3, 1])
                        left.write(row['label'])
                        right.markdown('**' + row['display'] + '**')
                elif rows:
                    st.plotly_chart(group_figure(rows, rows[0]['maximum']), use_container_width=True,
                                    config={'displayModeBar': False}, key='profile_' + key)
                for row in withheld:
                    st.caption(row['label'] + ': ' + row['reason'].lower() + '.')
    st.caption('Проверено в пилоте: 3 ролика, по 2 новых прогона. Штрихи на шкалах — разброс между запросами текущего запуска.')
    with st.expander('Что измеряют шкалы и как проверена устойчивость'):
        st.write('Логотип и пэкшот — среднее число секунд из двух разметок, делённое на длительность ролика. '
                 'Эти интервалы могут пересекаться, поэтому мы их не складываем. Сюжетные приёмы показываются '
                 'только при совпадении всех трёх разметок AIPM 1.0.')
        st.write('Для структуры и каналов сначала усредняем три оценки внутри запроса, затем десять запросов. '
                 'Покрытие посыла — в скольких из четырёх равных частей он выражен. Условия — среднее число '
                 'ограничений предложения на исходной шкале 0–4. Полнота по звуку и без звука — оценка 0–3; '
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
