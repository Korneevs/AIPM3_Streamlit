"""Manager-facing comparison of intended promise and recovered idea."""
from __future__ import annotations

import streamlit as st

from . import brief_details, creative_review as review, message_alignment


def show_message_comparison(comparison: dict, key: str) -> None:
    relation, gap = comparison['relation']['status'], comparison['gap']
    titles = {
        'equivalent': 'Основная идея совпадает с тем, что хотели донести',
        'partial': 'Основная идея и задуманный посыл совпадают не полностью',
        'different': 'Ролик доносит другую идею',
        'uncertain': 'Совпадение идей пока нельзя определить уверенно',
    }
    if set(comparison['relation']['coder_statuses']) == {'equivalent', 'partial'}:
        titles['uncertain'] = 'Идеи близки; спорна только полнота совпадения'
    if comparison.get('diagnostic_conflicts'):
        st.warning('Смысловое сравнение и разметка ответов расходятся. Уверенно оценить потерю посыла пока нельзя.')
    elif relation == 'equivalent':
        st.success(titles[relation])
    else:
        st.warning(titles[relation])
    for column, target, label in zip(st.columns(2), ['uvp', 'main_idea'],
                                     ['Хотели донести', 'Основная идея по модели']):
        with column, st.container(border=True):
            st.caption(label.upper())
            st.text(comparison[target])
            item = comparison['summary'][target]
            value = f"{item['matched_pct']:.0f}%"
            if item['lower_pct'] != item['upper_pct']:
                value = f"{item['lower_pct']:.0f}–{item['upper_pct']:.0f}%"
            st.metric('Смысл сохранился', '—' if comparison.get('diagnostic_conflicts') else value)
            caption = ('Есть конфликт разметки — частота пока не подтверждена.'
                       if comparison.get('diagnostic_conflicts') else
                       f"Уверенно — в {item['matched_count']} из {item['n']} ответов.")
            if not comparison.get('diagnostic_conflicts') and item['lower_matched_count'] != item['upper_matched_count']:
                caption += f" Ещё {item['upper_matched_count'] - item['lower_matched_count']} — спорные."
            st.caption(caption)
    direction = gap['direction']
    if direction == 'main_higher':
        st.markdown(f"**Задуманный посыл встречается на {gap['pp']:.1f} п.п. реже основной идеи.**")
    elif direction == 'uvp_higher':
        st.markdown(f"**Задуманный посыл встречается на {abs(gap['pp']):.1f} п.п. чаще основной идеи модели.**")
        st.write('В этой панели ответов UVP встречается чаще. Основную идею модель выбрала на этапе исходного анализа ролика.')
    elif direction == 'equal':
        if all(comparison['summary'][d]['matched_count'] == 0 for d in ['uvp', 'main_idea']):
            st.write('Ни один из смыслов не получил подтверждённых полных совпадений. Нулевая разница здесь не означает, что посыл считывается.')
        elif relation == 'equivalent':
            st.markdown('**Потери относительно основной идеи нет: разница 0 п.п.**')
        else:
            st.markdown('**Частота одинаковая, но это не означает совпадения смыслов. Разница 0 п.п.**')
    elif direction == 'uncertain' and gap['upper_pp'] == 0 and gap['lower_pp'] < 0:
        st.markdown('**Потери UVP относительно основной идеи не видно.**')
        st.write(f"Задуманный посыл встречается не реже основной идеи. В зависимости от разметки — "
                 f"с той же частотой или на {abs(gap['lower_pp']):.1f} п.п. чаще.")
    elif direction == 'uncertain' and gap['lower_pp'] == 0 and gap['upper_pp'] > 0:
        st.markdown(f"**Возможная потеря UVP — от 0 до {gap['upper_pp']:.1f} п.п.**")
        st.write('Разметки дают либо одинаковую частоту двух смыслов, либо более редкое упоминание задуманного посыла.')
    else:
        st.write('Надёжно определить, какой смысл считывается чаще, пока нельзя.')
        if not comparison.get('diagnostic_conflicts'):
            st.caption(f"При спорной разметке разница может составлять от {gap['lower_pp']:.1f} до {gap['upper_pp']:.1f} п.п.")
    if direction in {'main_higher', 'uvp_higher'} and gap['lower_pp'] != gap['upper_pp']:
        st.caption(f"С учётом расхождений разметки разница: {gap['lower_pp']:.1f}–{gap['upper_pp']:.1f} п.п. "
                   "Положительное значение означает, что UVP встречается реже.")
    st.caption('Сравнение на одних и тех же ответах синтетиков. Это не прогноз процента людей и не балл Message Delivery.')
    with st.expander('Почему такой вывод'):
        for reason in dict.fromkeys(comparison['relation'].get('reasons', [])):
            if reason:
                st.text(reason)
        st.write('Оба смысла проверены по одной инструкции на одной панели. Считается полная передача смысла; '
                 'частичное упоминание не приравнивается к полному. Расхождения двух разметок не скрываются.')
        rows = comparison['answers']
        selected = st.selectbox('Исходный ответ', range(len(rows)), key=key + '_original',
                                format_func=lambda i: rows[i]['respondent_id'])
        row = rows[selected]
        st.text('«' + row['answer'] + '»')
        for target, label in [('uvp', 'Задуманный посыл'), ('main_idea', 'Основная идея модели')]:
            st.caption(label + ': ' + review.STATUS_LABELS[row[target]['status']])
            item = comparison['summary'][target]
            if item['lower_matched_count'] != item['upper_matched_count']:
                st.caption(f"С учётом спорных ответов: {item['lower_matched_count']}–{item['upper_matched_count']} из {item['n']}.")


def show_promise_support(brief: dict, details: dict | None) -> None:
    if not brief.get('rtb'):
        st.caption('Подтверждение обещания не задано — эту часть не проверяли.')
        return
    st.markdown('#### Что осталось от подтверждения обещания')
    st.write('По брифу обещание должно подтверждаться так:')
    st.text('«' + brief['rtb'] + '»')
    if not details:
        st.caption('Проверка этой части ещё не готова.')
        return
    parts = [p for p in details['summary'] if p['dimension'] == 'rtb']
    complete_ids = set.intersection(*[{r['respondent_id'] for r in p['answers'] if r['status'] == 'matched'} for p in parts]) if parts else set()
    losses = []
    for part in parts:
        phrases = sorted({r.get('missing_brief_quote') for r in part['answers'] if r.get('missing_brief_quote')})
        for phrase in phrases:
            cases = [r for r in part['answers'] if r.get('missing_brief_quote') == phrase]
            if cases:
                losses.append((len(cases), part['n'], phrase, cases[0]['answer']))
    if any(p['counts']['contradicted'] > p['n'] / 2 for p in parts):
        st.write('В большинстве ответов появляется смысл, противоположный подтверждению обещания.')
    elif parts and len(complete_ids) >= .8 * parts[0]['n']:
        st.write('Подтверждение обещания сохраняется в большинстве ответов — вместе с важными деталями.')
        if losses:
            count, n, phrase, _ = sorted(losses, key=lambda x: (-x[0], x[2]))[0]
            st.caption(f'При этом в {count} из {n} ответов пропущена деталь «{phrase}», хотя остальная часть передана.')
    elif losses:
        count, n, phrase, answer = sorted(losses, key=lambda x: (-x[0], x[2]))[0]
        st.markdown(f'**Теряется важная деталь: «{phrase}».**')
        st.write('Встречается пересказ, в котором часть предложения есть, а эта деталь пропущена:')
        st.text('«' + answer + '»')
        st.caption(f'Такая потеря есть в {count} из {n} ответов. Оба разметчика её подтвердили.')
    elif any(p['counts']['absent'] > p['n'] / 2 for p in parts):
        st.write('В большинстве ответов одна из важных частей подтверждения не упоминается.')
    else:
        st.write('Подтверждение обещания сохраняется не во всех ответах. Однозначную потерю отдельной детали пока выделить нельзя.')
    if parts:
        st.caption(f"Подтверждение целиком, со всеми деталями: {len(complete_ids)} из {parts[0]['n']} ответов.")
    with st.expander('Проверить детали подтверждения'):
        for part in parts:
            c, n = part['counts'], part['n']
            st.write(f"«{part['brief_quote']}»: полностью — {c['matched']}, частично — {c['partial']}, "
                     f"не выражено — {c['absent']}, противоположный смысл — {c['contradicted']}, "
                     f"спорная разметка — {c['uncertain']}; всего {n} ответов.")
        st.caption('Неупоминание в коротком ответе не доказывает непонимание. '
                   'Здесь не проверяются истинность обещания и доверие к нему.')


def show_brief_review(result: dict, api_key: str) -> None:
    st.subheader('Считали ли то, что мы хотели сказать?')
    st.write('Сравним задуманный посыл с основной идеей ролика: совпал ли смысл и как часто каждый из них остаётся в ответах.')
    try:
        answers = message_alignment.source_answers(result)
    except ValueError:
        st.info('В этом результате нет полной панели свободных ответов после просмотра. '
                'Для сравнения нужен новый анализ ролика: оценочные описания признаков вместо неё не используются.')
        return
    idea = str(result.get('main_idea') or '').strip()
    diagnostics = result.setdefault('diagnostics', {})
    key = 'brief_' + result.get('source_sha', result.get('video_sha', 'unknown'))
    saved = diagnostics.get('current_brief', {})
    existing = diagnostics.get('message_alignment')
    with st.expander('Что хотели донести в этой кампании', expanded=not bool(existing)):
        brief = {
            'uvp': st.text_area('Главный посыл кампании (UVP)', value=saved.get('uvp', ''),
                                max_chars=2000, key=key + '_uvp',
                                help='Мысль о ценности предложения, которую зритель должен вынести из ролика.'),
            'rtb': st.text_area('Чем подтверждаем это обещание (RTB, необязательно)', value=saved.get('rtb', ''),
                                max_chars=2000, key=key + '_rtb',
                                help='Что в ролике даёт основание обещать эту выгоду: факт, свойство товара или демонстрация.'),
        }
    brief = review.normalize_brief(brief)
    diagnostics['current_brief'] = brief
    current = message_alignment.is_current(existing, idea, brief['uvp'], answers)
    details = diagnostics.get('brief_details')
    details_current = brief_details.is_current(details, brief, answers)
    needs_details = bool(brief['rtb']) and not details_current
    if not idea:
        st.info('Основная идея модели не определена. Сравнение с ней пока невозможно.')
    if st.button('Проверить задуманный посыл', disabled=not idea or not brief['uvp'] or (current and not needs_details),
                 key=key + '_run'):
        try:
            if not current:
                with st.spinner('Сравниваем два смысла по сохранённым ответам. Баллы ролика не пересчитываются…'):
                    diagnostics['message_alignment'] = message_alignment.compare(idea, brief['uvp'], answers, api_key)
            if needs_details:
                try:
                    with st.spinner('Проверяем, сохранилось ли подтверждение обещания…'):
                        diagnostics['brief_details'] = brief_details.compare(brief, answers, api_key)
                    diagnostics.pop('brief_details_error', None)
                except Exception as exc:
                    diagnostics['brief_details_error'] = type(exc).__name__
            st.rerun()
        except Exception as exc:
            st.error('Не удалось завершить сравнение. Баллы ролика сохранены. '
                     f'Тип ошибки: {type(exc).__name__}.')
    if current:
        show_message_comparison(existing, key)
        show_promise_support(brief, details if details_current else None)
        if needs_details and diagnostics.get('brief_details_error'):
            st.warning('Проверка подтверждения не завершилась. Нажмите «Проверить задуманный посыл» ещё раз — '
                       'сравнение идей и баллы ролика повторно рассчитываться не будут.')
    elif existing:
        st.info('Бриф изменён или обновилась основная идея. Предыдущее сравнение к ним не относится — запустите проверку.')
    with st.expander('Как считается считываемость посыла'):
        st.write(f'Оба смысла кодируются в одних и тех же {len(answers)} сохранённых ответах. '
                 'Синтетики не видели бриф. Два отдельных запроса проверяют смысловое совпадение и каждый ответ; '
                 'частичные совпадения не засчитываются как полные. Разница показана в процентных пунктах, '
                 'с диапазоном для спорных ответов. Это диагностическая доля ответов, не откалиброванный процент людей.')
        st.write('Это свободные ответы после полного просмотра из той панели, по которой выбрана основная идея. '
                 'Ответы без идеи и упоминания только бренда остаются в знаменателе. '
                 'Равная частота двух разных идей не означает совпадения замысла. '
                 'Эта проверка не меняет исходные признаки, модели, вероятности, баллы и классы.')
