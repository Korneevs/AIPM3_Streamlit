"""Manager-facing copy built only from the existing evidence and attributions.

No inference, score changes or invented creative recommendations here.
"""
from __future__ import annotations

import re

from .latest_interpretation import LABELS, observed_label

NEGATIVE_MIN_INDEX_POINTS = .5
NEGATIVE_MIN_STABILITY = 2 / 3


LEVELS = {
    "Ниже типичного уровня": ("Ниже нормы", "#B42318", "#FFF0EE"),
    "Типичный уровень": ("В норме", "#805B00", "#FFF6D6"),
    "Выше типичного уровня": ("Выше нормы", "#137547", "#E9F7EF"),
}
DESCRIPTIONS = {
    "n": "Насколько подача помогает обратить внимание на ролик и его предложение.",
    "m": "Насколько понятно, что предлагает Авито и как это связано с историей.",
    "r": "Насколько история связана с брендом и даёт повод его запомнить.",
}
# These explain the role of an observed feature, without claiming an audience
# response or turning an attribution into a promised effect of an edit.
PURPOSE = {
    "character_close_up_seconds": "Крупный план выделяет героя и его реакцию.",
    "monologue_to_camera": "Обращение в камеру делает реплику адресованной зрителю.",
    "pack_shot_duration_seconds": "В финале важно успеть прочитать название сервиса и основное предложение.",
    "panel__first_core_claim_time_band": "От этого зависит, сколько времени остаётся на объяснение предложения.",
    "product_demo_seconds": "Демонстрация связывает обещание с конкретным действием в сервисе.",
    "state_transformation_present": "Контраст «до и после» показывает, что изменилось для героя.",
    "fresh__audiovisual_claim_alignment": "Здесь смотрим, показано ли то действие или результат, о котором говорит озвучка.",
    "main_character": "Один главный герой задаёт точку, с которой зритель следит за историей.",
    "numeric_offer_on_screen": "Число полезно, когда понятно, к какой выгоде или условию оно относится.",
    "panel__distinct_benefit_count": "При нескольких обещаниях важно сохранить одно главное.",
    "panel__mandatory_inference_chain_length": "Чем больше приходится додумывать, тем важнее явно связать действие и результат.",
    "panel__message_specificity_level": "Конкретные действия и условия объясняют, что именно предлагает сервис.",
    "panel__offer_novelty_explanation_need": "Непривычное предложение может потребовать пояснения прямо в ролике.",
    "panel__product_role_reveal_time_band": "Важно понять роль Авито в истории до того, как ролик закончится.",
    "promo": "Выгоду акции и её условия стоит проверять вместе.",
    "brand_first_mention_seconds": "Первое появление бренда задаёт момент, когда история связывается с Авито.",
    "has_callback_to_opening": "Возвращение к началу связывает завязку и финал в одну историю.",
    "jingle_present": "Фирменная мелодия может служить звуковым напоминанием о бренде.",
    "problem_solution_arc_present": "Связь проблемы и решения показывает, зачем в этой истории нужен сервис.",
    "scene_pace_high": "Темп стоит оценивать вместе с тем, хватает ли времени рассмотреть важные детали.",
}
ABSENCE = {
    "monologue_to_camera": "История может обходиться без прямого обращения к зрителю.",
    "main_character": "Несколько героев допустимы, если понятно, что объединяет их истории.",
    "numeric_offer_on_screen": "Числовое предложение нужно только тогда, когда оно соответствует задаче ролика.",
    "promo": "Отсутствие акции само по себе не делает предложение слабее.",
    "has_callback_to_opening": "Кольцевой сюжет не обязателен: у истории может быть другой способ завершения.",
    "jingle_present": "Добавлять фирменную мелодию только ради оценки не нужно.",
    "problem_solution_arc_present": "Не каждый сюжет должен строиться как отдельная проблема и её решение.",
    "state_transformation_present": "Контраст «до и после» нужен, только если помогает передать основную мысль.",
    "scene_pace_high": "Медленный темп сам по себе не является недостатком.",
    "product_demo_seconds": "Интерфейс или использование сервиса не обязательно показывать в каждом сюжете.",
    "character_close_up_seconds": "Крупные планы не обязательны, если действие понятно и без них.",
}
ALIGNMENT_COPY = {
    0: "Изображение не раскрывает главное предложение или расходится с ним.",
    1: "Кадры связаны с темой предложения. Как именно показанные события раскрывают обещанную выгоду, по ним неочевидно.",
    2: "Показана одна часть обещания - действие или результат. Вся связь между ними на экране не раскрыта.",
    3: "Показаны действие и результат, а озвучка связывает их с основным предложением.",
}


def _alignment_copy(support, value):
    if int(value + .5) == 2 and support.get('shown_action') and not support.get('shown_result'):
        return "Действие из основного предложения показано. Его обещанный результат по этим кадрам подтвердить нельзя."
    if int(value + .5) == 2 and support.get('shown_result') and not support.get('shown_action'):
        return "Обещанный результат показан. Действие, которое к нему приводит, остаётся за кадром."
    return ALIGNMENT_COPY[int(value + .5)]


def _russian_prose(text):
    return len(re.findall(r"[а-яА-ЯёЁ]", text)) > max(8, len(re.findall(r"[a-zA-Z]", text)))


def observation_text(feature, support, value):
    text = plain_text(support["observation"])
    if feature == "panel__mandatory_inference_chain_length" and value < .5:
        return "Связь сюжета с предложением проговорена или показана прямо."
    if feature == "state_transformation_present" and value < .5:
        return "По ролику не удалось подтвердить, что ситуация для героя меняется именно благодаря Авито."
    if feature == "fresh__audiovisual_claim_alignment":
        # The supporting rubric may be in English and may mention its score.
        # Show the factual meaning of that rubric, never its internal rating.
        if not _russian_prose(text):
            return _alignment_copy(support, value)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(s for s in sentences if not re.search(
            r"\b(rating|score|оценк\w*|балл\w*|уровень\s+\d)\b", s, re.I))
        return text if _russian_prose(text) else _alignment_copy(support, value)
    return text


def feature_context(feature, value, absent):
    if feature == "state_transformation_present" and absent:
        return "В этом пункте важна связь перемены в жизни героя с использованием сервиса."
    if feature == "panel__distinct_benefit_count" and .5 <= value < 1.5:
        return "Здесь выделено одно основное обещание. Остальные детали стоит соотнести с ним."
    if feature == "panel__mandatory_inference_chain_length" and value < .5:
        return "Связь сюжета с предложением показана или проговорена прямо."
    if feature == "panel__offer_novelty_explanation_need" and value < .5:
        return "Здесь главное - ясно связать предложение с тем, что происходит в кадре."
    return ABSENCE[feature] if absent else PURPOSE[feature]


def plain_text(text):
    text = str(text or "").replace("—", "-").replace("–", "-")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\s*\((?:объект|действие|результат)\)", "", text, flags=re.I)
    text = re.sub(r"\b[Пп][эе]кшот(е|а)?\b", lambda m: {
        None: "финальный кадр с брендом", "е": "финальном кадре с брендом", "а": "финального кадра с брендом"
    }[m.group(1)], text)
    text = re.sub(r"\bоффер(ов|ы|а|ом)?\b", lambda m: {
        None: "предложение", "ов": "предложений", "ы": "предложения", "а": "предложения", "ом": "предложением"
    }[m.group(1)], text, flags=re.I)
    text = text.replace("Авито Услуги предоставляют мастеров", "Авито Услуги помогают найти мастеров")
    text = text.replace("дополнительных объяснений не требуется", "сложных для объяснения условий не выделено")
    text = text.replace("дополнительного объяснения не требуется", "сложных для объяснения условий не выделено")
    text = text.replace("Запоминающегося брендового джингла нет", "Фирменной мелодии нет")
    text = re.sub(r"(Финальный|финальный) кадр с брендом с логотипом", r"\1 кадр с логотипом", text)
    if text.startswith("Посыл конкретен:"):
        text = re.sub(r"(объект|действие|результат) \(([^)]+)\)", r"\2", text, flags=re.I)
        text = text.replace("Посыл конкретен: назван ", "В предложении названы ")
    text = text.replace("Финальный финальный", "Финальный").replace("финальный финальный", "финальный")
    return re.sub(r"\s+", " ", text).strip()


# A feature can explain the score only when both its evidence and direction are
# supported. Neutral scene descriptions are deliberately not rendered.
TAKEAWAYS = {
    'character_close_up_seconds': 'Крупные планы поддерживают заметность героя и его реакции.',
    'monologue_to_camera': 'Обращение к зрителю поддерживает заметность ключевой реплики.',
    'pack_shot_duration_seconds': 'Финальный кадр с брендом поддерживает заметность предложения.',
    'product_demo_seconds': 'Показ сервиса помогает выделить само предложение, а не только сюжет.',
    'state_transformation_present': 'Контраст между исходной ситуацией и результатом поддерживает заметность решения.',
    'main_character': 'Один главный герой помогает удержать фокус на одной задаче.',
    'numeric_offer_on_screen': 'Число на экране делает предложение более конкретным.',
    'has_callback_to_opening': 'Связь завязки и финала поддерживает цельность истории.',
    'jingle_present': 'Фирменная мелодия поддерживает звуковую связь с брендом.',
    'problem_solution_arc_present': 'Связь проблемы с решением помогает закрепить роль сервиса в истории.',
    'promo': 'Выделенная выгода акции поддерживает считываемость предложения.',
    'scene_pace_high': 'Темп смены сцен в этой версии поддерживает оценку запоминаемости.',
}
AUDIO_DEPENDENT = {
    'panel__first_core_claim_time_band', 'fresh__audiovisual_claim_alignment',
    'panel__distinct_benefit_count', 'panel__mandatory_inference_chain_length',
    'panel__message_specificity_level', 'panel__offer_novelty_explanation_need',
    'panel__product_role_reveal_time_band', 'brand_first_mention_seconds',
    'jingle_present', 'problem_solution_arc_present', 'monologue_to_camera',
}


def takeaway(feature, value, direction):
    positive = direction == 'supports'
    if feature == 'fresh__audiovisual_claim_alignment':
        if positive:
            return ('Изображение раскрывает обещание и поддерживает считываемость.' if value >= 2.5 else
                    'Показанная часть обещания поддерживает считываемость, хотя связь раскрыта не полностью.')
        return 'Связь изображения с главным обещанием ограничивает оценку считываемости.'
    if feature == 'panel__mandatory_inference_chain_length':
        return ('Прямое объяснение связи между сюжетом и предложением поддерживает считываемость.' if positive else
                'Неявная связь между событиями и выгодой ограничивает считываемость: часть смысла приходится достраивать.')
    if feature == 'panel__message_specificity_level':
        return ('Конкретное предложение поддерживает считываемость: есть опора для понимания услуги или выгоды.' if positive else
                'Общее обещание ограничивает считываемость: модель не выделяет достаточно конкретики о предложении.')
    if feature == 'panel__offer_novelty_explanation_need':
        return ('Отсутствие сложной для объяснения механики поддерживает считываемость.' if positive else
                'Непривычное предложение требует пояснения; в этой версии это ограничивает считываемость.')
    if feature == 'panel__first_core_claim_time_band':
        return ('Раннее появление главного предложения поддерживает его заметность.' if positive else
                'Позднее появление главного предложения оставляет меньше времени на него и ограничивает заметность.')
    if feature == 'panel__product_role_reveal_time_band':
        return ('Роль сервиса раскрывается достаточно рано и поддерживает считываемость.' if positive else
                'Роль сервиса становится понятна не сразу. Это ограничивает считываемость связи между историей и Авито.')
    if feature == 'brand_first_mention_seconds':
        return ('Раннее появление бренда поддерживает связь истории с Авито.' if positive else
                'Позднее появление бренда ограничивает связь истории с Авито в оценке запоминаемости.')
    if feature == 'panel__distinct_benefit_count':
        if value < 1.5:
            return 'Одно основное обещание поддерживает фокус сообщения.'
        return ('Несколько выделенных выгод поддерживают оценку этой версии. Это не означает, что добавление новых выгод её улучшит.' if positive else
                'Несколько разных выгод ограничивают фокус сообщения в оценке модели.')
    return TAKEAWAYS.get(feature, 'Эта деталь поддерживает оценку данной версии.')


def brief_evidence(text):
    """Keep one short justification, not the plot of the video."""
    text = plain_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    first = sentences[0]
    if len(first) > 220:
        first = first[:217].rsplit(' ', 1)[0].rstrip('.,;:') + '…'
    return first


def negative_takeaway(feature, value, label, task, directional):
    """Local score attribution does not imply a universal creative rule."""
    if directional and feature in {
        'fresh__audiovisual_claim_alignment', 'panel__mandatory_inference_chain_length',
        'panel__message_specificity_level', 'panel__offer_novelty_explanation_need',
        'panel__first_core_claim_time_band', 'panel__product_role_reveal_time_band',
        'brand_first_mention_seconds',
    }:
        return 'В конкретном ролике: ' + takeaway(feature, value, 'limits')
    metric = {'n': 'заметности', 'm': 'считываемости', 'r': 'запоминаемости'}.get(task, 'ролика')
    return f'В конкретном ролике «{label}» снижает оценку {metric}.'


def report_cards(interpretation):
    cards = []
    incomplete = interpretation.get('audio_status') in {'partial', 'absent'}
    for card in interpretation['cards']:
        task = card.get('task')
        drivers = interpretation.get('details', {}).get(task, {}).get('drivers')
        if drivers is None:
            drivers = [*card.get('strengths', []), *card.get('limitations', [])]
        groups = {'strengths': [], 'limitations': []}
        uncertain = {k: [] for k in ['audio', 'unverified', 'unstable', 'association', 'production']}
        for d in drivers:
            feature = d.get('feature', '')
            negative = d.get('direction') == 'limits'
            minimum = NEGATIVE_MIN_INDEX_POINTS if negative else 1
            if feature in {'brand_history', 'is_celeb'} or abs(d.get('index_points', 0)) < minimum:
                continue
            label = LABELS.get(feature, d.get('label', feature))
            if feature.startswith('phys__'):
                if abs(d.get('index_points', 0)) >= 2:
                    uncertain['production'].append(label)
                continue
            if incomplete and feature in AUDIO_DEPENDENT:
                uncertain['audio'].append(label)
                continue
            support = d.get('evidence', {})
            if not support.get('verified'):
                if abs(d.get('index_points', 0)) >= 2:
                    uncertain['unverified'].append(label)
                continue
            supported = d.get('usable') or (negative and
                d.get('stable_fraction', 0) >= NEGATIVE_MIN_STABILITY)
            if not supported:
                if abs(d.get('index_points', 0)) >= 2:
                    uncertain['unstable'].append(label)
                continue
            if feature not in PURPOSE:
                continue
            value = d.get('value', 0)
            absent = feature in ABSENCE and value < (.5 if feature not in {
                'character_close_up_seconds', 'product_demo_seconds'} else 1e-9)
            directional = d.get('interpretation_kind') == 'observed_driver'
            if feature == 'panel__distinct_benefit_count' and value < .5:
                directional = False
            if d.get('direction') != 'supports' and (absent or feature in {
                    'character_close_up_seconds', 'pack_shot_duration_seconds',
                    'product_demo_seconds', 'panel__distinct_benefit_count'} or
                    (feature == 'panel__message_specificity_level' and value >= 1.5)):
                directional = False
            local_negative = negative and (feature in ABSENCE or feature == 'pack_shot_duration_seconds'
                                          or (feature == 'panel__distinct_benefit_count' and value >= 1.5))
            if not directional and not local_negative:
                if abs(d.get('index_points', 0)) >= 2:
                    uncertain['association'].append(label)
                continue
            group = 'strengths' if d.get('direction') == 'supports' else 'limitations'
            if negative:
                label = observed_label(feature, value)
            groups[group].append(dict(feature=feature, label=label,
                takeaway=(negative_takeaway(feature, value, label, task, directional) if negative
                          else takeaway(feature, value, d.get('direction'))),
                evidence=brief_evidence(observation_text(feature, support, value)),
                episodes=support.get('episodes', [])[:1],
                check=plain_text(d.get('check')) if group == 'limitations' else ''))
        if task == 'r' and interpretation.get('celebrity_present') is True:
            groups['strengths'].insert(0, dict(
                feature='manual_celebrity', label='Участие медийной персоны',
                takeaway='В конкретном ролике участие медийной персоны положительно влияет на оценку запоминаемости.',
                evidence='Участие медийной персоны отмечено вами.', episodes=[], check=''))
        notes = []
        reasons = {
            'audio': 'Без полной озвучки нельзя уверенно оценить эти стороны сообщения. Выводы по ним стоит отложить до версии с речью.',
            'unverified': 'По этим деталям не удалось уверенно подтвердить основание оценки. Поэтому они не названы ни сильными, ни слабыми сторонами.',
            'unstable': 'Влияние этих деталей на оценку неустойчиво. Оснований рекомендовать их изменение пока недостаточно.',
            'association': 'Модель учитывает эти детали, но не даёт понятного основания считать их достоинствами или недостатками. Менять их только ради балла не стоит.',
            'production': 'Черновой звук и анимация влияют на оценку файла. По ним нельзя судить о качестве будущего готового ролика или советовать менять громкость и темп.',
        }
        if interpretation.get('material_kind') != 'neuromatics':
            reasons['production'] = 'Модель учитывает звук и движение, но эти измерения сами по себе не объясняют, что стоит изменить в подаче. По ним нельзя советовать менять громкость или темп.'
        for kind, labels in uncertain.items():
            if labels:
                notes.append(dict(reason=kind, features=list(dict.fromkeys(labels)), text=reasons[kind]))
        summary = ('Эта часть оценки объясняется лишь частично. Ниже - подтверждённые плюсы и ограничения.' if notes and any(groups.values()) else
                   'По этой части пока нет надёжного объяснения сильных и слабых сторон. Это не означает, что у ролика их нет.' if not any(groups.values()) else
                   'Эти детали объясняют оценку данной версии; эффект правок нужно проверять отдельно.')
        cards.append(dict(card, description=DESCRIPTIONS.get(task, ''), summary=summary, **groups, unassessed=notes))
    return cards
