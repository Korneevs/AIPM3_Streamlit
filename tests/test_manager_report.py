from copy import deepcopy

from aipm3.manager_report import LEVELS, report_cards


def driver(feature, *, value=1, verified=True, usable=True, kind='observed_driver', direction='supports'):
    return dict(feature=feature, value=value, label=feature, usable=usable,
                interpretation_kind=kind, direction=direction, index_points=5,
                evidence=dict(verified=verified, observation='В кадре показано действие.', episodes=[]),
                check=None)


def test_more_than_two_features_without_promoting_uncertainty_to_advice():
    drivers = [driver(feature) for feature in ['character_close_up_seconds', 'monologue_to_camera',
                'pack_shot_duration_seconds', 'product_demo_seconds']]
    drivers += [driver('state_transformation_present', verified=False),
                driver('jingle_present', value=0, kind='association_only', direction='limits'),
                driver('phys__motion_mean'), driver('brand_history'), driver('is_celeb')]
    interpretation = dict(cards=[dict(task='n', title='Заметность')], details={'n': {'drivers': drivers}})
    before = deepcopy(interpretation)
    card = report_cards(interpretation)[0]
    assert len(card['strengths']) == 4
    assert not card['limitations']
    assert [item['feature'] for item in card['observations']] == ['jingle_present']
    assert 'только ради оценки не нужно' in card['observations'][0]['context']
    assert len(card['pending']) == 1
    assert interpretation == before


def test_unstable_association_is_neutral_and_status_colors_follow_quality():
    d = driver('product_demo_seconds', usable=False, direction='limits')
    card = report_cards(dict(cards=[dict(task='n')], details={'n': {'drivers': [d]}}))[0]
    assert not card['limitations'] and len(card['observations']) == 1
    assert [LEVELS[x][0] for x in ['Ниже типичного уровня', 'Типичный уровень', 'Выше типичного уровня']] == ['Ниже нормы', 'В норме', 'Выше нормы']
    assert LEVELS['Ниже типичного уровня'][1] == '#B42318'
    assert LEVELS['Типичный уровень'][1] == '#805B00'
    assert LEVELS['Выше типичного уровня'][1] == '#137547'


def test_specific_offer_and_optional_absence_are_not_presented_as_defects():
    drivers = [driver('panel__message_specificity_level', value=2, direction='limits'),
               driver('product_demo_seconds', value=0, direction='limits'),
               driver('pack_shot_duration_seconds', value=4, direction='limits')]
    card = report_cards(dict(cards=[dict(task='m')], details={'m': {'drivers': drivers}}))[0]
    assert not card['limitations']
    assert len(card['observations']) == 3
    assert all(not x['check'] for x in card['observations'])


def test_independent_fact_is_never_promoted_to_a_score_reason():
    d = driver('product_demo_seconds', verified=False)
    d['factual_observation'] = dict(value=0, label='Без демонстрации сервиса',
        evidence=dict(verified=True, observation='Показаны условные карточки, а не приложение.', episodes=[]))
    card = report_cards(dict(cards=[dict(task='n')], details={'n': {'drivers': [d]}}))[0]
    assert not card['strengths'] and not card['limitations'] and not card['pending']
    assert card['observations'][0]['label'] == 'Без демонстрации сервиса'
    assert not card['observations'][0]['check']


def test_alignment_does_not_expose_english_rubric_or_internal_score():
    d = driver('fresh__audiovisual_claim_alignment', value=2)
    d['evidence']['observation'] = 'The action is shown. Therefore the rating is 2.'
    card = report_cards(dict(cards=[dict(task='m')], details={'m': {'drivers': [d]}}))[0]
    text = card['strengths'][0]['observation']
    assert 'rating' not in text and '2' not in text and 'действие' in text


def test_mixed_language_alignment_is_plain_russian_after_rubric_is_removed():
    d = driver('fresh__audiovisual_claim_alignment', value=3)
    d['evidence']['observation'] = 'The video shows the action and the result. Оценка 3.'
    card = report_cards(dict(cards=[dict(task='m')], details={'m': {'drivers': [d]}}))[0]
    assert card['strengths'][0]['observation'] == 'Показаны действие и результат, а озвучка связывает их с основным предложением.'


def test_missing_product_link_does_not_claim_that_the_hero_never_changes():
    d = driver('state_transformation_present', value=0, direction='limits', kind='association_only')
    d['evidence']['observation'] = 'Показана только суета.'
    card = report_cards(dict(cards=[dict(task='n')], details={'n': {'drivers': [d]}}))[0]
    item = card['observations'][0]
    assert 'именно благодаря Авито' in item['observation']
    assert 'только суета' not in item['observation']
    assert item['label'] == 'Изменение ситуации благодаря сервису'
    assert card['observations_title'] == 'Что видно в ролике'


def test_direct_message_describes_the_video_without_guaranteeing_understanding():
    d = driver('panel__mandatory_inference_chain_length', value=0)
    d['evidence']['observation'] = 'Смысл понятен каждому, додумывать не нужно.'
    card = report_cards(dict(cards=[dict(task='m')], details={'m': {'drivers': [d]}}))[0]
    assert card['strengths'][0]['observation'] == 'Связь сюжета с предложением проговорена или показана прямо.'
