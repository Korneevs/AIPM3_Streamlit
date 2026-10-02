from copy import deepcopy

from aipm3.manager_report import LEVELS, report_cards


def driver(feature, *, value=1, verified=True, usable=True, kind='observed_driver', direction='supports'):
    return dict(feature=feature, value=value, label=feature, usable=usable,
                interpretation_kind=kind, direction=direction, index_points=5,
                evidence=dict(verified=verified, observation='В кадре показано действие.', episodes=[]), check=None)


def report(drivers, **kwargs):
    return report_cards(dict(cards=[dict(task='n', title='Заметность')],
                             details={'n': {'drivers': drivers}}, **kwargs))[0]


def test_reliable_features_have_a_conclusion_and_short_support_not_scene_inventory():
    drivers=[driver(f) for f in ['character_close_up_seconds','monologue_to_camera','pack_shot_duration_seconds','product_demo_seconds']]
    before=deepcopy(drivers); card=report(drivers)
    assert len(card['strengths'])==4
    assert all(item['takeaway'] and item['evidence'] for item in card['strengths'])
    assert 'observations' not in card and 'pending' not in card
    assert not card['limitations'] and drivers==before


def test_conflicting_fact_and_unstable_effect_are_never_score_reasons():
    d=driver('product_demo_seconds',verified=False)
    d['factual_observation']=dict(value=0,label='Без демонстрации',evidence=dict(verified=True,observation='Карточки',episodes=[]))
    card=report([d,driver('pack_shot_duration_seconds',usable=False)])
    assert not card['strengths'] and not card['limitations']
    assert {item['reason'] for item in card['unassessed']}=={'unverified','unstable'}


def test_optional_absence_specific_offer_and_duration_are_not_defects():
    ds=[driver('panel__message_specificity_level',value=2,direction='limits'),
        driver('product_demo_seconds',value=0,direction='limits'),
        driver('pack_shot_duration_seconds',value=4,direction='limits'),
        driver('jingle_present',value=0,kind='association_only',direction='limits')]
    card=report(ds)
    assert not card['limitations']
    assert len(card['unassessed'])==1 and card['unassessed'][0]['reason']=='association'
    assert 'только ради балла не стоит' in card['unassessed'][0]['text']


def test_partial_voice_blocks_audio_conclusions_but_preserves_visual_evidence():
    card=report([driver('brand_first_mention_seconds'),driver('jingle_present'),
                 driver('fresh__audiovisual_claim_alignment',value=3),driver('character_close_up_seconds')],audio_status='partial')
    assert [x['feature'] for x in card['strengths']]==['character_close_up_seconds']
    assert len(card['unassessed'][0]['features'])==3
    assert card['unassessed'][0]['reason']=='audio'


def test_alignment_and_direct_message_have_no_rubric_or_audience_guarantees():
    d=driver('fresh__audiovisual_claim_alignment',value=3)
    d['evidence']['observation']='The video shows the action and the result. Оценка 3.'
    item=report([d])['strengths'][0]
    assert item['evidence']=='Показаны действие и результат, а озвучка связывает их с основным предложением.'
    d=driver('panel__mandatory_inference_chain_length',value=0)
    d['evidence']['observation']='Смысл понятен каждому.'
    assert 'каждому' not in report([d])['strengths'][0]['evidence']


def test_quality_colors_are_red_yellow_green():
    assert [LEVELS[x][1] for x in ['Ниже типичного уровня','Типичный уровень','Выше типичного уровня']]==['#B42318','#805B00','#137547']
