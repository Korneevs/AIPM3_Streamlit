"""Deterministic explanations of the selected 9/8/7 model, with evidence gates.

Exact per-head Shapley values explain predictions, not causal edit effects.
Independent video observations corroborate facts but never alter a score.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from .latest_manual_inputs import effective_feature_rows, celebrity_presence, recall_multiplier

VERSION = 'manager-decisions-v8-local-criticism-manual-celebrity'
NAMES = {'n':'Заметность', 'm':'Считываемость', 'r':'Запоминаемость'}
LABELS = {
 'character_close_up_seconds':'Крупные планы героя',
 'is_celeb':'Участие медийного героя',
 'monologue_to_camera':'Обращение к зрителю',
 'pack_shot_duration_seconds':'Финальный кадр с брендом',
 'panel__first_core_claim_time_band':'Когда появляется главное предложение',
 'phys__audio_dynamic_range_db':'Перепады громкости',
 'phys__motion_mean':'Движение в кадре',
 'product_demo_seconds':'Демонстрация сервиса',
 'state_transformation_present':'Изменение состояния героя или объекта',
 'fresh__audiovisual_claim_alignment':'Связь изображения и озвучки',
 'main_character':'Один главный герой',
 'numeric_offer_on_screen':'Числа в предложении на экране',
 'panel__distinct_benefit_count':'Число разных выгод',
 'panel__mandatory_inference_chain_length':'Что зрителю приходится додумывать',
 'panel__message_specificity_level':'Конкретность предложения',
 'panel__offer_novelty_explanation_need':'Сложность объяснения предложения',
 'panel__product_role_reveal_time_band':'Когда понятна роль сервиса',
 'promo':'Акция или бонус',
 'brand_first_mention_seconds':'Первое появление бренда',
 'brand_history':'Исторический ориентир бренда',
 'has_callback_to_opening':'Возвращение к началу в финале',
 'human_characters_count':'Количество персонажей',
 'jingle_present':'Фирменная мелодия',
 'problem_solution_arc_present':'Связь проблемы с решением',
 'scene_pace_high':'Темп смены сцен',
}
CHECKS = {
 'character_close_up_seconds':'Проверьте, помогают ли крупные планы следить за основной историей.',
 'is_celeb':'Проверьте узнаваемость персонажа; добавление знаменитости само по себе не гарантирует улучшения.',
 'monologue_to_camera':'Проверьте, понятно ли, кому адресована реплика героя.',
 'pack_shot_duration_seconds':'Посмотрите финал в обычном темпе: хватает ли времени прочитать бренд и предложение.',
 'panel__first_core_claim_time_band':'Проверьте, можно ли понять главную выгоду до финала ролика.',
 'phys__audio_dynamic_range_db':'Прослушайте переходы между речью, музыкой и финалом на одной громкости.',
 'phys__motion_mean':'Проверьте, остаются ли главный объект и предложение заметными во время движения.',
 'product_demo_seconds':'Проверьте, видно ли действие с сервисом, которое приводит к обещанной выгоде.',
 'state_transformation_present':'Проверьте, заметна ли разница между исходной ситуацией и результатом.',
 'fresh__audiovisual_claim_alignment':'Сопоставьте главную фразу с изображением в этот момент: показывается ли именно обещанная выгода.',
 'main_character':'Проверьте, понятно ли, чью задачу решает сервис.',
 'numeric_offer_on_screen':'Проверьте, успевает ли зритель прочитать число и понять, к чему оно относится.',
 'panel__distinct_benefit_count':'Попросите пересказать главную выгоду одной фразой и проверьте, не конкурируют ли с ней другие обещания.',
 'panel__mandatory_inference_chain_length':'Назовите промежуточный смысловой шаг прямо или покажите его действием, затем сравните версии.',
 'panel__message_specificity_level':'Проверьте, можно ли после просмотра сказать, что предлагает сервис и в чем выгода.',
 'panel__offer_novelty_explanation_need':'Проверьте, хватает ли объяснения, чтобы понять непривычную механику предложения.',
 'panel__product_role_reveal_time_band':'Проверьте, понятно ли до финала, как именно сервис помогает герою.',
 'promo':'Проверьте, отделена ли выгода акции от ее условий.',
 'brand_first_mention_seconds':'Проверьте, связано ли первое появление бренда с главным действием или выгодой.',
 'has_callback_to_opening':'Проверьте, помогает ли финал завершить ту же историю, которая началась в первых кадрах.',
 'human_characters_count':'Проверьте, можно ли без путаницы проследить роли персонажей.',
 'jingle_present':'Проверьте распознавание фирменного звука. Добавлять джингл только ради балла не стоит.',
 'problem_solution_arc_present':'Проверьте, показано ли, что проблему решает именно сервис.',
 'scene_pace_high':'Посмотрите ролик без паузы: хватает ли времени увидеть важные детали между сменами сцен.',
}
BINARY={'is_celeb','monologue_to_camera','state_transformation_present','main_character',
        'numeric_offer_on_screen','promo','has_callback_to_opening','jingle_present',
        'problem_solution_arc_present','scene_pace_high'}

def observed_label(feature,value):
    absent={'monologue_to_camera':'Нет обращения к зрителю','state_transformation_present':'Нет изменения состояния героя или объекта',
      'main_character':'Нет одного главного героя','numeric_offer_on_screen':'Нет числового предложения на экране',
      'promo':'Нет акции или бонуса','has_callback_to_opening':'Финал не возвращается к началу',
      'jingle_present':'Без фирменной мелодии','problem_solution_arc_present':'Нет отдельной истории «проблема - решение»',
      'scene_pace_high':'Невысокий темп смены сцен','character_close_up_seconds':'Без крупных планов героя',
      'product_demo_seconds':'Без демонстрации сервиса'}
    if feature in absent and value<(.5 if feature in BINARY else 1e-9):return absent[feature]
    return LABELS[feature]

def _predictor(head):
    s=head.state;sp=s['spec']
    def fn(z):
        z=np.asarray(z,float)
        p=(np.c_[np.ones(len(z)),z]@head.estimator if sp['learner']=='ridge'
           else head.estimator.predict(z))
        p=p*s['target_scale']+s['target_mean']
        if sp['target']=='log':p=np.exp(np.clip(p,-12,3))
        elif sp['target']=='rank':p=np.quantile(s['raw_y'],np.clip(p,0,1))
        else:assert sp['target']=='raw',sp['target']
        if sp.get('md_refinement'):
            p=s['calibration_intercept']+s['calibration_slope']*p
        return np.clip(p,.001,1 if sp['task']!='r' else max(1,max(s['raw_y'])*1.2))
    return fn

def design_predictor(head,frame,train):
    """Return aligned one-column-per-meaning designs and exact numeric scoring."""
    cols=head.state['columns']
    if callable(getattr(head,'predict_design',None)):
        # Adapted heads own their transformation and score conversion. Reusing
        # the finished-video SVR here would explain a different prediction.
        x=head.design(frame);bg=head.design(train);fn=head.predict_design
    elif 'members' not in head.state['spec']:
        x=head.design(frame);bg=head.design(train);fn=_predictor(head)
    else:
        members=[type(head)(s,e) for s,e in head.estimator]
        x=np.zeros((len(frame),len(cols)));bg=np.zeros((len(train),len(cols)))
        seen=set();functions=[]
        for m in members:
            ix=[cols.index(c) for c in m.state['columns']]
            a,b=m.design(frame),m.design(train)
            for k,j in enumerate(ix):
                if j in seen:
                    assert np.allclose(x[:,j],a[:,k]) and np.allclose(bg[:,j],b[:,k])
                else:x[:,j]=a[:,k];bg[:,j]=b[:,k];seen.add(j)
            functions.append((ix,_predictor(m)))
        def fn(z):return np.mean([f(np.asarray(z)[:,ix]) for ix,f in functions],axis=0)
    assert np.allclose(fn(x),head.predict(frame),atol=1e-12,rtol=0)
    return x,bg,fn

def exact_shap(x,background,predict):
    """All coalitions, empirical background. No random SHAP approximation."""
    p=x.shape[1];masks=((np.arange(2**p)[:,None]>>np.arange(p))&1).astype(bool)
    unique,inv=np.unique(x,axis=0,return_inverse=True);out=[];bases=[]
    sizes=masks.sum(1)
    for row in unique:
        v=[]
        for start in range(0,len(masks),32):
            a=np.where(masks[start:start+32,None,:],row,background[None,:,:])
            v.extend(predict(a.reshape(-1,p)).reshape(len(a),len(background)).mean(1))
        v=np.asarray(v);phi=np.zeros(p)
        for j in range(p):
            ix=np.flatnonzero(~masks[:,j]);w=np.array([1/(p*math.comb(p-1,int(sizes[k]))) for k in ix])
            phi[j]=np.sum(w*(v[ix|(1<<j)]-v[ix]))
        assert abs(phi.sum()+v[0]-v[-1])<1e-9
        out.append(phi);bases.append(v[0])
    return np.asarray(out)[inv],np.asarray(bases)[inv]

def _evidence(feature,rows,observed,duration):
    if feature=='is_celeb':
        return dict(verified=False,observation='Участие медийного героя нужно подтвердить по данным о съемке.',episodes=[],agreement=0.)
    if feature=='fresh__audiovisual_claim_alignment':
        return alignment_support(rows,observed,duration)
    if feature=='human_characters_count':
        return dict(verified=False,observation='В расчете используется автоматически оцененное число персонажей. Для объяснения этого вклада нужно сверить людей по кадрам: свободные описания иногда противоречат числу.',episodes=[],agreement=0.)
    matches=[next((x for x in r['values']['observations'] if x['feature']==feature),None) for r in rows]
    def valid_number(v):
        return isinstance(v,(float,int)) and (not isinstance(v,bool) or feature in BINARY) and math.isfinite(v)
    matches=[x for x in matches if x and x['status']!='uncertain' and valid_number(x['value'])]
    if feature.startswith('phys__'):
        return dict(verified=True,source='physical',observation=('Перепад между тихими и громкими участками: '
             +f'{observed:.1f} дБ.' if feature.endswith('range_db') else 'Движение измерено по последовательным кадрам ролика.'),episodes=[],agreement=1.)
    if feature=='brand_history':
        return dict(verified=True,source='historical_context',observation='Уровень запоминаемости бренда в исторических исследованиях; это не свойство монтажа или сюжета.',episodes=[],agreement=1.)
    if len(matches)<3 or not math.isfinite(observed):return dict(verified=False,observation='Для этого наблюдения нет трех согласованных проверок по видео.',episodes=[],agreement=len(matches)/max(1,len(rows)))
    target=math.floor(observed+.5) if feature.startswith(('panel__','fresh__')) else observed
    tolerance=.25 if feature in BINARY else max(1.,duration*.1) if feature.endswith('_seconds') else .5
    valid=[m for m in matches if abs(float(m['value'])-target)<=tolerance]
    if len(valid)<3:return dict(verified=False,observation='Повторные проверки расходятся между собой или с признаками, на которых рассчитан балл.',episodes=[],agreement=len(valid)/max(1,len(rows)))
    # Earliest independent repeat among agreeing observations; deterministic selection.
    m=valid[0];episodes=[e for e in m['episodes'] if 0<=e['start']<=e['end']<=duration+1]
    if has_person_name(m['observation']+' '+str(episodes)):
        return dict(verified=False,observation='В описании есть непроверенное имя или собственное название. Нужна проверка текста наблюдения.',episodes=[],agreement=0.)
    return dict(verified=True,observation=m['observation'],episodes=episodes,agreement=len(valid)/len(rows),source='independent_video_review')

def has_person_name(text):
    # Conservative language gate; never identifies a face or replaces it with a name.
    text=re.sub(r'\bАвито(?: (?:Авто|Работ[а-яё]*|Доставк[а-яё]*|Недвижимост[а-яё]*|Услуг[а-яё]*))?\b','',text)
    # A named fictional role in a holiday story is not the actor's identity.
    text=re.sub(r'\bДед(?:а|у|ом|е)? Мороз(?:а|у|ом|е)?\b','',text)
    return bool(re.search(r'\b[А-ЯЁ][а-яё]+(?:[- ][А-ЯЁ][а-яё]+){1,2}\b',text) or
      re.search(r'\b(Фоменко|Журавл[её]в|Куркова|Macan|MACAN|Макан)\b',text))

def factual_observation(feature, rows, duration):
    """A corroborated fact may be shown neutrally even when scoring disagrees.

    It never becomes a score explanation or a replacement scoring input.
    Reuse the same three-read, source, timestamp and identity gates.
    """
    if feature.startswith('phys__') or feature in {'brand_history', 'is_celeb', 'human_characters_count'}:
        return None
    if feature == 'fresh__audiovisual_claim_alignment':
        candidates = range(4)
    else:
        values = [x['value'] for row in rows for x in row['values']['observations']
                  if x['feature'] == feature and x['status'] != 'uncertain'
                  and isinstance(x['value'], (int, float)) and math.isfinite(x['value'])]
        if len(values) < 3:
            return None
        candidates = [float(np.median(values))]
    for value in candidates:
        support = _evidence(feature, rows, value, duration)
        if support['verified']:
            return dict(value=value, label=observed_label(feature, value), evidence=support)
    return None

def public_result(result,interpretation):
    """Keep scores and inputs; do not publish unsupported identities from free text."""
    out=deepcopy(result);out['interpretation']=deepcopy(interpretation)
    out.pop('evidence',None)
    if isinstance(out.get('metadata'),dict):out['metadata'].pop('evidence',None)
    for row in out.get('independent_evidence',[]):
        redacted=[]
        for key in ('synopsis','main_claim'):
            if has_person_name(row['values'].get(key,'')):
                row['values'][key]='Формулировка требует проверки.';redacted.append(key)
        for obs in row['values']['observations']:
            if obs['feature']=='is_celeb' or has_person_name(obs['observation']+' '+str(obs['episodes'])):
                obs.update(status='uncertain',value=None,episodes=[],observation='Наблюдение требует проверки по данным о съемке.')
                redacted.append(obs['feature'])
        if 'alignment_evidence' in row and has_person_name(str(row['alignment_evidence']['values'])):
            row.pop('alignment_evidence');redacted.append('alignment_evidence')
        row['public_redactions']=redacted
    return out

def alignment_support(rows,observed,duration):
    from .latest_alignment_evidence import validate_record
    from jsonschema.exceptions import ValidationError
    checked=[];ids=set()
    for parent in rows:
        row=parent.get('alignment_evidence')
        if not row:continue
        try:
            validate_record(row,parent['source_sha'],parent['prepared_sha256'],parent['repeat'],duration)
            if row['request_id'] in ids or row['request_id']==parent.get('request_id'):continue
            ids.add(row['request_id'])
            if row['values']['rating'] is not None:checked.append(row['values'])
        except (KeyError,TypeError,ValueError,ValidationError):continue
    target=math.floor(observed+.5) if math.isfinite(observed) else None
    valid=[r for r in checked if r['rating']==target]
    if len(valid)<3:
        message='Для этого признака нужно отдельно проверить показанные действие и результат; одного совпадения текста с озвучкой недостаточно.'
        if len(checked)>=2:
            ratings=[r['rating'] for r in checked]
            message='Проверки показанных действия и результата не подтверждают исходный разбор модели.'
            if not has_person_name(checked[0]['rating_reason']):message+=' '+checked[0]['rating_reason']
        return dict(verified=False,observation=message,episodes=[],agreement=len(valid)/3,source='structured_alignment_review')
    v=valid[0];episodes=v['action_evidence']['episodes']+v['result_evidence']['episodes']
    if not episodes:episodes=v['main_phrase_episodes']
    if has_person_name(v['rating_reason']+' '+str(episodes)):
        return dict(verified=False,observation='В описании есть непроверенное имя или собственное название. Нужна проверка текста наблюдения.',episodes=[],agreement=0.)
    phrase_key=lambda text:re.sub(r'[^а-яёa-z0-9]','',text.lower())
    same_phrase=len({phrase_key(x['main_phrase']) for x in valid})==1
    phrase=v['main_phrase'] if same_phrase and not has_person_name(v['main_phrase']) else None
    return dict(verified=True,observation=v['rating_reason'],episodes=episodes,
      agreement=len(valid)/3,source='structured_alignment_review', main_phrase=phrase,
      shown_action=(all(x['action_evidence']['status']=='shown' for x in valid)),
      shown_result=(all(x['result_evidence']['status']=='shown' for x in valid)))

def _reading_kind(feature,value,reference,contribution):
    """Distinguish arithmetic attribution from a defensible editorial check."""
    usual_sign={
      'fresh__audiovisual_claim_alignment':1,'panel__message_specificity_level':1,
      'panel__mandatory_inference_chain_length':-1,'panel__offer_novelty_explanation_need':-1,
      'panel__first_core_claim_time_band':-1,'panel__product_role_reveal_time_band':-1,
      'brand_first_mention_seconds':-1,
    }.get(feature)
    contradiction=usual_sign is not None and (value-reference)*usual_sign*contribution < -1e-8
    specific_good_but_lower=(contribution<0 and (
      (feature=='fresh__audiovisual_claim_alignment' and value>=2.5) or
      (feature=='panel__message_specificity_level' and value>=2.5) or
      (feature in ['panel__offer_novelty_explanation_need','panel__mandatory_inference_chain_length'] and value<.5) or
      (feature=='brand_first_mention_seconds' and value<=2)))
    if feature=='brand_history':return 'context','Исторический ориентир бренда учтен отдельно от содержания ролика.'
    if feature in BINARY and (value<.5 or contribution<0):
        return 'association_only','Модель учитывает наличие или отсутствие этого приема. Этот вклад сам по себе не означает, что прием нужно добавить или убрать.'
    if contradiction or specific_good_but_lower:
        return 'association_only','Направление вклада нельзя переводить в совет по креативу: это особенность связи в обученной модели. Не стоит менять понятную подачу ради этого балла.'
    if feature.startswith('phys__'):
        return 'association_only','Это характеристика видеоряда или звука, которую учитывает модель. По ней одной нельзя рекомендовать ускорение монтажа или изменение громкости.'
    return 'observed_driver','Наблюдение подтверждено повторными разборами. Оно объясняет часть оценки; результат правки нужно проверить на новой версии ролика.'

def editorial_check(feature,value):
    # Absence does not justify advice to add a creative device or an offer.
    if feature in BINARY and value<.5:return None
    if (feature.endswith('_seconds') or feature=='panel__distinct_benefit_count') and value<=0:return None
    return CHECKS.get(feature)

def semantic_measurement(head,frame,feature):
    """Compare video evidence to the value actually used by recall consensus."""
    if callable(getattr(head,'observed_measurement',None)):
        return head.observed_measurement(frame,feature)
    observed=float(pd.to_numeric(frame[feature],errors='coerce').mean()) if feature in frame else float('nan')
    sp=head.state['spec']
    if 'members' in sp:
        values=[]
        for state,estimator in head.estimator:
            if feature in state['spec'].get('consensus_features',[]) and feature in state['columns']:
                member=type(head)(state,estimator)
                values.append(float(member.raw(frame)[:,state['columns'].index(feature)].mean()))
        if values:
            if not np.allclose(values,values[0],equal_nan=True):raise ValueError('Conflicting recall consensus')
            return values[0],observed
    return observed,observed

def build_latest_interpretation(result,model=None,evidence=None,bundle_dir=None,strict_evidence=True):
    from .latest_runtime import material_kind_for_result
    from .display_calibration import neuromatics_reference, display_values, audio_status, audio_note
    material_kind=material_kind_for_result(result)
    calibration=neuromatics_reference(result['scoring_version']) if material_kind=='neuromatics' else None
    voice=audio_status(result) if material_kind=='neuromatics' else 'complete'
    if model is None or bundle_dir is None:
        from .latest_runtime import load_models,BUNDLE_DIR
        model=model or load_models(material_kind);bundle_dir=bundle_dir or BUNDLE_DIR
    evidence=evidence or [];details={};cards=[];rejected=0
    if strict_evidence:
        from .latest_evidence import validate_evidence
        nframe=pd.DataFrame(result['feature_rows']['n'])
        sha=str(nframe.sha.iloc[0]);duration=float(nframe['phys__duration'].iloc[0])
        valid=[];requests=set()
        for row in evidence:
            try:
                validate_evidence(row,sha,duration)
                if not row.get('request_id') or row['request_id'] in requests:raise ValueError('Repeated evidence request')
                requests.add(row['request_id']);valid.append(row)
            except (ValueError,KeyError,TypeError):rejected+=1
        evidence=valid
    effective_rows=effective_feature_rows(result['feature_rows'],result)
    for task in 'nmr':
        frame=pd.DataFrame(effective_rows[task]);head=model.heads[task]
        train=pd.read_csv(Path(bundle_dir)/'data'/f'fit_{task}.csv',float_precision='round_trip')
        # One real cached reading per family, selected without outcomes or query.
        order=[c for c in ['family','record','repeat'] if c in train]
        train=train.sort_values(order).groupby('family',sort=True).nth(0).reset_index(drop=True)
        x,bg,predict=design_predictor(head,frame,train)
        phi,bases=exact_shap(x,bg,predict);p=predict(x);base=float(bases.mean())
        score=float(p.mean());vals=phi.mean(0);columns=head.state['columns']
        spread=np.quantile(p,[.1,.9]);drivers=[]
        dur=float(pd.to_numeric(frame.get('phys__duration',frame.get('duration',frame.get('total_video_duration_sec',pd.Series([30])))),errors='coerce').mean())
        for j,c in enumerate(columns):
            # A fixed reference value is not an observed property of this video.
            if c in getattr(head,'neutralized_features',()):
                continue
            v,original_mean=semantic_measurement(head,frame,c)
            # Physical measurements describe the file as observed. An adapted
            # scoring value must not be presented as its literal sound level.
            adjusted_physical=(material_kind=='neuromatics' and c.startswith('phys__')
                and not np.isclose(v,original_mean,rtol=0,atol=1e-12))
            support=_evidence(c,evidence,original_mean if adjusted_physical else v,dur)
            if adjusted_physical:
                support['source']='physical_neuromatics_adjusted'
                support['observation']+=' В оценке нейроматика влияние крайних значений ограничено, чтобы черновая анимация или озвучка не определяли оценку идеи.'
            sign=1 if vals[j]>0 else -1
            stability=float(np.mean(phi[:,j]*sign>1e-10))
            material=abs(vals[j]/base)>=.01
            usable=support['verified'] and stability>=.8 and material and c!='brand_history'
            reference=float(pd.to_numeric(train[c],errors='coerce').median()) if c in train else float('nan')
            kind,why=_reading_kind(c,v,reference,float(vals[j]))
            driver=dict(feature=c,label=observed_label(c,v),value=v,contribution=float(vals[j]),
                index_points=float(100*vals[j]/base),direction='supports' if sign>0 else 'limits',
                stable_fraction=stability,usable=bool(usable),evidence=support,
                interpretation_kind=kind,why=why,reference_value=reference,
                original_measurement_mean=original_mean,
                factual_observation=factual_observation(c,evidence,dur) if not support['verified'] else None,
                check=editorial_check(c,v) if usable and sign<0 and kind=='observed_driver' else None)
            drivers.append(driver)
        drivers.sort(key=lambda d:-abs(d['contribution']))
        positives=[d for d in drivers if d['usable'] and d['direction']=='supports'][:2]
        negatives=[d for d in drivers if d['usable'] and d['direction']=='limits'][:2]
        unresolved=[dict(d, support_reason=(d['evidence']['observation'] if not d['evidence']['verified']
            else 'Направление вклада меняется между повторными оценками.'))
            for d in drivers if abs(d['index_points'])>=2 and
            (not d['evidence']['verified'] or d['stable_fraction']<.8)]
        total_mass=sum(abs(d['index_points']) for d in drivers)
        missed_mass=sum(abs(d['index_points']) for d in drivers if
            not d['evidence']['verified'] or d['stable_fraction']<.8)
        coverage=1-missed_mass/total_mass if total_mass else 1.
        partial=coverage<.75 or any(abs(d['index_points'])>=5 for d in unresolved)
        unadjusted_score=score
        factor=recall_multiplier(result) if task=='r' else 1.
        score*=factor;spread*=factor
        bgpred=predict(bg);q1,q2=np.quantile(bgpred,[1/3,2/3])
        level='Выше типичного уровня' if score>q2 else 'Ниже типичного уровня' if score<q1 else 'Типичный уровень'
        card=dict(task=task,title=NAMES[task],score=score,index=100*score/base,level=level,
          repeat_index_range=[100*float(z)/base for z in spread],baseline=base,
          strengths=positives,limitations=negatives,
          unresolved=[d['label'] for d in unresolved],unresolved_drivers=unresolved,
          explanation_status='partial' if partial else 'supported',
          verified_contribution_coverage=coverage)
        if calibration:
            display=calibration['references'][task]
            card.update(display_values(score,display))
            card['repeat_index_range']=[100*float(z)/display['mean'] for z in spread]
        card['assessment_available']=voice not in {'partial','absent'}
        adjustment=score-unadjusted_score
        details[task]=dict(drivers=drivers,expected=base,actual=score,
              manual_adjustment=adjustment,base_actual=unadjusted_score,
              additivity_error=float(abs(base+vals.sum()+adjustment-score)),
              per_repeat_scores=(p*factor).tolist(),per_repeat_shap=phi.tolist(),feature_order=columns)
        cards.append(card)
    from .latest_runtime import MESSAGE_BUNDLE_DIR, message_artifact_hashes
    message_artifact_hashes()
    reference=json.loads((MESSAGE_BUNDLE_DIR/'finished_display_reference.json').read_text())
    q=float(result['scores']['Q']);lo,hi=reference['Q_tertiles']
    overall=dict(title=('AIPM3.0 (для нейроматиков)' if material_kind=='neuromatics' else 'AIPM 3.0'),score=q,index=100*q/reference['Q_mean'],
      level='Выше типичного уровня' if q>hi else 'Ниже типичного уровня' if q<lo else 'Типичный уровень')
    if calibration:overall.update(display_values(q,calibration['references']['Q']))
    overall['assessment_available']=voice not in {'partial','absent'}
    interpretation=dict(version=VERSION,material_kind=material_kind,cards=cards,details=details,overall=overall,
      celebrity_present=celebrity_presence(result),
      display_calibration=calibration['version'] if calibration else None,audio_status=voice,audio_note=audio_note(voice),
      scale_note='100 - средняя оценка исторических роликов в этом компоненте. Это индекс модели, а не процент зрителей.',
      interpretation_note='Плюсы и ограничения объясняют расчет модели относительно исторических роликов. Проверки ниже помогут обсудить правки с Марком и ресерчем; эффект отдельной правки нужно проверить на новой версии.',
      repeat_note='Диапазон показывает разброс повторных разборов, а не доверительный интервал эффективности.',
      evidence_runs=len(evidence),evidence_rejected=rejected)
    from .neuromatics_manager import build_manager_draft
    from .latest_alignment_evidence import validate_record
    interpretation['manager_semantic']=build_manager_draft(interpretation,evidence,
        source_sha=result.get('source_sha'),duration=dur,validate_alignment_record=validate_record)
    from .manager_display import with_display_classes
    return with_display_classes(interpretation,result['scoring_version'])
