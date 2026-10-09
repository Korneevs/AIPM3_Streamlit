"""Fast offline behavior checks: no video, training or real API requests."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

from aipm3.neuromatics_manager import build_manager_draft, manager_profile_rows, draft_fingerprint
from aipm3.interpretation_checker import apply_review, review_draft, validate_reply, packet_for, VERSION


def driver(feature, value=1, contribution=.01, verified=True):
    return dict(feature=feature, label=feature, value=value, contribution=contribution,
                stable_fraction=1, evidence=dict(verified=verified, observation="Это прямо показано в ролике."),
                factual_observation=None)


def explanation(drivers=None):
    drivers = drivers or [driver("state_transformation_present")]
    return dict(material_kind="neuromatics", audio_status="complete",
                details={t: dict(drivers=deepcopy(drivers), actual=.2) for t in "nmr"})


def reply_for(draft):
    return dict(items=[dict(id=r['id'], action='keep', finding=r['finding'], check=r['check'],
                           direction=r['direction'], support_ids=list(r['evidence_facts']), reason='Подтверждено.')
                       for rows in draft['profiles'].values() for r in rows])


class ManagerDraftTests(unittest.TestCase):
    def test_finished_uses_same_review_logic_without_rewriting_scores(self):
        source = explanation()
        source['material_kind'] = 'finished'
        before = deepcopy(source)
        draft = build_manager_draft(source)
        self.assertEqual(draft['version'], 'finished-manager-v1')
        self.assertEqual(draft['scores'], {t: .2 for t in 'nrm'})
        self.assertEqual(source, before)
        self.assertNotEqual(draft['fingerprint'], build_manager_draft(explanation())['fingerprint'])

    def test_finished_retains_measured_physical_inputs_without_claiming_causality(self):
        source = explanation([driver('phys__motion_mean', .04, -.1),
                              driver('phys__audio_dynamic_range_db', 12, .1),
                              driver('scene_pace_high', 0, -.1)])
        source['material_kind'] = 'finished'
        rows = build_manager_draft(source)['profiles']['n']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['direction'] == 'balanced' for row in rows))
        self.assertTrue(all('нейроматик' not in row['finding'] for row in rows))

    def test_missing_finished_feature_is_not_renamed_to_neuromatic(self):
        source = explanation([driver('character_close_up_seconds', 0, -.1, False)])
        source['material_kind'] = 'finished'
        row = build_manager_draft(source)['profiles']['n'][0]
        self.assertEqual(row['status'], 'unassessed')
        self.assertIn('этому ролику', row['finding'])

    def test_optional_absence_is_not_a_flaw(self):
        source = explanation([driver(f, 0, -.1) for f in
                              ['monologue_to_camera', 'promo', 'main_character', 'jingle_present']])
        draft = build_manager_draft(source)
        for rows in draft['profiles'].values():
            self.assertTrue(all(r['direction'] == 'balanced' and not r['check'] for r in rows))

    def test_good_alignment_cannot_be_called_a_defect(self):
        d = driver('fresh__audiovisual_claim_alignment', 3, -.1)
        d['evidence'].update(shown_action=True, shown_result=True)
        draft = build_manager_draft(explanation([d]))
        row = draft['profiles']['m'][0]
        self.assertEqual(row['direction'], 'balanced')
        self.assertIn('не выявлено', row['finding'])

    def test_unverified_contribution_has_no_color_direction(self):
        draft = build_manager_draft(explanation([driver('jingle_present', 1, .1, False)]))
        row = draft['profiles']['r'][0]
        self.assertEqual((row['direction'], row['status']), ('balanced', 'unassessed'))

    def test_contradictory_scoring_input_never_replaces_factual_observation(self):
        d = driver('state_transformation_present', 0, -.1, False)
        d['factual_observation'] = dict(value=1, evidence=dict(verified=True,
                                                            observation='Показан результат помощи сервиса.'))
        source = explanation([d])
        before = deepcopy(source)
        row = build_manager_draft(source)['profiles']['n'][0]
        self.assertEqual(row['direction'], 'balanced')
        self.assertIn('перемена', row['finding'])
        self.assertEqual(source, before)

    def test_incomplete_speech_does_not_prove_absence(self):
        source = explanation([driver('promo', 0, -.1)])
        source['audio_status'] = 'partial'
        row = build_manager_draft(source)['profiles']['m'][0]
        self.assertEqual(row['status'], 'unassessed')
        self.assertIn('Без полной речи', row['finding'])
        self.assertNotIn('не заявлены', row['finding'])

    def test_excluded_features_stay_excluded(self):
        source = explanation([driver(f) for f in ['state_transformation_present', 'is_celeb',
                            'has_callback_to_opening', 'panel__offer_novelty_explanation_need',
                            'panel__product_role_reveal_time_band', 'brand_history', 'phys__motion_mean']])
        draft = build_manager_draft(source)
        self.assertEqual({r['feature'] for rows in draft['profiles'].values() for r in rows},
                         {'state_transformation_present'})

    def test_grouping_retains_feature_coverage(self):
        source = explanation([driver('panel__message_specificity_level', 3),
                              driver('panel__mandatory_inference_chain_length', 0)])
        draft = build_manager_draft(source)
        self.assertEqual(len(draft['profiles']['m']), 1)
        self.assertEqual(len(draft['profiles']['m'][0]['features']), 2)
        self.assertEqual(draft['coverage']['m']['supported'], 2)

    def test_video_and_audio_changes_invalidate_review(self):
        source = explanation()
        a = build_manager_draft(source, source_sha='a' * 64)
        b = build_manager_draft(source, source_sha='b' * 64)
        self.assertNotEqual(a['fingerprint'], b['fingerprint'])
        source['audio_status'] = 'partial'
        c = build_manager_draft(source, source_sha='a' * 64)
        self.assertNotEqual(a['fingerprint'], c['fingerprint'])

    def test_missing_offer_is_not_described_as_late(self):
        draft = build_manager_draft(explanation([driver('panel__first_core_claim_time_band', 4, -.1)]))
        row = draft['profiles']['n'][0]
        self.assertIn('не подтверждён', row['finding'])
        self.assertNotIn('появляется во второй', row['finding'])

    def test_timing_uses_actual_duration(self):
        source = explanation([driver('panel__first_core_claim_time_band', 2, .1)])
        short = build_manager_draft(source, duration=20)['profiles']['n'][0]
        long = build_manager_draft(source, duration=40)['profiles']['n'][0]
        self.assertIn('во второй половине', short['finding'])
        self.assertNotIn('во второй половине', long['finding'])

    def test_partial_agreement_does_not_prove_action_is_absent(self):
        rows = []
        for repeat, action in enumerate(['not_shown', 'not_shown', 'shown']):
            rows.append(dict(source_sha='a' * 64, prepared_sha256='b' * 64, repeat=repeat,
                request_id='observation-' + str(repeat), alignment_evidence=dict(
                    request_id='alignment-' + str(repeat), values=dict(relation='related',
                        main_phrase='Ищите мастеров на Авито Услугах',
                        action_evidence=dict(status=action), result_evidence=dict(status='shown')))))
        draft = build_manager_draft(explanation([driver('fresh__audiovisual_claim_alignment', 2, -.1, False)]),
                                    evidence=rows, validate_alignment_record=lambda *_: None)
        fact = draft['profiles']['m'][0]['evidence_facts']['fresh__audiovisual_claim_alignment']
        self.assertIsNone(fact['evidence']['shown_action'])
        self.assertTrue(fact['evidence']['shown_result'])
        self.assertEqual(fact['evidence']['main_phrase'], 'Ищите мастеров на Авито Услугах')
        packet_fact = packet_for(draft)['items'][0]['facts']['fresh__audiovisual_claim_alignment']
        self.assertIsNone(packet_fact['shown_action'])
        self.assertTrue(packet_fact['main_phrase'])

    def test_equivalent_serialized_numbers_reuse_the_same_review(self):
        source = explanation()
        original = build_manager_draft(source, duration=20)
        source['details']['n']['actual'] += 1e-15
        equivalent = build_manager_draft(source, duration=20.0)
        self.assertEqual(original['fingerprint'], equivalent['fingerprint'])
        source['details']['n']['actual'] += .01
        changed = build_manager_draft(source, duration=20.0)
        self.assertNotEqual(original['fingerprint'], changed['fingerprint'])
        self.assertNotEqual(draft_fingerprint({'fact': True}), draft_fingerprint({'fact': 1}))


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.source = explanation()
        self.draft = build_manager_draft(self.source, source_sha='a' * 64)
        self.reply = reply_for(self.draft)

    def test_valid_review_does_not_change_scores_or_source(self):
        source = dict(self.source, manager_semantic=self.draft)
        before = deepcopy(source)
        receipt = dict(version=VERSION, status='reviewed', fingerprint=self.draft['fingerprint'], reply=self.reply)
        reviewed = apply_review(source, receipt)
        self.assertEqual(reviewed['details'], before['details'])
        self.assertEqual(source, before)
        self.assertTrue(reviewed['manager_semantic']['reviewed'])

    def test_cannot_reverse_direction(self):
        self.reply['items'][0]['direction'] = 'down'
        with self.assertRaises(ValueError):
            validate_reply(self.reply, self.draft)

    def test_cannot_invent_fact_references(self):
        self.reply['items'][0]['support_ids'] = ['unknown_fact']
        with self.assertRaises(ValueError):
            validate_reply(self.reply, self.draft)

    def test_checker_cannot_turn_unknown_showing_into_absence(self):
        item = driver('fresh__audiovisual_claim_alignment', 2, -.1)
        item['evidence'].update(shown_action=None, shown_result=True)
        draft = build_manager_draft(explanation([item]))
        reply = reply_for(draft)
        reply['items'][0].update(action='rewrite', finding='Действие с сервисом не показано. Нужно переделать связку с результатом.')
        with self.assertRaisesRegex(ValueError, 'unconfirmed_is_not_absent'):
            validate_reply(reply, draft)

    def test_cannot_remove_rows_or_duplicate_ids(self):
        self.reply['items'].pop()
        with self.assertRaises(ValueError):
            validate_reply(self.reply, self.draft)

    def test_unsafe_copy_is_rejected(self):
        for text in ['Добавьте скидку, тогда каждый зритель точно запомнит предложение.',
                     'Доказанный эффект вырастет на 20%, если изменить ролик.',
                     'SHAP показывает вклад признака в оценку модели.',
                     'Авито гарантирует качество и скорость услуги.',
                     '<script>Это дополнительное пояснение для менеджера.</script>']:
            with self.subTest(text=text):
                reply = deepcopy(self.reply)
                reply['items'][0].update(action='rewrite', finding=text)
                with self.assertRaises(ValueError):
                    validate_reply(reply, self.draft)

    def test_unobserved_fact_cannot_acquire_an_interpretation(self):
        draft = build_manager_draft(explanation([driver('jingle_present', 1, .1, False)]))
        reply = reply_for(draft)
        reply['items'][0].update(action='rewrite', finding='Фирменная мелодия надёжно связывает финал с брендом.')
        with self.assertRaises(ValueError):
            validate_reply(reply, draft)

    def test_stale_review_is_ignored(self):
        source = dict(self.source, manager_semantic=self.draft)
        reviewed = apply_review(source, dict(version=VERSION, status='reviewed', fingerprint='stale', reply=self.reply))
        self.assertEqual(reviewed, source)

    def test_withheld_claim_stays_visible_without_false_direction(self):
        self.source = explanation([driver('state_transformation_present'), driver('main_character'),
                                   driver('brand_first_mention_seconds', 5)])
        self.draft = build_manager_draft(self.source)
        self.reply = reply_for(self.draft)
        self.reply['items'][0].update(action='withhold', direction='balanced', support_ids=[])
        reviewed = apply_review(dict(self.source, manager_semantic=self.draft),
            dict(version=VERSION, status='reviewed', fingerprint=self.draft['fingerprint'], reply=self.reply))
        row = manager_profile_rows(reviewed)['n'][0]
        self.assertEqual((row['status'], row['direction']), ('unassessed', 'balanced'))

    def test_checker_cannot_reduce_a_supported_profile_below_half(self):
        for item in self.reply['items']:
            item.update(action='withhold', direction='balanced', support_ids=[])
        with self.assertRaisesRegex(ValueError, 'too thin'):
            validate_reply(self.reply, self.draft)

    def test_single_text_call_then_cache_hit(self):
        response = types.SimpleNamespace(model='test-model', id='test-request', usage=None,
            choices=[types.SimpleNamespace(finish_reason='stop',
                message=types.SimpleNamespace(content=json.dumps(self.reply, ensure_ascii=False)))])
        calls = []
        class Client:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def create(self, **kwargs):
                calls.append(kwargs)
                return response
        with TemporaryDirectory() as root, patch.dict('sys.modules', {'openai': types.SimpleNamespace(OpenAI=Client)}):
            first = review_draft(self.draft, api_key='fake', cache_dir=root)
            second = review_draft(self.draft, api_key='fake', cache_dir=root)
        self.assertEqual(first['status'], 'reviewed')
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(isinstance(m['content'], str) for m in calls[0]['messages']))
        self.assertNotIn('image_url', json.dumps(calls))

    def test_bad_llm_reply_is_not_published_or_cached(self):
        reply = deepcopy(self.reply)
        reply['items'][0].update(action='rewrite', finding='Признак автоматически повышает оценку на 10%.')
        response = types.SimpleNamespace(model='test', id='test', usage=None, choices=[
            types.SimpleNamespace(finish_reason='stop', message=types.SimpleNamespace(content=json.dumps(reply)))])
        class Client:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_: response))
            def __enter__(self): return self
            def __exit__(self, *_): pass
        with TemporaryDirectory() as root, patch.dict('sys.modules', {'openai': types.SimpleNamespace(OpenAI=Client)}):
            receipt = review_draft(self.draft, api_key='fake', cache_dir=root)
            self.assertEqual(receipt['status'], 'unavailable')
            self.assertNotIn('reply', receipt)
            self.assertFalse([p for p in Path(root).rglob('*.json') if not p.name.endswith('.rejected.json')])


if __name__ == '__main__':
    unittest.main()
