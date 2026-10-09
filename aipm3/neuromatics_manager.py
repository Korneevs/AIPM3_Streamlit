"""Source-bound editorial explanations, separate from score attribution.

This layer does not measure, replace or change any scoring input. A direction
is shown only when both the arithmetic and the corroborated meaning support it.
Optional storytelling devices are never mandatory recommendations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

VERSION = "neuromatics-manager-v2"
FINISHED_VERSION = "finished-manager-v1"
ORDER = ("n", "r", "m")
OMIT = {"is_celeb", "brand_history", "has_callback_to_opening",
        "panel__offer_novelty_explanation_need", "panel__product_role_reveal_time_band"}
AUDIO = {"monologue_to_camera", "panel__first_core_claim_time_band",
         "fresh__audiovisual_claim_alignment", "promo", "brand_first_mention_seconds",
         "jingle_present", "panel__message_specificity_level",
         "panel__distinct_benefit_count", "panel__mandatory_inference_chain_length"}


def _fact(driver, audio_incomplete):
    feature = driver["feature"]
    if audio_incomplete and feature in AUDIO:
        return None
    evidence = driver.get("evidence", {})
    if evidence.get("verified"):
        return dict(value=driver["value"], text=evidence.get("observation", ""),
                    same_input=True, evidence=deepcopy(evidence))
    fact = driver.get("factual_observation")
    if fact and fact.get("evidence", {}).get("verified"):
        return dict(value=fact["value"], text=fact["evidence"].get("observation", ""),
                    same_input=False, evidence=deepcopy(fact["evidence"]))
    return None


def _direction(driver, fact, desired):
    if not fact or not fact["same_input"] or desired == "balanced":
        return "balanced"
    if driver.get("stable_fraction", 0) < 2 / 3:
        return "balanced"
    contribution = float(driver["contribution"])
    sign = "up" if contribution > 0 else "down" if contribution < 0 else "balanced"
    return desired if sign == desired else "balanced"


def _copy(feature, fact):
    """Return business meaning, finding, conditional check, semantic preference."""
    if not fact:
        return ("Эта деталь оценивается только по тому, что можно подтвердить в самом файле.",
                "По этому нейроматику нельзя уверенно назвать эту деталь сильной или слабой стороной.",
                "", "balanced")
    value = float(fact["value"])
    if feature == 'character_close_up_seconds':
        return ('Крупный план выделяет героя и его реакцию в истории.',
                ('Крупные планы героя подтверждены. Их стоит сохранять там, где реакция помогает понять задачу '
                 'или результат истории; само по себе увеличение их длительности не делает ролик сильнее.' if value > 0 else
                 'Крупные планы героя не подтверждены. Это не обязательный приём: добавлять их стоит, '
                 'если без реакции героя теряется смысл сцены.'), '', 'up' if value > 0 else 'balanced')
    if feature == 'pack_shot_duration_seconds':
        return ('Финальный кадр связывает историю с брендом и главным предложением.',
                ('Финальный кадр с брендом подтверждён. Важно сохранить в нём главное предложение различимым: '
                 'не перегружать его несколькими сообщениями.' if value > 0 else
                 'Отдельный финальный кадр с брендом не подтверждён. Это не означает, что бренд не считывается: '
                 'его связь с историей стоит проверить по всему ролику.'), '', 'up' if value > 0 else 'balanced')
    if feature == 'product_demo_seconds':
        return ('Показ использования помогает понять, что именно можно сделать с помощью сервиса.',
                ('Использование продукта или сервиса показано. Такой момент даёт конкретную опору предложению; '
                 'важно сохранить понятную связь между действием и задачей героя.' if value > 0 else
                 'Показ использования продукта или сервиса не подтверждён. Это не требует обязательно добавлять '
                 'интерфейс: стоит проверить, объясняет ли сама история, как сервис помогает.'), '',
                'up' if value > 0 else 'balanced')
    if feature == 'phys__audio_dynamic_range_db':
        return ('Контраст громкости может выделять важные моменты, но речь должна оставаться различимой.',
                'Громкость в записи меняется. По одному этому наблюдению нельзя назвать звук достоинством или недостатком. '
                'При проверке следующей версии стоит послушать, не теряется ли главное предложение на фоне музыки и эффектов.',
                '', 'balanced')
    if feature == 'phys__motion_mean':
        return ('Движение и смена планов должны помогать выделять главное действие.',
                'Изменения изображения учтены, но сами по себе не показывают, удачен ли темп ролика. '
                'Стоит проверить, остаются ли задача героя и предложение понятными при переходах между планами.',
                '', 'balanced')
    if feature == 'scene_pace_high':
        return ('Темп смены сцен помогает вести историю, если за переходами сохраняется основная мысль.',
                ('Сцены сменяются быстро. Это может поддерживать ритм, но не является преимуществом само по себе: '
                 'стоит проверить, понятна ли связь между событиями и предложением.' if value >= .5 else
                 'Быстрая смена сцен не подтверждена. Спокойный темп не считается недостатком: '
                 'ускорять монтаж стоит только при понятной задаче для истории.'), '', 'balanced')
    if feature == "monologue_to_camera":
        if value < .5:
            return ("Прямое обращение в камеру - один из способов донести предложение.",
                    "Здесь предложение передаётся через сюжет, без прямого обращения к зрителю. "
                    "Само по себе это не недостаток: добавлять такое обращение без задачи не нужно.", "", "balanced")
        return ("Прямое обращение может выделить главное предложение.",
                "В ролике есть прямое обращение к зрителю. Важно, чтобы реплика помогала понять предложение, "
                "а не дублировала то, что уже раскрыто историей.", "", "up")
    if feature == "state_transformation_present":
        if value >= .5:
            return ("Контраст «до и после» делает результат использования сервиса видимым.",
                    "В истории показана перемена от исходной трудности к результату после обращения к сервису. "
                    "Это даёт конкретный повод обратить внимание на его роль; такую связь стоит сохранить.", "", "up")
        return ("Перемена должна быть связана с сервисом, а не только с новой сценой.",
                "Уверенная связь «было трудно - сервис помог - стало лучше» здесь не подтверждена. "
                "Если результат - основа идеи, его стоит сделать заметнее.",
                "Сравните вариант, в котором результат помощи сервиса показан яснее.", "down")
    if feature == "panel__first_core_claim_time_band":
        if value >= 3.5:
            return ("Важен момент, когда становится понятно, что предлагает сервис.",
                    "Момент с явным главным предложением не подтверждён. Стоит проверить, раскрывает ли история "
                    "не только ситуацию, но и конкретную помощь сервиса.",
                    "Попросите после просмотра назвать главное предложение без дополнительной подсказки.", "down")
        band = max(0, min(3, math.floor(value + .5)))
        start = (0, 5, 10, 15)[band]
        late = start >= float(fact.get('duration', 30)) / 2
        return ("Важен момент, когда становится понятно, что предлагает сервис.",
                ("Главное предложение появляется во второй половине ролика. Сначала внимание держит история; "
                 "стоит проверить, не теряется ли связь с сервисом до этого момента." if late else
                 "Главное предложение раскрывается по ходу истории. У неё появляется опора в том, "
                 "что именно предлагает сервис; этот момент стоит сохранить."),
                "Сравните версию с более ранней подсказкой о главном предложении, сохранив сюжет." if late else "", "down" if late else "up")
    if feature == "fresh__audiovisual_claim_alignment":
        evidence = fact["evidence"]
        action, result = evidence.get("shown_action"), evidence.get("shown_result")
        if action is True and result is True:
            return ("Картинка должна подтверждать главное предложение из озвучки.",
                    "Озвученное предложение поддержано показом действия и его результата. "
                    "Противоречия между основным обещанием и картинкой здесь не выявлено; эту связку стоит сохранить.", "", "up")
        if result is True:
            return ("Картинка должна подтверждать главное предложение из озвучки.",
                    "Результат показан, но путь к нему через сервис подтверждён не полностью. "
                    "Это повод проверить понятность связи, а не требование показывать настоящий интерфейс.",
                    "Покажите без пояснений: понятно ли, как действие с сервисом приводит к результату?", "down")
        if action is True:
            return ("Картинка должна подтверждать главное предложение из озвучки.",
                    "Действие с сервисом показано; подтверждения обещанного результата не хватает. "
                    "Если результат - главное обещание, стоит проверить его визуальную опору.",
                    "Сравните версию с более явным показом результата.", "down")
        return ("Картинка должна подтверждать главное предложение из озвучки.",
                "Нельзя уверенно подтвердить, что изображение раскрывает основное обещание. "
                "Отдельный недостаток без сверки озвучки и картинки здесь не утверждаем.",
                "Сопоставьте основную фразу с изображением в этот момент.", "balanced")
    if feature == "main_character":
        if value >= .5:
            return ("Важнее ясность ролей, чем обязательное наличие одного героя.",
                    "У истории есть один центральный герой. Это помогает связать задачу и результат в одной линии; "
                    "его роль стоит сохранить ясной.", "", "up")
        return ("Важнее ясность ролей, чем обязательное наличие одного героя.",
                "История распределена между несколькими героями. Само по себе это не минус: сокращать их число "
                "стоит только при путанице в том, кому и как помогает сервис.", "", "balanced")
    if feature == "promo":
        return ("Акция - отдельное обещание, если она входит в задачу кампании.",
                ("Акция или бонус не заявлены. Ролик может работать на обычную выгоду сервиса; "
                 "отсутствие скидки не делает предложение слабым." if value < .5 else
                 "В ролике есть акция или бонус. Их смысл и условия должны быть различимы отдельно от обычной выгоды сервиса."),
                "", "balanced")
    if feature == "panel__message_specificity_level":
        if value >= 2:
            return ("Конкретное предложение объясняет, что можно сделать с помощью сервиса.",
                    "Предложение раскрыто через конкретную задачу или действие, а не только через название бренда. "
                    "Это даёт содержательную опору для выбора сервиса; конкретность стоит сохранить.", "", "up")
        return ("Конкретное предложение объясняет, что можно сделать с помощью сервиса.",
                "Предложение остаётся общим: конкретное действие или задача раскрыты недостаточно. "
                "Стоит проверить, можно ли после просмотра назвать, зачем здесь сервис.",
                "Сравните вариант с одной конкретной задачей, которую помогает решить сервис.", "down")
    if feature == "panel__distinct_benefit_count":
        examples = bool(re.search(r"маникюр|Дед[а]? Мороз|мастеров красоты|мастеров праздника", fact["text"], re.I))
        if examples:
            return ("Разные примеры услуг не обязательно означают разные выгоды.",
                    "Показаны разные задачи, с которыми можно обратиться к сервису. Это раскрывает выбор специалистов, "
                    "но само число примеров не делает предложение сильнее. Важно, чтобы они поддерживали одну основную мысль.", "", "balanced")
        return ("Важно, какие причины выбрать сервис названы, а не сколько их всего.",
                ("В ролике раскрыты несколько выгод. Они должны поддерживать главное предложение, "
                 "а не конкурировать за внимание." if value > 1 else
                 "Акцент сделан на одной основной выгоде. Добавлять другие обещания только ради их количества не нужно."),
                "", "balanced")
    if feature == "panel__mandatory_inference_chain_length":
        return ("Связь между событиями и предложением не должна требовать лишних догадок.",
                ("Основная связь раскрыта прямо. Дополнительные объяснения стоит добавлять только там, "
                 "где действительно возникает вопрос о роли сервиса." if value < .5 else
                 "В истории есть переход, который зрителю нужно связать с предложением самостоятельно. "
                 "Это повод проверить понятность перехода, а не считать любой образ или метафору ошибкой."),
                "Попросите пересказать связь события с предложением без дополнительной подсказки." if value >= .5 else "",
                "up" if value < .5 else "down")
    if feature == "brand_first_mention_seconds":
        return ("Бренд должен быть связан с задачей и результатом истории.",
                "Появление бренда подтверждено. Для следующей версии важнее сохранить связь названия с помощью сервиса, "
                "чем сдвигать его раньше или позже только ради времени появления.", "", "balanced")
    if feature == "jingle_present":
        return ("Фирменный музыкальный мотив может поддержать связь с брендом.",
                ("Фирменная мелодия подтверждена. Её стоит сохранить различимой, без конкуренции с главным предложением." if value >= .5 else
                 "Фирменная мелодия не подтверждена. Фоновая музыка сама по себе её не заменяет, "
                 "но отсутствие джингла не означает, что ролик надо переделывать."), "", "up" if value >= .5 else "balanced")
    if feature == "problem_solution_arc_present":
        return ("История связывает исходную задачу с помощью сервиса.",
                ("Есть связка «проблема - решение с помощью сервиса». Она даёт истории завершение и объясняет, "
                 "зачем появляется бренд; эту причинную линию стоит сохранить." if value >= .5 else
                 "Отдельная связка «проблема - решение» не подтверждена. Это не обязательный сюжетный приём; "
                 "если он задуман, стоит проверить, видна ли роль сервиса в решении."), "", "up" if value >= .5 else "balanced")
    return ("Важна роль этой детали в основной истории.",
            "По этой детали нет достаточного основания для отдельной рекомендации.", "", "balanced")


def _partial_fact(driver, evidence, duration, validate_alignment_record):
    """Corroborate a shared meaning without choosing a disputed rubric value."""
    feature = driver["feature"]
    if feature == "panel__message_specificity_level":
        observations = [next((o for o in r["values"]["observations"]
                              if o["feature"] == feature), None) for r in evidence]
        observations = [o for o in observations if o and o["status"] != "uncertain"
                        and isinstance(o["value"], (int, float)) and 2 <= o["value"] <= 3]
        from .latest_interpretation import has_person_name
        if len(observations) >= 3 and not any(has_person_name(o["observation"]) for o in observations):
            return dict(value=2, text="Названа конкретная задача или действие с помощью сервиса.",
                        same_input=False, evidence=dict(verified=True, agreement=1,
                            observation="Названа конкретная задача или действие с помощью сервиса.",
                            source="shared_meaning", episodes=[]))
    if feature == "fresh__audiovisual_claim_alignment" and validate_alignment_record:
        from jsonschema.exceptions import ValidationError
        rows = []
        request_ids = set()
        for row in evidence:
            item = row.get("alignment_evidence")
            if not item:
                continue
            try:
                validate_alignment_record(item, row["source_sha"], row["prepared_sha256"], row["repeat"], duration)
                if item["request_id"] in request_ids or item["request_id"] == row["request_id"]:
                    continue
                request_ids.add(item["request_id"])
            except (KeyError, TypeError, ValueError, ValidationError):
                continue
            if item["values"]["relation"] == "related":
                rows.append(item["values"])
        if len(rows) >= 3:
            action = all(r["action_evidence"]["status"] == "shown" for r in rows)
            result = all(r["result_evidence"]["status"] == "shown" for r in rows)
            if action or result:
                from .latest_interpretation import has_person_name
                phrase_key = lambda text: re.sub(r'[^а-яёa-z0-9]', '', text.lower())
                phrases = [r["main_phrase"] for r in rows]
                phrase = phrases[0] if len({phrase_key(p) for p in phrases}) == 1 else None
                if phrase and has_person_name(phrase):
                    phrase = None
                text = "Подтверждён показ результата." if result else "Подтверждён показ действия."
                if phrase:
                    text += " Основное предложение из озвучки: " + phrase + "."
                return dict(value=3 if action and result else 2, text=text,
                            same_input=False, evidence=dict(verified=True, shown_action=True if action else None,
                                shown_result=True if result else None, main_phrase=phrase,
                                agreement=1, source="shared_visual_evidence", episodes=[]))
    return None


def draft_fingerprint(draft):
    """Bind review to content, ignoring serialization-only numeric differences."""
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            number = round(value, 12)
            return int(number) if number == int(number) else number
        return value
    content = {key: value for key, value in draft.items() if key not in {"fingerprint", "reviewed"}}
    return hashlib.sha256(json.dumps(canonical(content), ensure_ascii=False,
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_manager_draft(interpretation, evidence=None, source_sha=None, duration=30,
                        validate_alignment_record=None):
    kind = interpretation.get('material_kind')
    if kind not in {'neuromatics', 'finished'}:
        return None
    prototype = kind == 'neuromatics'
    incomplete = interpretation.get("audio_status") in {"partial", "absent"}
    profiles = {}
    for task in ORDER:
        rows = []
        for driver in interpretation["details"][task]["drivers"]:
            feature = driver["feature"]
            if feature in OMIT or (prototype and feature.startswith("phys__")):
                continue
            fact = _fact(driver, incomplete)
            if not fact and not (incomplete and feature in AUDIO):
                fact = _partial_fact(driver, evidence or [], duration, validate_alignment_record)
            if fact:
                fact['duration'] = duration
            meaning, finding, check, preference = _copy(feature, fact)
            direction = _direction(driver, fact, preference)
            status = "supported" if fact else "unassessed"
            if not fact and feature == "human_characters_count":
                meaning = "Число людей само по себе не определяет ясность истории."
                finding = ("По этому нейроматику нет надёжного основания рекомендовать больше или меньше персонажей. "
                           "Решение стоит принимать по ясности их ролей, а не по числу людей в кадре.")
            if not fact and incomplete and feature in AUDIO:
                finding = "Без полной речи эту часть предложения нельзя оценить окончательно. Вернитесь к ней после добавления озвучки."
            if not prototype:
                finding = finding.replace('этому нейроматику', 'этому ролику')
            row = dict(feature=feature, features=[feature], label=driver["label"],
                       meaning=meaning, finding=finding, check=check, direction=direction,
                       status=status, observation="", importance=0, claim=None,
                       evidence_facts=({feature: fact} if fact else {}))
            # Labels should describe the topic rather than proclaim a flaw.
            from .latest_interpretation import LABELS
            row["label"] = LABELS.get(feature, row["label"])
            rows.append(row)
        profiles[task] = rows
    # Combine overlapping clarity questions; keep independently useful topics.
    clarity = {r["feature"]: r for r in profiles["m"]}
    specific = clarity.get("panel__message_specificity_level")
    inference = clarity.get("panel__mandatory_inference_chain_length")
    if specific and inference:
        specific["label"] = "Насколько понятно предложение"
        specific["features"] += inference["features"]
        specific["evidence_facts"].update(inference["evidence_facts"])
        if inference["status"] == "supported":
            specific["finding"] += " " + inference["finding"]
            specific["check"] = specific["check"] or inference["check"]
            if inference["direction"] == "down":
                specific["direction"] = "balanced"
        else:
            specific["finding"] += " При этом не подтверждено, что здесь есть лишние смысловые переходы: усложнять пояснение без проверки не нужно."
        profiles["m"] = [r for r in profiles["m"] if r is not inference]
    for task, rows in profiles.items():
        rows.sort(key=lambda r: ({"down": 0, "up": 1, "balanced": 2}[r["direction"]], r["feature"]))
        for row in rows:
            row["id"] = task + ":" + row["feature"]
    draft = dict(version=VERSION if prototype else FINISHED_VERSION, profiles=profiles, source_sha=source_sha,
                 scores={t: interpretation['details'][t].get('actual') for t in ORDER},
                 audio_status=interpretation.get("audio_status"), coverage={})
    for task, rows in profiles.items():
        features = {f for r in rows for f in r["features"]}
        supported = {f for r in rows for f in r["evidence_facts"]}
        minimum = math.ceil(len(features) / 2)
        if task == 'm':
            minimum = max(minimum, min(len(features), math.ceil(
                len(interpretation['details'][task].get('feature_order', features)) / 2)))
        draft["coverage"][task] = dict(visible=len(features), supported=len(supported), minimum=minimum)
    draft["fingerprint"] = draft_fingerprint(draft)
    return draft


def manager_profile_rows(interpretation):
    semantic = interpretation.get("manager_semantic")
    if semantic:
        return deepcopy(semantic["profiles"])
    from .latest_profile_ui import profile_rows
    return profile_rows(interpretation)


def effect_label(row):
    if row.get('status') == 'unassessed':
        return 'Не оценено окончательно'
    if row['direction'] == 'up':
        return '+ В плюс в этом ролике'
    if row['direction'] == 'down':
        return '− В минус в этом ролике'
    return 'Стоит проверить' if row.get('check') else 'Без обязательной правки'


def manager_cards(interpretation):
    from .manager_report import report_cards
    if not interpretation.get('manager_semantic'):
        return report_cards(interpretation)
    profiles = manager_profile_rows(interpretation)
    cards = []
    for card in interpretation['cards']:
        task = card['task']
        value = {k: deepcopy(v) for k, v in card.items()
                 if k not in {'strengths', 'limitations', 'unresolved', 'unresolved_drivers'}}
        value.update(strengths=[r for r in profiles[task] if r['direction'] == 'up'],
                     limitations=[r for r in profiles[task] if r['direction'] == 'down'],
                     context=[r for r in profiles[task] if r['direction'] == 'balanced' and r['status'] == 'supported'],
                     unassessed=[r for r in profiles[task] if r['status'] == 'unassessed'])
        cards.append(value)
    return cards
