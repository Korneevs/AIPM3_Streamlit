"""Manager-facing copy built only from the existing evidence and attributions.

No inference, score changes or invented creative recommendations here.
"""
from __future__ import annotations

import re

from .latest_interpretation import LABELS


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
    text = text.replace("Запоминающегося брендового джингла нет", "Фирменной мелодии нет")
    text = re.sub(r"(Финальный|финальный) кадр с брендом с логотипом", r"\1 кадр с логотипом", text)
    if text.startswith("Посыл конкретен:"):
        text = re.sub(r"(объект|действие|результат) \(([^)]+)\)", r"\2", text, flags=re.I)
        text = text.replace("Посыл конкретен: назван ", "В предложении названы ")
    text = text.replace("Финальный финальный", "Финальный").replace("финальный финальный", "финальный")
    return re.sub(r"\s+", " ", text).strip()


def report_cards(interpretation):
    """Expose more grounded features without relaxing the evidence gates."""
    cards = []
    for card in interpretation["cards"]:
        task = card.get("task")
        drivers = interpretation.get("details", {}).get(task, {}).get("drivers")
        if drivers is None:
            drivers = [*card.get("strengths", []), *card.get("limitations", [])]
        groups = {"strengths": [], "limitations": [], "observations": []}
        pending = []
        for driver in drivers:
            feature = driver.get("feature", "")
            if feature == "brand_history" or feature.startswith("phys__") or feature == "is_celeb":
                continue
            support = driver.get("evidence", {})
            factual = driver.get("factual_observation")
            independent_only = not support.get("verified", False) and bool(factual)
            if independent_only:
                support = factual["evidence"]
            if not support.get("verified", False):
                if abs(driver.get("index_points", 0)) >= 2:
                    pending.append(LABELS.get(feature, driver["label"]))
                continue
            if feature not in PURPOSE:
                continue
            value = factual["value"] if independent_only else driver.get("value", 0)
            absent = feature in ABSENCE and value < (.5 if feature not in {
                "character_close_up_seconds", "product_demo_seconds"} else 1e-9)
            context = feature_context(feature, value, absent)
            phrase = support.get("main_phrase")
            if feature == "fresh__audiovisual_claim_alignment" and phrase and re.search(r"[а-яА-ЯёЁ]", phrase):
                quote = plain_text(phrase).strip('«»"').rstrip('.')
                context = "Здесь проверяем связь изображения с предложением: «" + quote + "»" + (
                    "" if quote.endswith(('?', '!')) else ".")
            item = dict(feature=feature, label=factual["label"] if independent_only else driver["label"],
                        observation=observation_text(feature, support, value),
                        context=context, episodes=support.get("episodes", [])[:2],
                        check=plain_text(driver.get("check")))
            if feature == "state_transformation_present" and absent:
                item["label"] = "Изменение ситуации благодаря сервису"
            # An odd association or a changing sign is a factual observation,
            # never a reason to add/remove a device or a list of shortcomings.
            directional = (not independent_only and driver.get("usable")
                           and driver.get("interpretation_kind") == "observed_driver")
            if feature == "panel__distinct_benefit_count" and value < .5:
                directional = False
            # A relative arithmetic contribution is not a creative defect.
            # Neither absent optional devices nor a specific offer should be
            # presented as a shortcoming merely because their contribution is negative.
            if driver.get("direction") != "supports":
                if absent or feature in {"character_close_up_seconds", "pack_shot_duration_seconds",
                                         "product_demo_seconds", "panel__distinct_benefit_count"}:
                    directional = False
                if feature == "panel__message_specificity_level" and value >= 1.5:
                    directional = False
            group = ("strengths" if driver.get("direction") == "supports" else "limitations") if directional else "observations"
            if group != "limitations":
                item["check"] = ""
            groups[group].append(item)
        cards.append(dict(card, description=DESCRIPTIONS.get(task, ""), **groups,
                          observations_title=("Ещё о подаче ролика" if groups['strengths'] or groups['limitations']
                                              else "Что видно в ролике"),
                          pending=list(dict.fromkeys(pending))))
    return cards
