"""Deterministic, feature-grounded creative observations; never SHAP or advice."""
from __future__ import annotations

from collections import defaultdict
import math
import re

from .creative_review import VERSION as SCENE_VERSION, fingerprint

VERSION = "feature-diagnosis-v1"
CONSENSUS = 0.8


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _text(value):
    return " ".join(re.findall(r"[а-яa-z0-9]+", str(value).lower().replace("ё", "е")))


def panel_signal(result, feature, predicate):
    """Require agreement both among answers and among separate calls, not fake N=30 independence."""
    by_call = defaultdict(list)
    for row in result.get("diagnostic_panel", []):
        value = _number(row.get(feature))
        if value is not None and row.get("call_id") is not None:
            by_call[str(row["call_id"])].append(bool(predicate(value)))
    answers = [v for group in by_call.values() for v in group]
    calls = [sum(group) / len(group) >= 2 / 3 for group in by_call.values()]
    stable = (len(answers) >= 24 and len(calls) >= 8 and
              sum(answers) / len(answers) >= CONSENSUS and sum(calls) / len(calls) >= CONSENSUS)
    return {"stable": stable, "supports": sum(answers), "n": len(answers),
            "calls": sum(calls), "total_calls": len(calls), "feature": feature}


def _legacy_signal(result, component, feature, predicate):
    values = [_number(r.get(feature)) for r in result.get("objective_runs", {}).get(component, [])]
    values = [v for v in values if v is not None]
    required = 3 if component == "aipm1" else 2
    return len(values) >= required and sum(bool(predicate(v)) for v in values) / len(values) >= CONSENSUS


def scene_example(result, groups):
    evidence = result.get("diagnostics", {}).get("scene_evidence", {})
    if (not result.get("source_sha") or evidence.get("source_sha") != result["source_sha"]
            or evidence.get("version") != SCENE_VERSION):
        return None
    transcripts = [_text(row.get("transcript", "")) for row in result.get("transcripts", [])]
    idea_words = {w[:5] for w in _text(result.get('main_idea', '')).split()
                  if len(w) > 3 and w not in {'авито', 'можно', 'нужно'}}
    candidates = []
    for scene in evidence.get("scenes", []):
        quote = _text(scene.get("quote", ""))
        overlap = len(set(scene.get("groups", [])) & set(groups))
        # Do not turn an unverified standalone scene description into a fact.
        start, end = _number(scene.get("start_seconds")), _number(scene.get("end_seconds"))
        duration = _number(result.get("duration_seconds"))
        valid_time = start is not None and end is not None and duration and 0 <= start < end <= duration
        if valid_time and overlap and quote and len(transcripts) >= 2 and sum(quote in t for t in transcripts) >= 2:
            relevance = len(idea_words & {w[:5] for w in quote.split() if len(w) > 3})
            candidates.append((overlap, relevance, scene))
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: (-s[0], -s[1], s[2]["start_seconds"], s[2]["quote"]))[0][2]


def transcript_example(result):
    """Literal contextual quote when no optional scene review has been requested."""
    texts = [str(r.get('transcript', '')) for r in result.get('transcripts', [])]
    normalized = [_text(t) for t in texts]
    idea = {w[:5] for w in _text(result.get('main_idea', '')).split() if len(w) > 3 and w not in {'авито', 'можно', 'нужно'}}
    candidates = []
    for text in texts:
        for sentence in re.findall(r'[^.!?…]+[.!?…]?', text):
            sentence = sentence.strip()
            norm = _text(sentence)
            overlap = len(idea & {w[:5] for w in norm.split() if len(w) > 3})
            if 15 <= len(sentence) <= 300 and overlap and sum(norm in t for t in normalized) >= 2:
                candidates.append((overlap, sentence))
    return sorted(candidates, key=lambda c: (-c[0], -len(c[1]), c[1]))[0][1] if candidates else ''


def build_diagnosis(result):
    findings = []
    unstable = []

    def add(key, side, title, meaning, evidence, groups=(), signal=None, priority=50):
        findings.append({"id": key, "side": side, "title": title, "meaning": meaning,
                         "evidence": evidence, "signal": signal, "scene": scene_example(result, groups),
                         "priority": priority})

    def panel_rule(key, feature, predicate, side, title, meaning, groups, priority=50):
        signal = panel_signal(result, feature, predicate)
        if signal["stable"]:
            evidence = (f"Подтверждено в {signal['supports']} из {signal['n']} оценок; "
                        f"в {signal['calls']} из {signal['total_calls']} отдельных запросов.")
            add(key, side, title, meaning, evidence, groups, signal, priority)
        return signal

    direct = panel_rule("direct", "main_claim_linguistic_directness", lambda v: v == 3, "strength",
               "Предложение сформулировано прямо",
               "Коммерческий смысл назван словами, а не оставлен только в шутке или образе. "
               "Это опора для понимания предложения при полном просмотре.",
               ["message_focus", "completeness"], 100)
    panel_rule("outcome", "demonstrated_user_outcome_count", lambda v: v >= 1, "strength",
               "Показано, что получает пользователь",
               "В истории есть наблюдаемый результат использования сервиса. "
               "Обещание получает сюжетное воплощение, а не остаётся только слоганом.",
               ["offer", "attention", "completeness"], 70)
    panel_rule("recipient", "benefit_recipient_clarity", lambda v: v == 3, "strength",
               "Понятно, кому адресовано предложение",
               "Выгода привязана к узнаваемой ситуации пользователя. "
               "Не требуется отдельно угадывать, кому и зачем нужен сервис.", ["offer", "message_focus"], 60)
    audio = panel_signal(result, "audio_only_message_completeness", lambda v: v == 3)
    visual = panel_signal(result, "visual_only_message_completeness", lambda v: v < 3)
    if audio["stable"] and visual["stable"]:
        add("sound_dependency", "risk", "Без звука предложение передаётся неполно",
            "Реплики несут часть смысла, которую изображение и экранный текст не восполняют полностью. "
            "Это ограничение контакта без звука, а не претензия к самому наличию диалога.",
            f"Полнота по звуку: полная в {audio['supports']}/{audio['n']} оценках. "
            f"Без звука: неполная в {visual['supports']}/{visual['n']}.",
            ["speech", "completeness", "message_focus"], visual, 95)
    if audio["stable"] and not direct["stable"]:
        add("audio_full", "strength", "По репликам можно восстановить предложение",
            "Звуковая дорожка содержит достаточно информации для пересказа основного сообщения. "
            "Она не требует непрерывного внимания к изображению.",
            f"Полная передача по звуку: {audio['supports']}/{audio['n']} оценок, "
            f"{audio['calls']}/{audio['total_calls']} запросов.",
            ["speech", "completeness", "message_focus"], audio, 90)
    panel_rule("late", "first_core_claim_time_band", lambda v: 2 <= v <= 3, "risk",
               "Предложение раскрывается только после десятой секунды",
               "Короткий контакт может закончиться до явного раскрытия коммерческого смысла. "
               "Сама по себе понятность полного ролика не снимает этот риск.", ["message_focus", "completeness"], 110)
    panel_rule("no_claim", "first_core_claim_time_band", lambda v: v == 4, "risk",
               "Нет момента, в котором предложение становится явным",
               "Сюжет можно пересказать, но коммерческий посыл приходится достраивать из контекста.",
               ["message_focus"], 120)
    panel_rule("inference", "mandatory_inference_chain_length", lambda v: v >= 2, "risk",
               "Между сюжетом и предложением нужно достроить несколько связей",
               "Для понимания рекламы недостаточно назвать происходящее: нужны как минимум два смысловых перехода. "
               "Это делает пересказ предложения менее прямым.", ["attention", "message_focus"], 90)
    panel_rule("competing", "secondary_claim_count", lambda v: v >= 2, "risk",
               "С главным посылом конкурируют другие тезисы",
               "Кроме основной мысли в ролике есть как минимум два самостоятельных рекламных тезиса. "
               "В открытом пересказе они могут занять её место.", ["offer", "message_focus"], 85)
    panel_rule("conditions", "offer_condition_count", lambda v: v >= 2, "risk",
               "Предложение нельзя пересказать без нескольких условий",
               "Чтобы не исказить обещание, нужно удержать как минимум два ограничения или условия. "
               "Упрощённый пересказ рискует потерять важную часть предложения.", ["offer", "completeness"], 80)
    cta_no = panel_rule("cta", "cta_clarity", lambda v: v == 0, "risk",
                        "Ясность следующего действия под вопросом",
                        "Панель не выделяет однозначного следующего шага. Финальная реплика может называть сервис "
                        "и покупку, но в разметке этого недостаточно для ясного действия. Это спорное место коммуникации, "
                        "а не утверждение, что призыва или финального экрана совсем нет.", ["cta"], 75)
    cta_yes = panel_signal(result, "cta_clarity", lambda v: v == 1)
    if cta_no["n"] and not cta_no["stable"] and not cta_yes["stable"]:
        unstable.append(f"Ясность следующего действия: положительных оценок {cta_yes['supports']}/{cta_yes['n']}. "
                        "Оценки расходятся — однозначный плюс или минус не выставлен.")
    a2 = result.get("objective_features", {}).get("aipm2", {})
    if (_legacy_signal(result, "aipm1", "main_character", lambda v: v == 1)
            and _legacy_signal(result, "aipm1", "state_transformation", lambda v: v == 1)):
        add("hero_change", "strength", "Есть герой, чья ситуация меняется",
            "Ролик можно воспринимать как историю одного пользователя и результата, а не как набор разрозненных сцен. "
            "Это понятная сюжетная опора для предложения сервиса.",
            "Главный герой и изменение его состояния отмечены во всех трёх разметках AIPM 1.0.",
            ["attention"], priority=95)
    if (_legacy_signal(result, "aipm1", "humor", lambda v: v == 1)
            and _legacy_signal(result, "aipm2", "has_callback_to_opening", lambda v: v == 1)):
        add("humor_callback", "strength", "Юмор и возвращение к началу задают форму истории",
            "Юмористическая подача сопровождается возвращением к началу истории. "
            "Это узнаваемый приём ролика; сам по себе он не доказывает, что зритель запомнит предложение.",
            "Юмор: 3 из 3 разметок AIPM 1.0; возвращение к началу: 2 из 2 разметок AIPM 2.0.",
            ["emotion"], priority=85)
    duration = _number(result.get("duration_seconds"))
    logo = _number(a2.get("brand_logo_screen_seconds"))
    if duration and logo is not None and _legacy_signal(result, "aipm2", "brand_logo_screen_seconds", lambda v: v / duration >= .8):
        add("branding", "strength", "Бренд присутствует почти весь ролик",
            "История визуально связана с сервисом на протяжении контакта, а не только в последнем кадре. "
            "Это экранная привязка к бренду, не гарантия запоминания.",
            f"Логотип: около {logo:g} из {duration:.1f} сек.; обе разметки подтверждают не менее 80% длительности.",
            ["brand"], priority=80)
    curve = result.get("recovery_curve", {})
    needed = ["full", "nested_50_m1", "nested_50_m2"]
    if all(_number(curve.get(k)) is not None for k in needed):
        full, half1, half2 = [float(curve[k]) for k in needed]
        if full >= .7 and full - max(half1, half2) >= .2:
            add("partial", "risk", "На половине ролика содержательный пересказ заметно теряется",
                "В обоих наборах фрагментов содержательных ответов меньше, чем после полного просмотра. "
                "Это зависимость от полноты контакта, а не проверка именно заданного UVP.",
                f"Сглаженная доля содержательных ответов: полный ролик {full:.0%}; половины {half1:.0%} и {half2:.0%}.",
                priority=105)
        elif abs(half1 - half2) >= .25:
            add("fragment_choice", "risk", "Результат зависит от того, какая половина ролика показана",
                "Разные фрагменты одного ролика дают заметно разную содержательность пересказов. "
                "Сам объём просмотра здесь не описывает качество контакта.",
                f"Половины ролика: {half1:.0%} и {half2:.0%} содержательных ответов после сглаживания.", priority=100)
        # A flat low curve is not called a strength.
        if full <= .6:
            add("full_recovery", "risk", "Даже полный просмотр не всегда даёт содержательный пересказ",
                "После полного просмотра часть синтетических респондентов не формулирует самостоятельную идею. "
                "Наличие понятной ключевой реплики и её попадание в каждый пересказ — разные вещи.",
                f"Сглаженная доля содержательных ответов полного просмотра: {full:.0%}. "
                "Это показатель синтетической панели, не процент людей.", priority=115)
    selected = {side: sorted([f for f in findings if f["side"] == side], key=lambda f: (-f["priority"], f["id"]))[:4]
                for side in ["strength", "risk"]}
    for finding in selected['strength'] + selected['risk']:
        if not finding['scene'] and finding['id'] in {'direct', 'audio_full', 'sound_dependency'}:
            finding['transcript_quote'] = transcript_example(result)
    for finding in selected['risk']:
        if finding['id'] == 'full_recovery':
            answers = [r for r in result.get('diagnostic_recovery', [])
                       if r.get('condition_group') == 'full' and r.get('answer_type') != 'valid' and r.get('raw_answer')]
            if answers:
                finding['answer_example'] = sorted(answers, key=lambda r: str(r.get('respondent_uid')))[0]['raw_answer']
    idea = str(result.get("main_idea") or "").strip()
    headline = f"Самая частая идея в свободных ответах: «{idea}»." if idea else "Основная идея не определена однозначно."
    if selected["risk"]:
        headline += " На что обратить внимание: " + selected["risk"][0]["title"].lower() + "."
    return {"version": VERSION, "headline": headline, "strengths": selected["strength"],
            "risks": selected["risk"], "uncertain": unstable,
            "has_panel": bool(result.get("diagnostic_panel")),
            "input_hash": fingerprint({"features": {k: result.get(k) for k in [
                "source_sha", "objective_features", "objective_runs", "diagnostic_panel", "recovery_curve",
                "main_idea", "duration_seconds", "transcripts", "diagnostic_recovery"]},
                "scene_evidence": result.get("diagnostics", {}).get("scene_evidence")})}
