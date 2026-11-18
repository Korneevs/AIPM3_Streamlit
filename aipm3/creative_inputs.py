"""Bounded, lossless text imports for the creative scenario selector."""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree


MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_SCENARIOS = 60
MAX_SCENARIO_CHARS = 12000
MAX_TOTAL_CHARS = 180000
MAX_BRIEF_CHARS = 15000
_HEADING = re.compile(r"^\s*===\s*(.+?)\s*===\s*$", re.MULTILINE)
_ID = re.compile(r"^[\w.-]{1,64}$", re.UNICODE)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class CreativeInputError(ValueError):
    """A short input issue that can be shown to the user."""


def _text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CreativeInputError("Сохраните текст или CSV в кодировке UTF-8.") from exc


def parse_pasted(text: str, *, default_title: str = "Сценарий 1") -> list[dict]:
    """Split only on explicit standalone === Title === headings."""
    if not text.strip():
        return []
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [{"title": default_title, "text": text.strip()}]
    if text[:matches[0].start()].strip():
        raise CreativeInputError("Перед первым разделителем «=== Название ===» есть текст. Перенесите его в сценарий или бриф.")
    rows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if not body:
            raise CreativeInputError(f"У сценария «{match.group(1).strip()[:100]}» нет текста.")
        rows.append({"title": match.group(1).strip(), "text": body})
    return rows


def _docx(data: bytes) -> str:
    try:
        with ZipFile(BytesIO(data)) as archive:
            if len(archive.infolist()) > 5000:
                raise CreativeInputError("DOCX содержит слишком много вложений. Загрузите только текст сценария.")
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCUMENT_BYTES:
                raise CreativeInputError("Текст DOCX слишком большой. Разделите его на несколько файлов.")
            raw = archive.read(info)
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise CreativeInputError("Не удалось прочитать структуру DOCX. Сохраните сценарий как TXT.")
        document = ElementTree.fromstring(raw)
        paragraphs = []
        for paragraph in document.iter(_W + "p"):
            parts = []
            for node in paragraph.iter():
                if node.tag == _W + "t":
                    parts.append(node.text or "")
                elif node.tag == _W + "tab":
                    parts.append("\t")
                elif node.tag in {_W + "br", _W + "cr"}:
                    parts.append("\n")
            paragraphs.append("".join(parts))
        return "\n".join(paragraphs).strip()
    except (BadZipFile, KeyError, ElementTree.ParseError, RuntimeError) as exc:
        raise CreativeInputError("Не удалось прочитать DOCX. Сохраните документ заново или загрузите TXT.") from exc


def _csv(data: bytes) -> list[dict]:
    text = _text(data)
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(StringIO(text, newline=""), dialect=dialect, strict=True)
    aliases = {"id": "id", "title": "title", "text": "text", "название": "title", "сценарий": "text", "текст": "text"}
    try:
        headers = reader.fieldnames or []
    except csv.Error as exc:
        raise CreativeInputError("Не удалось прочитать CSV. Проверьте разделители и кавычки.") from exc
    normalized_headers = [name.strip().lower() for name in headers]
    recognized_headers = [aliases[name] for name in normalized_headers if name in aliases]
    if len(set(normalized_headers)) != len(normalized_headers) or len(set(recognized_headers)) != len(recognized_headers):
        raise CreativeInputError("В CSV повторяются столбцы названия, текста или ID.")
    names = {name: aliases.get(name.strip().lower()) for name in headers}
    if "title" not in names.values() or "text" not in names.values():
        raise CreativeInputError("В CSV нужны столбцы title и text (или «Название» и «Текст»). Столбец id необязателен.")
    rows = []
    try:
        for number, row in enumerate(reader, 2):
            if None in row:
                raise CreativeInputError(f"Строка CSV {number}: лишние значения. Проверьте разделитель и кавычки.")
            known = {names[key]: (value or "").strip() for key, value in row.items() if names.get(key)}
            if not any(known.values()):
                continue
            if not known.get("title") or not known.get("text"):
                raise CreativeInputError(f"Строка CSV {number}: нужны название и полный текст сценария.")
            rows.append(known)
    except csv.Error as exc:
        raise CreativeInputError("Не удалось прочитать CSV. Проверьте разделители и кавычки.") from exc
    return rows


def parse_upload(name: str, data: bytes) -> list[dict]:
    if len(data) > MAX_FILE_BYTES:
        raise CreativeInputError(f"Файл «{Path(name).name[:100]}» больше 4 МБ. Загрузите только текст.")
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        rows = _csv(data)
    elif suffix in {".txt", ".md", ".docx"}:
        text = _docx(data) if suffix == ".docx" else _text(data)
        rows = parse_pasted(text, default_title=Path(name).stem)
    else:
        raise CreativeInputError("Поддерживаются TXT, MD, DOCX и CSV.")
    if not rows:
        raise CreativeInputError(f"В файле «{Path(name).name[:100]}» нет сценариев.")
    return rows


def collect_scenarios(uploads: list[tuple[str, bytes]], pasted: str) -> list[dict]:
    rows = []
    for name, data in uploads:
        rows.extend(parse_upload(name, data))
    rows.extend(parse_pasted(pasted))
    if len(rows) > MAX_SCENARIOS:
        raise CreativeInputError(f"В одном запуске можно сравнить до {MAX_SCENARIOS} сценариев. Сейчас их {len(rows)}.")
    explicit = [row["id"] for row in rows if row.get("id")]
    if any(not _ID.fullmatch(item) for item in explicit):
        raise CreativeInputError("ID в CSV: до 64 букв, цифр и знаков _ . - без пробелов.")
    if len(set(explicit)) != len(explicit):
        raise CreativeInputError("В CSV повторяются ID сценариев. Сделайте их уникальными.")
    ids = set(explicit)
    total = 0
    for index, row in enumerate(rows, 1):
        if len(row["title"]) > 200:
            raise CreativeInputError(f"Название сценария {index} длиннее 200 символов. Сократите только название.")
        if len(row["text"]) > MAX_SCENARIO_CHARS:
            raise CreativeInputError(f"Сценарий «{row['title']}» длиннее {MAX_SCENARIO_CHARS:,} символов. Сократите его перед загрузкой.")
        total += len(row["text"]) + len(row["title"])
        if not row.get("id"):
            generated = f"s{index:03d}"
            while generated in ids:
                generated = "s" + generated
            row["id"] = generated
            ids.add(generated)
    if total > MAX_TOTAL_CHARS:
        raise CreativeInputError(f"Общий объём больше {MAX_TOTAL_CHARS:,} символов. Разделите сценарии на несколько запусков.")
    return [{"id": row["id"], "title": row["title"], "text": row["text"]} for row in rows]


def safe_csv_cell(value) -> str:
    """Prevent a downloaded text cell from becoming an Excel formula."""
    text = str(value or "")
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text
