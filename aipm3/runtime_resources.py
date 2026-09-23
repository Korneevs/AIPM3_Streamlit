"""Bound peak memory on Community Cloud without changing scoring or prompts."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid


_ANALYSIS_LOCK = threading.Lock()
_MEDIA_LOCK = threading.Lock()


class AnalysisBusy(RuntimeError):
    pass


@contextmanager
def analysis_slot():
    if not _ANALYSIS_LOCK.acquire(blocking=False):
        raise AnalysisBusy("Сервер уже анализирует другой ролик. Дождитесь завершения и повторите запуск.")
    try:
        yield
    finally:
        _ANALYSIS_LOCK.release()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_stage(stage: str) -> None:
    record = {"event": "analysis_stage", "stage": stage}
    # cgroup includes FFmpeg children, unlike Python's own resident memory.
    for key, candidates in {
        "memory_bytes": ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"],
        "memory_limit_bytes": ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"],
    }.items():
        for candidate in candidates:
            try:
                record[key] = int(Path(candidate).read_text().strip())
                break
            except (OSError, ValueError):
                pass
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run_video_command(command: list[str], timeout: int = 300) -> None:
    """One single-threaded encode at a time; publish only a complete output."""
    output = Path(command[-1])
    temporary = output.with_name(f".{output.stem}-{uuid.uuid4().hex}.partial{output.suffix}")
    bounded = [command[0], "-threads", "1", "-filter_threads", "1",
               "-filter_complex_threads", "1", *command[1:-1], "-threads", "1", str(temporary)]
    try:
        with _MEDIA_LOCK:
            result = subprocess.run(bounded, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"Не удалось подготовить видео: {result.stderr[-1600:]}")
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("Обработка видео завершилась без готового файла.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
