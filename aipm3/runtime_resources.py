"""Bound peak memory on Community Cloud without changing scoring or prompts."""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid


_ANALYSIS_LOCK = threading.Lock()
_MEDIA_LOCK = threading.Lock()
_QUEUE_CONDITION = threading.Condition()
_ANALYSIS_QUEUE = deque()
_REQUEST_DEADLINE = ContextVar("aipm_request_deadline", default=None)


class AnalysisBusy(RuntimeError):
    pass


class AnalysisTimeout(RuntimeError):
    pass


@contextmanager
def analysis_slot(*, wait=False, progress=None, on_queue=None, timeout=None):
    """Keep one heavy job active; optionally wait in a cancellable FIFO queue."""
    ticket = object()
    acquired = False
    started = time.monotonic()
    if not wait:
        with _QUEUE_CONDITION:
            acquired = not _ANALYSIS_QUEUE and _ANALYSIS_LOCK.acquire(blocking=False)
        if not acquired:
            raise AnalysisBusy("Сервер уже анализирует другой ролик. Дождитесь завершения и повторите запуск.")
    else:
        with _QUEUE_CONDITION:
            _ANALYSIS_QUEUE.append(ticket)
    try:
        if wait:
            last_notice = None
            while not acquired:
                with _QUEUE_CONDITION:
                    if _ANALYSIS_QUEUE[0] is ticket and _ANALYSIS_LOCK.acquire(blocking=False):
                        _ANALYSIS_QUEUE.popleft()
                        acquired = True
                        break
                    position = _ANALYSIS_QUEUE.index(ticket) + 1
                elapsed = time.monotonic() - started
                if timeout is not None and elapsed >= timeout:
                    raise AnalysisBusy("Не удалось дождаться запуска анализа. Повторите запуск позже.")
                notice = (position, int(elapsed // 5))
                if notice != last_notice:
                    if progress:
                        progress(f"Сервер занят. Ваш номер в очереди: {position}. Анализ начнётся автоматически.")
                    if on_queue:
                        on_queue(position)
                    last_notice = notice
                with _QUEUE_CONDITION:
                    _QUEUE_CONDITION.wait(timeout=1.0 if timeout is None else max(0., min(1.0, timeout - elapsed)))
            if on_queue:
                on_queue(0)
        yield
    finally:
        with _QUEUE_CONDITION:
            if acquired:
                _ANALYSIS_LOCK.release()
            elif ticket in _ANALYSIS_QUEUE:
                _ANALYSIS_QUEUE.remove(ticket)
            _QUEUE_CONDITION.notify_all()


@contextmanager
def analysis_deadline(seconds=1800):
    """Propagate the job budget to requests, including copied worker contexts."""
    token = _REQUEST_DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def request_timeout(max_seconds=300):
    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        return max_seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AnalysisTimeout("Анализ занял слишком много времени. Готовые этапы сохранены; повторите запуск.")
    return min(max_seconds, 120, remaining)


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
