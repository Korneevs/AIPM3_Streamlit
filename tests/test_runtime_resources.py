"""Bound concurrent work, preserve complete panels and discard failed media."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from aipm3 import message_delivery_runtime as md
from aipm3 import pipeline, runtime_resources as resources


def test_another_analysis_cannot_overlap_and_lock_recovers_after_failure():
    with pytest.raises(ValueError):
        with resources.analysis_slot():
            with ThreadPoolExecutor(max_workers=1) as pool:
                def other():
                    with resources.analysis_slot():
                        pytest.fail("Two analyses share the same memory budget")
                with pytest.raises(resources.AnalysisBusy):
                    pool.submit(other).result()
            raise ValueError("failed stage")
    with resources.analysis_slot():
        pass


def test_file_hash_streams_instead_of_loading_whole_upload(tmp_path, monkeypatch):
    data = b"input video" * 200_000
    path = tmp_path / "source.mp4"
    path.write_bytes(data)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("Unbounded whole-file read"))
    assert resources.file_sha256(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("failure", ["returncode", "timeout", "empty"])
def test_failed_encode_never_publishes_partial_media(tmp_path, monkeypatch, failure):
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"old complete clip")
    def run(command, **kwargs):
        if failure != "empty":
            Path(command[-1]).write_bytes(b"unfinished clip")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=int(failure == "returncode"), stderr="encoder failed")
    monkeypatch.setattr(resources.subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        resources.run_video_command(["ffmpeg", "-i", "source.mp4", str(output)])
    assert output.read_bytes() == b"old complete clip"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]


def test_encodes_are_serial_and_single_threaded(tmp_path, monkeypatch):
    counts = {"active": 0, "peak": 0}
    lock = threading.Lock()
    def run(command, **kwargs):
        assert command[command.index("-threads") + 1] == "1"
        assert command[-3:-1] == ["-threads", "1"]
        assert command[command.index("-filter_complex_threads") + 1] == "1"
        assert kwargs["timeout"] == 300
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        time.sleep(.02)
        Path(command[-1]).write_bytes(b"complete")
        with lock:
            counts["active"] -= 1
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(resources.subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(resources.run_video_command, ["ffmpeg", "-i", "source.mp4", str(tmp_path / f"{i}.mp4")])
                   for i in range(3)]
        for f in futures:
            f.result()
    assert counts["peak"] == 1
    assert len(list(tmp_path.glob("*.mp4"))) == 3


def test_message_delivery_stages_run_in_order_without_overlap(tmp_path, monkeypatch):
    caller = threading.get_ident()
    events = []
    def run(name, values):
        def stage(*args):
            assert threading.get_ident() == caller
            events.append(name)
            return pd.DataFrame(), values
        return stage
    monkeypatch.setattr(md, "extract_panel30", run("panel", {"panel": 30}))
    monkeypatch.setattr(md, "extract_transcript", run("transcript", 2.0))
    monkeypatch.setattr(md, "extract_recovery", run("recovery", {"recovery": 84}))
    observed = []
    result = pipeline._message_delivery_extract(tmp_path / "video.mp4", "encoded", "sha", 60, tmp_path, "test", observed.append)
    assert events == ["panel", "transcript", "recovery"]
    assert result["panel"] == {"panel": 30} and result["recovery"] == {"recovery": 84}
    assert len(observed) == 3


def test_recovery_keeps_all_six_masks_and_84_answers_with_two_active_payloads(tmp_path, monkeypatch):
    video = tmp_path / "full.mp4"
    video.write_bytes(b"full video")
    generated, calls, encoded_in_workers = [], [], []
    owner = threading.get_ident()
    def clip(source, output, fraction, mask, seed):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"{fraction}-{mask}".encode())
        generated.append((fraction, mask, seed))
        return output
    def encode(path):
        assert threading.get_ident() != owner, "Do not pre-encode all variants"
        encoded_in_workers.append(path.name)
        return "fragment", resources.file_sha256(path)
    counts = {"active": 0, "peak": 0}
    lock = threading.Lock()
    def request(**kwargs):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        time.sleep(.005)
        task = kwargs["task_key"]
        group = int(task.rsplit("_g", 1)[1])
        calls.append(task)
        people = md.RECOVERY_PERSONAS[(group - 1) * 4:group * 4]
        with lock:
            counts["active"] -= 1
        return {"answers": [{"respondent_id": key, "answer": "test idea", "brand_only": False, "no_idea": False}
                            for key, _ in people]}
    monkeypatch.setattr(md, "make_nested_clip", clip)
    monkeypatch.setattr(md, "encode_video", encode)
    monkeypatch.setattr(md, "call_json", request)
    monkeypatch.setattr(md, "cluster_recovery", lambda rows, *args: [dict(row, canonical_idea="test idea", answer_type="valid") for row in rows])
    frame, values = md.extract_recovery(video, "full", "seed", tmp_path / "out", tmp_path / "unused", False)
    assert len(generated) == 6 and all(seed == "seed" for _, _, seed in generated)
    assert len(calls) == 21 and len(set(calls)) == 21
    assert len(encoded_in_workers) == 18 and counts["peak"] == 2
    assert len(frame) == 84 and set(frame.condition_group) == set(md.RECOVERY_CONDITIONS)
    assert frame.groupby("condition_group").size().eq(12).all()
    assert values["p12__cluster_valid_mask_mae_smoothed"] == 0
