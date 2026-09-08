"""Real media smoke test with no FFmpeg on PATH; no Gemini requests."""
import hashlib
from pathlib import Path
import subprocess

import imageio_ffmpeg
import pytest

from aipm3 import message_delivery_runtime as md
from aipm3 import objective_features as objective


def test_cloud_does_not_require_apt():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "packages.txt").exists()
    assert "imageio-ffmpeg==0.6.0" in (root / "requirements.txt").read_text().splitlines()


def test_all_media_operations_without_system_ffmpeg(tmp_path, monkeypatch):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Only the explicit wheel binary remains available; no host ffmpeg/ffprobe.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(md.shutil, "which", lambda _: None)
    assert md.ffmpeg_executable() == ffmpeg
    source = tmp_path / "fixture.mov"
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(source),
    ], check=True, capture_output=True, timeout=30)
    original_hash = hashlib.sha256(source.read_bytes()).digest()
    assert md.video_duration(source) == pytest.approx(4, abs=0.15)

    monkeypatch.setattr(objective, "AIPM1_MAX_BINARY_MB", 0)
    a1 = objective.prepare_legacy_video(source, tmp_path / "a1", "aipm1")
    a2 = objective.prepare_legacy_video(source, tmp_path / "a2", "aipm2")
    prepared = md.prepare_video(source, tmp_path / "md")
    for video in [a1, a2, prepared]:
        assert video != source and video.stat().st_size > 0
        assert md.video_duration(video) == pytest.approx(4, abs=0.15)

    outputs = [a1, a2, prepared]
    for mask in [1, 2]:
        for fraction in [0.25, 0.5, 0.75]:
            output = tmp_path / f"mask-{mask}-{fraction}.mp4"
            md.make_nested_clip(prepared, output, fraction, mask, "smoke-test")
            assert md.video_duration(output) == pytest.approx(4 * fraction, abs=0.25)
            outputs.append(output)
    for output in outputs:
        decoded = subprocess.run(
            [ffmpeg, "-i", str(output), "-f", "null", "-"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        assert "Video:" in decoded.stderr and "Audio:" in decoded.stderr
    assert hashlib.sha256(source.read_bytes()).digest() == original_hash
