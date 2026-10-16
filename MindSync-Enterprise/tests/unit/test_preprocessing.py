"""Step-2 preprocessing tests (no webcam needed). Run: pytest tests/unit -v

Covers: valid video, corrupt/unreadable, no frames, sampling at
different FPS, invalid metadata fallback, short clip, resource cleanup,
cancellation.
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("numpy")

from ai_services.video_facial_emotion_behavior.preprocessing.frames import (
    _compute_step, _effective_fps, iter_sampled_frames, preprocess_video)


def _make_preprocess_video(path, *, duration_s=4.0, fps=10, w=160, h=120):
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    assert out.isOpened()
    for i in range(int(duration_s * fps)):
        out.write(np.full((h, w, 3), 200 - i % 50, dtype=np.uint8))
    out.release()


def test_valid_video_ok(tmp_path):
    p = str(tmp_path / "ok.mp4")
    _make_preprocess_video(p, duration_s=4.0, fps=10)
    r = preprocess_video(p, sample_fps=2.0, max_frames=120, min_valid_frames=2)
    assert r.ok, r.message()
    assert r.sampled_count >= 2 and r.width == 160
    r.frames.clear()


def test_corrupt_file_fails(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"not a video" * 100)
    r = preprocess_video(str(p))
    assert not r.ok


def test_missing_file_fails(tmp_path):
    r = preprocess_video(str(tmp_path / "nope.mp4"))
    assert not r.ok and "not found" in r.message().lower()


def test_short_clip_insufficient(tmp_path):
    p = str(tmp_path / "tiny.mp4")
    _make_preprocess_video(p, duration_s=0.5, fps=10)
    r = preprocess_video(p, sample_fps=2.0, min_valid_frames=10)
    assert not r.ok and "insufficient" in r.message().lower()
    r.frames.clear()


def test_sampling_different_fps(tmp_path):
    p = str(tmp_path / "hi.mp4")
    _make_preprocess_video(p, duration_s=4.0, fps=30)
    got = list(iter_sampled_frames(p, sample_fps=2.0, max_frames=120))
    assert 6 <= len(got) <= 10, len(got)
    assert got[0].timestamp_s == pytest.approx(0.0)
    assert got[-1].timestamp_s > got[0].timestamp_s
    q = str(tmp_path / "lo.mp4")
    _make_preprocess_video(q, duration_s=4.0, fps=5)
    got2 = list(iter_sampled_frames(q, sample_fps=2.0, max_frames=120))
    assert len(got2) == 10, len(got2)


def test_invalid_fps_fallback():
    assert _effective_fps(0.0) == 30.0
    assert _effective_fps(float("nan")) == 30.0
    assert _effective_fps(-5) == 30.0
    assert _effective_fps(float("inf")) == 30.0
    assert _effective_fps(25.0) == pytest.approx(25.0)
    assert _compute_step(30.0, 2.0) == 15
    assert _compute_step(10.0, 20.0) == 1


def test_cancel_stops(tmp_path):
    p = str(tmp_path / "long.mp4")
    _make_preprocess_video(p, duration_s=6.0, fps=10)

    def _cancel():
        return True

    r = preprocess_video(p, is_cancelled=_cancel)
    assert not r.ok and "cancel" in r.message().lower()
    r.frames.clear()


def test_capture_released_after_failure(tmp_path):
    import os

    p = tmp_path / "bad2.mp4"
    p.write_bytes(b"junk" * 50)
    preprocess_video(str(p))
    os.remove(str(p))  # PermissionError on Windows if handle leaked
    assert not p.exists()
