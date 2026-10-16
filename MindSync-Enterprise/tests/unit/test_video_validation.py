"""Step-1 validation tests (no webcam needed). Run: pytest tests/unit -v"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("numpy")

from ai_services.video_facial_emotion_behavior.preprocessing.video_loader import (
    probe_video, validate_video_file)


def _make_validation_video(path, *, duration_s=2.0, fps=10, w=160, h=120):
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    assert out.isOpened()
    for i in range(int(duration_s * fps)):
        out.write(np.full((h, w, 3), 200 - i % 50, dtype=np.uint8))
    out.release()


def test_missing_file(tmp_path):
    r = validate_video_file(str(tmp_path / "nope.mp4"))
    assert not r.ok and "not found" in r.message().lower()


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    r = validate_video_file(str(p))
    assert not r.ok and "unsupported" in r.message().lower()


def test_short_video_rejected(tmp_path):
    p = str(tmp_path / "short.mp4")
    _make_validation_video(p, duration_s=2.0)
    r = validate_video_file(p, min_duration_s=25.0, max_duration_s=35.0)
    assert not r.ok and "too short" in r.message().lower()
    assert r.props is not None and r.props.readable


def test_valid_window_accepts(tmp_path):
    p = str(tmp_path / "ok.mp4")
    _make_validation_video(p, duration_s=30.0, fps=10)
    r = validate_video_file(p, min_duration_s=25.0, max_duration_s=35.0)
    assert r.ok, r.message()


def test_corrupt_file(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"not a video" * 100)
    r = validate_video_file(str(p))
    assert not r.ok


def test_aggregator_and_probs():
    from ai_services.video_facial_emotion_behavior.emotion_recognition.classifier import (
        aggregate_frame_probs, validate_probs)
    from ai_services.video_facial_emotion_behavior.scoring.vbs import (
        VBSSpecMissingError, compute_vbs)

    ok, _ = validate_probs([0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2])
    assert ok
    bad, _ = validate_probs([0.5, 0.5])  # wrong length
    assert not bad
    mean = aggregate_frame_probs([[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]])
    assert abs(sum(mean) - 1.0) < 1e-6 and mean[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        aggregate_frame_probs([])
    with pytest.raises(VBSSpecMissingError):
        compute_vbs(mean)  # no approved formula -> must refuse, not invent


