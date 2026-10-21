"""Step-3 face + landmark tests (no webcam needed). Run: pytest tests/unit -v."""
from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
import numpy as np

from ai_services.video_facial_emotion_behavior.landmarks import (
    FaceStageResult, crop_face, run_face_stage)


def _blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_invalid_frame_returns_no_faces():
    from ai_services.video_facial_emotion_behavior.face_detection.faces import (
        detect_faces)

    assert detect_faces(None) == []
    assert detect_faces(_blank(0, 0)) == []


def test_blank_frame_has_no_face():
    from ai_services.video_facial_emotion_behavior.face_detection.faces import (
        detect_faces)

    # Plain black frame: Haar must not crash, and must not invent a face.
    assert detect_faces(_blank()) == []


def test_crop_clamped_at_borders():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box hanging off the bottom-right corner: must clamp, not crash.
    crop = crop_face(frame, (80, 80, 50, 50), padding=0.1)
    assert crop is not None and crop.size > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100
    # Degenerate / invalid inputs -> None, never raise.
    assert crop_face(frame, None) is None
    assert crop_face(None, (10, 10, 20, 20)) is None
    assert crop_face(frame, (10, 10, 0, 0)) is None


def test_run_face_stage_no_frames():
    r = run_face_stage([], min_usable=1, min_coverage=0.0)
    assert not r.ok and "no sampled" in r.message().lower()


def test_run_face_stage_insufficient_faces(monkeypatch):
    from ai_services.video_facial_emotion_behavior.preprocessing.frames import (
        SampledFrame)

    # Force detector to find nothing: must report insufficient, not crash.
    import ai_services.video_facial_emotion_behavior.landmarks as stage

    monkeypatch.setattr(
        "ai_services.video_facial_emotion_behavior.face_detection.faces"
        ".detect_faces", lambda *a, **k: [])
    frames = [SampledFrame(index=i, timestamp_s=i * 0.5, frame=_blank())
              for i in range(4)]
    r = stage.run_face_stage(frames, min_usable=2, min_coverage=0.1)
    assert not r.ok and "too few usable" in r.message().lower()
    assert r.without_face == 4
    r.clear()


def test_run_face_stage_mocked_face_no_mesh(monkeypatch):
    from ai_services.video_facial_emotion_behavior.preprocessing.frames import (
        SampledFrame)
    import ai_services.video_facial_emotion_behavior.landmarks as stage

    monkeypatch.setattr(
        "ai_services.video_facial_emotion_behavior.face_detection.faces"
        ".detect_faces", lambda *a, **k: [(50, 30, 60, 60)])
    # Mesh missing -> landmarks None, but crop kept (CNN can still use it).
    monkeypatch.setattr(
        "ai_services.video_facial_emotion_behavior.landmarks.mesh"
        ".extract_landmarks_with_backend", lambda *a, **k: (None, "none"))
    frames = [SampledFrame(index=i, timestamp_s=i * 0.5,
                           frame=np.full((120, 160, 3), 128, dtype=np.uint8))
              for i in range(4)]
    r = stage.run_face_stage(frames, min_usable=2, min_coverage=0.1)
    assert r.ok, r.message()
    assert r.with_face == 4 and len(r.usable) == 4
    assert all(f.crop is not None and f.landmark_count == 0
               for f in r.usable)
    assert any("mediapipe" in w.lower() for w in r.warnings)
    # Privacy: clear() drops the numpy refs.
    r.clear()
    assert r.usable == []


def test_mesh_invalid_input_safe():
    from ai_services.video_facial_emotion_behavior.landmarks.mesh import (
        close, extract_landmarks)

    assert extract_landmarks(None) is None
    assert extract_landmarks(_blank(0, 0)) is None
    close()  # must not raise even when never opened


def test_mesh_reports_real_count_or_none():
    # No webcam needed: use a blank crop. Either None (no face / no model)
    # or a real count from the installed config — never a fake 468.
    from ai_services.video_facial_emotion_behavior.landmarks.mesh import (
        close, extract_landmarks_with_backend)

    pts, backend = extract_landmarks_with_backend(_blank())
    assert backend in ("tasks", "solutions", "none")
    if pts is not None:
        assert len(pts) > 0
    close()
