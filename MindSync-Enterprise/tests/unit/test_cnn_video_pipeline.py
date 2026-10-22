"""Focused CNN + video pipeline tests using synthetic fixtures.

Fixtures are built in tmp_path from FER2013 *Training* images
(PrivateTest never touched) or from solid-color arrays. No media is committed.
No VBS/DSI logic is touched; the VBS gate is only exercised to confirm refusal.
"""
from __future__ import annotations

import csv
import os
import sys

import pytest

ROOT = r"C:\Users\User\Desktop\J26-IT-328\MindSync-Enterprise"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ai_services.video_facial_emotion_behavior.emotion_recognition import predict as P  # noqa: E402
from ai_services.video_facial_emotion_behavior.emotion_recognition.classifier import (  # noqa: E402
    ModelNotAvailableError,
    aggregate_frame_probs,
    validate_probs,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.cnn_model import (  # noqa: E402
    build_model,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (  # noqa: E402
    EMOTIONS,
)
from ai_services.video_facial_emotion_behavior.face_detection.faces import (  # noqa: E402
    detect_faces,
)
from ai_services.video_facial_emotion_behavior.inference.pipeline import (  # noqa: E402
    run_emotion_stage,
)
from ai_services.video_facial_emotion_behavior.landmarks import (  # noqa: E402
    run_face_stage,
)
from ai_services.video_facial_emotion_behavior.preprocessing.frames import (  # noqa: E402
    preprocess_video,
)
from ai_services.video_facial_emotion_behavior.preprocessing.video_loader import (  # noqa: E402
    validate_video_file,
)
from ai_services.video_facial_emotion_behavior.scoring.vbs import (  # noqa: E402
    VBSSpecMissingError,
    compute_vbs,
)

CSV_PATH = os.path.join("data", "fer2013", "fer2013.csv")
_DETECTABLE_CACHE: list | None = None


def _detectable_training_images(n: int = 8, size: int = 200):
    """Return up to n (emotion, gray_48x48) FER2013 Training rows Haar detects."""
    global _DETECTABLE_CACHE
    if _DETECTABLE_CACHE is not None:
        return _DETECTABLE_CACHE[:n]
    found = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("Usage") or "").strip() != "Training":
                continue
            parts = (row.get("pixels") or "").split()
            if len(parts) != 2304:
                continue
            g = np.array(parts, dtype=np.uint8).reshape(48, 48)
            up = cv2.cvtColor(
                cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_GRAY2BGR,
            )
            if detect_faces(up, min_size=60):
                found.append((int(row["emotion"]), g))
            if len(found) >= 16:
                break
    _DETECTABLE_CACHE = found
    return _DETECTABLE_CACHE[:n]


def _write_mp4(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), f"could not open VideoWriter {path}"
    for f in frames:
        vw.write(f)
    vw.release()


def _face_video(tmp_path, images, *, seconds=5.0, fps=10, size=200):
    per = [cv2.cvtColor(cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_GRAY2BGR) for _, g in images]
    frames = [per[i % len(per)] for i in range(int(seconds * fps))]
    p = str(tmp_path / "face.mp4")
    _write_mp4(p, frames, fps=fps)
    return p


def _blank_video(tmp_path, *, seconds=5.0, fps=10, size=200):
    frames = [np.full((size, size, 3), 190, dtype=np.uint8)
              for _ in range(int(seconds * fps))]
    p = str(tmp_path / "blank.mp4")
    _write_mp4(p, frames, fps=fps)
    return p


def _two_face_video(tmp_path, images, *, seconds=5.0, fps=10):
    per = [cv2.cvtColor(cv2.resize(g, (160, 160), interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_GRAY2BGR) for _, g in images]
    canvas = np.full((240, 560, 3), 200, dtype=np.uint8)
    frames = []
    for i in range(int(seconds * fps)):
        canvas[40:200, 40:200] = per[i % len(per)]
        canvas[40:200, 320:480] = per[(i + 1) % len(per)]
        frames.append(canvas.copy())
    p = str(tmp_path / "two.mp4")
    _write_mp4(p, frames, fps=fps)
    return p


@pytest.fixture(scope="module", autouse=True)
def _ensure_model():
    info = P.load_model()
    assert info.get("loaded") is True, info
    yield


# ---------------------------------------------------------------------------
# 1. valid faces end-to-end
# ---------------------------------------------------------------------------
def test_valid_faces_end_to_end(tmp_path):
    images = _detectable_training_images(8)
    assert len(images) >= 4, "need detectable FER2013 Training fixtures"
    path = _face_video(tmp_path, images, seconds=45.0, fps=10)

    vr = validate_video_file(path)
    assert vr.ok, vr.message()

    pre = preprocess_video(path, sample_fps=2.0, max_frames=120, min_valid_frames=10)
    assert pre.ok, pre.message()
    assert pre.sampled_count >= 2

    face = run_face_stage(pre.frames, min_usable=4, min_coverage=0.2)
    assert face.with_face > 0, face.message()
    assert len(face.usable) > 0

    emo = run_emotion_stage(face.usable)
    assert emo.ok, emo.message()
    assert len(emo.frames) > 0
    assert emo.skipped == 0

    probs = [f.probs for f in emo.frames]
    assert all(validate_probs(p)[0] for p in probs)
    mean = aggregate_frame_probs(probs)
    assert abs(sum(mean) - 1.0) < 1e-6
    assert EMOTIONS[int(np.argmax(mean))] in EMOTIONS

    face.clear()
    pre.frames.clear()


# 2. no faces reported honestly (no fake Neutral)
def test_no_faces_reported_honestly(tmp_path):
    path = _blank_video(tmp_path, seconds=5.0, fps=10)
    pre = preprocess_video(path, sample_fps=2.0, max_frames=120, min_valid_frames=2)
    assert pre.ok
    assert pre.sampled_count > 0

    face = run_face_stage(pre.frames, min_usable=10, min_coverage=0.3)
    assert not face.ok
    assert face.with_face == 0
    assert "face" in face.message().lower()

    # CNN must refuse empty input, never invent Neutral
    emo = run_emotion_stage(face.usable)
    assert not emo.ok
    pre.frames.clear()


# 3. corrupt / unreadable video rejected at both validation and preprocessing
def test_corrupt_video_rejected(tmp_path):
    p = str(tmp_path / "bad.mp4")
    with open(p, "wb") as fh:
        fh.write(b"not a video at all" * 100)
    vr = validate_video_file(p)
    assert not vr.ok
    pr = preprocess_video(p)
    assert not pr.ok


# 4. missing file rejected
def test_missing_video_rejected(tmp_path):
    missing = str(tmp_path / "nope.mp4")
    vr = validate_video_file(missing)
    assert not vr.ok and "not found" in vr.message().lower()
    pr = preprocess_video(missing)
    assert not pr.ok


# 5. missing model artifact raises, never falls back
def test_missing_model_artifact_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DEFAULT_CKPT", str(tmp_path / "nope.pt"))
    with pytest.raises(ModelNotAvailableError):
        P.load_model()


# 6. invalid metadata raises (label order must match model output)
def test_invalid_metadata_raises(tmp_path, monkeypatch):
    import json

    d = str(tmp_path / "meta_bad")
    os.makedirs(d, exist_ok=True)
    ckpt = os.path.join(d, "w.pt")
    torch.save(build_model().state_dict(), ckpt)
    meta = os.path.join(d, "w.json")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({"label_order": ["Z"] * 7, "input": [1, 48, 48]}, fh)
    monkeypatch.setattr(P, "DEFAULT_CKPT", ckpt)
    monkeypatch.setattr(P, "DEFAULT_META", meta)
    with pytest.raises(ModelNotAvailableError) as cm:
        P.load_model()
    assert "label order" in str(cm.value).lower()


# 7. multiple faces per frame -> largest-box policy, one crop per frame
def test_multiple_faces_largest_box(tmp_path):
    images = _detectable_training_images(8)
    path = _two_face_video(tmp_path, images, seconds=5.0, fps=10)
    pre = preprocess_video(path, sample_fps=2.0, max_frames=120, min_valid_frames=10)
    assert pre.ok
    face = run_face_stage(pre.frames, min_usable=4, min_coverage=0.2)
    assert face.with_face > 0
    # largest-box policy: at most one crop per frame
    assert len(face.usable) <= pre.sampled_count
    emo = run_emotion_stage(face.usable)
    if emo.ok:
        assert len(emo.frames) == len(face.usable)
        probs = [f.probs for f in emo.frames]
        assert all(validate_probs(p)[0] for p in probs)
    face.clear()
    pre.frames.clear()


# 8. resource release / privacy: no leaked video handles, file removable
def test_video_handle_released_after_run(tmp_path):
    images = _detectable_training_images(4)
    path = _face_video(tmp_path, images, seconds=3.0, fps=10)
    pre = preprocess_video(path, sample_fps=2.0, max_frames=60, min_valid_frames=2)
    face = run_face_stage(pre.frames, min_usable=1, min_coverage=0.0)
    face.clear()
    pre.frames.clear()
    # On Windows the file must be removable (no leaked VideoCapture handle)
    os.remove(path)
    assert not os.path.exists(path)


# 9. VBS gate refuses to score without an approved formula
def test_vbs_gate_refuses(tmp_path):
    images = _detectable_training_images(4)
    path = _face_video(tmp_path, images, seconds=3.0, fps=10)
    pre = preprocess_video(path, sample_fps=2.0, max_frames=60, min_valid_frames=2)
    face = run_face_stage(pre.frames, min_usable=1, min_coverage=0.0)
    emo = run_emotion_stage(face.usable)
    face.clear()
    pre.frames.clear()
    if emo.ok:
        mean = aggregate_frame_probs([f.probs for f in emo.frames])
        with pytest.raises(VBSSpecMissingError):
            compute_vbs(mean)
