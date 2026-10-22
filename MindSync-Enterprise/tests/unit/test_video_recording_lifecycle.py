from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.config import (
    VIDEO_DURATION_TOLERANCE_S,
    VIDEO_MAX_DURATION_S,
    VIDEO_MIN_DURATION_S,
    VIDEO_TARGET_DURATION_S,
)
from app.ui.pages.video_analysis_page import AnalysisWorker
from app.ui.pages.webcam_recorder import RecorderWorker
from ai_services.video_facial_emotion_behavior.preprocessing.video_loader import (
    validate_video_file,
)


def test_webcam_duration_is_45_seconds():
    assert VIDEO_TARGET_DURATION_S == 45.0
    assert VIDEO_MIN_DURATION_S == VIDEO_TARGET_DURATION_S - VIDEO_DURATION_TOLERANCE_S
    assert VIDEO_MAX_DURATION_S == VIDEO_TARGET_DURATION_S + VIDEO_DURATION_TOLERANCE_S
    assert RecorderWorker()._duration == VIDEO_TARGET_DURATION_S


class _FakeCapture:
    def __init__(self, frames=True):
        self.frames = frames
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def get(self, prop):
        return {3: 160, 4: 120, 5: 20, 7: 1}.get(prop, 0)

    def read(self):
        if self.frames:
            self.frames = False
            return True, np.zeros((120, 160, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self, opened, state):
        self.opened = opened
        self.state = state
        self.state["writer"] = self
        self.path = None

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.state["frames"] += 1

    def release(self):
        self.state["released"] = True
        if self.path:
            self.path.write_bytes(b"video")


def _fake_cv2(tmp_path, *, writer_opened=True, frames=True):
    state = {"frames": 0, "released": False, "writer": None}
    camera = _FakeCapture(frames=frames)
    module = types.SimpleNamespace(
        CAP_DSHOW=700,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        VideoCapture=lambda path, *_: camera,
        VideoWriter_fourcc=lambda *args: 1,
        VideoWriter=lambda path, *_: _FakeWriter(writer_opened, state),
    )
    original_writer = module.VideoWriter

    def make_writer(path, *args):
        writer = original_writer(path, *args)
        writer.path = tmp_path / "recording.mp4"
        return writer

    module.VideoWriter = make_writer
    return module, state


def test_writer_is_released_before_finished_signal(tmp_path, monkeypatch):
    fake_cv2, state = _fake_cv2(tmp_path)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        "app.ui.pages.webcam_recorder.new_temp_video_path",
        lambda suffix: tmp_path / "recording.mp4",
    )

    worker = RecorderWorker(duration_s=0)
    observed = []
    worker.finished.connect(lambda path, wall: observed.append(state["released"]))
    worker.run()

    assert observed == [True]
    assert state["frames"] == 1


@pytest.mark.parametrize(
    ("writer_opened", "frames", "expected"),
    [(False, True, "writer"), (True, False, "no frames")],
)
def test_recording_failures_are_reported(tmp_path, monkeypatch, writer_opened, frames, expected):
    fake_cv2, _ = _fake_cv2(
        tmp_path, writer_opened=writer_opened, frames=frames)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(
        "app.ui.pages.webcam_recorder.new_temp_video_path",
        lambda suffix: tmp_path / "recording.mp4",
    )

    errors = []
    worker = RecorderWorker(duration_s=0)
    worker.error.connect(errors.append)
    worker.run()

    assert errors
    assert expected in errors[-1].lower()


def test_duration_validation_uses_tolerance(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"placeholder")

    class Props:
        readable = True
        width = 160
        height = 120
        fps = 20.0
        frame_count = 800
        duration_s = 40.0

    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        "ai_services.video_facial_emotion_behavior.preprocessing.video_loader.probe_video",
        lambda _: Props(),
    )
    try:
        assert validate_video_file(str(path)).ok
        short = validate_video_file(
            str(path), duration_override_s=VIDEO_MIN_DURATION_S - 0.1)
        assert not short.ok
        assert "too short" in short.message().lower()
    finally:
        monkey.undo()


def test_analysis_worker_surfaces_unexpected_exceptions(monkeypatch):
    monkeypatch.setattr(
        "app.ui.pages.video_analysis_page.validate_video_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    results = []
    worker = AnalysisWorker("video.mp4")
    worker.done.connect(results.append)
    worker.run()

    assert len(results) == 1
    assert "validation error" in results[0].error_message.lower()
