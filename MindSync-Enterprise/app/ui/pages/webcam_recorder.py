"""Webcam recorder worker — runs on a QThread, never blocks the UI.

Writes mp4 to a temp path (verifies it reopens), emits countdown ticks.
Camera is released in ALL paths (stop / cancel / error / thread finish).
"""
from __future__ import annotations

import os
import time
import logging

from PySide6.QtCore import QObject, QThread, Signal

from app.config import (WEBCAM_DEVICE_INDEX, WEBCAM_FRAME_HEIGHT,
                        WEBCAM_FRAME_WIDTH, WEBCAM_FOURCC, WEBCAM_RECORD_FPS,
                        VIDEO_TARGET_DURATION_S)
from app.utils.paths import new_temp_video_path

logger = logging.getLogger(__name__)


class RecorderWorker(QObject):
    tick = Signal(int)            # seconds remaining
    preview = Signal(object)      # BGR numpy frame (for live preview)
    hint = Signal(str)            # non-blocking notice (e.g. lens covered)
    # path + wall-clock seconds actually spent recording. Header FPS can lie
    # (see comment in run()), so the page validates with wall time for
    # fresh webcam takes, header duration for uploads.
    finished = Signal(str, float)  # temp video path, wall duration s
    error = Signal(str)

    def __init__(self, duration_s: float = VIDEO_TARGET_DURATION_S,
                 parent: QObject | None = None):
        super().__init__(parent)
        self._duration = float(duration_s)
        self._stop = False
        self._cancel = False

    def request_stop(self):
        self._stop = True

    def request_cancel(self):
        self._cancel = True
        self._stop = True

    def run(self):
        import cv2

        out_path = str(new_temp_video_path(".mp4"))
        cap = cv2.VideoCapture(WEBCAM_DEVICE_INDEX, cv2.CAP_DSHOW)
        writer = None
        frames_written = 0
        wall_s = 0.0
        recording_failed = False
        try:
            if not cap.isOpened():
                self.error.emit("Webcam unavailable (not found, busy, or no permission).")
                recording_failed = True
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_FRAME_HEIGHT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or WEBCAM_FRAME_WIDTH)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or WEBCAM_FRAME_HEIGHT)
            fourcc = cv2.VideoWriter_fourcc(*WEBCAM_FOURCC)
            writer = cv2.VideoWriter(out_path, fourcc, WEBCAM_RECORD_FPS, (w, h))
            if not writer.isOpened():
                self.error.emit("Could not open video writer (codec issue).")
                recording_failed = True
                return
            # Start the recording clock only after both camera and writer are ready.
            t0 = time.monotonic()
            stop_at = t0 + self._duration
            frames_written = 0
            # Paced capture: aim for WEBCAM_RECORD_FPS wall-clock.
            # Old code slept a fixed 33ms + read/write cost, so real capture
            # ran slower (~15fps) than the 20fps file header -> a 45s wall
            # take was stamped as ~22s and then failed the 25s check.
            frame_interval = 1.0 / float(WEBCAM_RECORD_FPS) if WEBCAM_RECORD_FPS > 0 else 0.05
            while not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    self.error.emit(
                        "Webcam returned an empty frame; no frames were written. "
                        "Please check the webcam and try again.")
                    recording_failed = True
                    break
                # Warn once if the camera sends a dark/blocked placeholder
                # (physical shutter closed, kill-switch, or lens covered).
                try:
                    import numpy as _np

                    if float(_np.mean(frame)) < 4.0 and not getattr(self, "_dark_warned", False):
                        self._dark_warned = True
                        self.preview.emit(frame.copy())
                        self.hint.emit(
                            "Camera looks covered — check the physical shutter slider, "
                            "lens cover, or kill-switch. Recording continues.")
                        # Don't break: keep recording so user can uncover mid-take.
                except Exception:
                    pass
                writer.write(frame)
                frames_written += 1
                # Copy: the next cap.read() reuses the buffer, and the queued
                # signal may arrive after that. Emitting the original would
                # show torn/flickering preview frames.
                self.preview.emit(frame.copy())
                remain = int(max(0, round(stop_at - time.monotonic())))
                self.tick.emit(remain)
                if time.monotonic() >= stop_at:
                    break
                # Pace to the target FPS: sleep only the remainder of one
                # frame interval. Fixed 33ms + read/write overhead ran slower
                # than the 20fps header, stamping 45s wall takes as ~33s.
                elapsed = time.monotonic() - t0
                expected = frames_written * frame_interval
                sleep_s = expected - elapsed
                if sleep_s > 0:
                    QThread.msleep(int(sleep_s * 1000))

            wall_s = time.monotonic() - t0
            if self._cancel:
                self.error.emit("__CANCELLED__")
        except Exception as e:  # noqa: BLE001 — surface to UI
            logger.exception("Unhandled webcam recording failure")
            self.error.emit(f"Recording failed: {e}")
            recording_failed = True
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                cap.release()
        if self._cancel:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return
        if recording_failed:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return
        if frames_written <= 0:
            self.error.emit("Recording produced no frames. Please check the webcam and try again.")
            return
        if not os.path.exists(out_path) or os.path.getsize(out_path) <= 0:
            self.error.emit("Recording produced an empty video file. Please try again.")
            return
        # Validation must never race an open writer. Probe only after release.
        import cv2 as _cv

        probe = _cv.VideoCapture(out_path)
        try:
            if not probe.isOpened():
                self.error.emit("Recorded video could not be reopened. Please try again.")
                return
            count = int(probe.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(probe.get(_cv.CAP_PROP_FPS) or 0.0)
            if count <= 0 or fps <= 0:
                self.error.emit("Recorded video has invalid frame metadata. Please try again.")
                return
            logger.info(
                "Webcam recording finalized: wall_duration=%.3fs, "
                "header_duration=%.3fs, fps=%.3f, frames=%d, size=%d bytes",
                wall_s, count / fps, fps, count, os.path.getsize(out_path))
        finally:
            probe.release()
        self.finished.emit(out_path, float(wall_s))
