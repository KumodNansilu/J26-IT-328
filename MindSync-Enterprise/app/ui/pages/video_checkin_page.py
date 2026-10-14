"""Video Check-in page: Upload Video OR Record Video (webcam)."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QProgressBar,
                               QVBoxLayout, QWidget)

from app.config import VIDEO_PICKER_FILTER, VIDEO_TARGET_DURATION_S
from app.services.video_input_service import validate_video_input
from app.utils.paths import is_temp_mindsync_path
from app.ui.pages.webcam_recorder import RecorderWorker


def _bgr_to_pixmap(frame, max_w: int = 640, max_h: int = 480):
    """Convert BGR numpy frame -> QPixmap (copies data so it stays valid).

    Previous version passed rgb.data directly into QImage without keeping
    the numpy array alive — the QImage then pointed at freed memory, which
    shows as black / flickering / torn preview frames.
    """
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # .copy() detaches the QImage from the numpy buffer lifetime.
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    # Fit inside max_w x max_h while keeping aspect ratio (no stretching,
    # no side-bar distortion on wide windows).
    if pm.width() > max_w or pm.height() > max_h:
        pm = pm.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


class VideoCheckinPage(QWidget):
    """Two-option check-in screen. Exposes selected_path for analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_path: str | None = None
        self._is_temp_recording = False
        # Wall-clock seconds for the freshest webcam take (header FPS can
        # under-report; uploads pass None to use header duration).
        self._recorded_wall_s: float | None = None
        self._thread: QThread | None = None
        self._worker: RecorderWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("Video Check-in")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(title)
        info = QLabel("Upload a ~30-second video, or record one with your webcam. "
                      "Analysis starts only after you confirm. "
                      "Your original file is never deleted.")
        info.setWordWrap(True)
        root.addWidget(info)

        up = QGroupBox("Option 1 — Upload Video")
        ul = QVBoxLayout(up)
        row = QHBoxLayout()
        self.btn_upload = QPushButton("Choose Video…")
        self.btn_upload.clicked.connect(self.choose_file)
        row.addWidget(self.btn_upload)
        self.lbl_file = QLabel("No file chosen.")
        self.lbl_file.setWordWrap(True)
        row.addWidget(self.lbl_file, 1)
        ul.addLayout(row)
        root.addWidget(up)

        rec = QGroupBox("Option 2 — Record Video (30 s)")
        rl = QVBoxLayout(rec)
        self.lbl_preview = QLabel("Camera preview appears here.")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumHeight(360)
        self.lbl_preview.setStyleSheet("background:#111; color:#eee;")
        self.lbl_preview.setScaledContents(False)
        self.lbl_preview.setWordWrap(True)
        rl.addWidget(self.lbl_preview, 1)
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop Recording")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_cancel.clicked.connect(self.cancel_recording)
        for b in (self.btn_start, self.btn_stop, self.btn_cancel):
            btns.addWidget(b)
        rl.addLayout(btns)
        self.lbl_count = QLabel(f"Countdown: {VIDEO_TARGET_DURATION_S:.0f}s")
        rl.addWidget(self.lbl_count)
        root.addWidget(rec)

        det = QGroupBox("Selected video")
        fl = QFormLayout(det)
        self.lbl_details = QLabel("--")
        self.lbl_details.setWordWrap(True)
        fl.addRow("Details:", self.lbl_details)
        self.lbl_validation = QLabel("Not validated yet.")
        self.lbl_validation.setWordWrap(True)
        fl.addRow("Validation:", self.lbl_validation)
        root.addWidget(det)

        acts = QHBoxLayout()
        self.btn_rechoose = QPushButton("Choose Another / Retake")
        self.btn_rechoose.clicked.connect(self.reset_selection)
        self.btn_continue = QPushButton("Continue to Analysis")
        self.btn_continue.setEnabled(False)
        self.btn_continue.clicked.connect(self.continue_to_analysis)
        acts.addWidget(self.btn_rechoose)
        acts.addWidget(self.btn_continue)
        root.addLayout(acts)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.lbl_status = QLabel("Status: idle")
        root.addWidget(self.lbl_status)
        root.addStretch(1)

    @Slot()
    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select check-in video", "", VIDEO_PICKER_FILTER)
        if not path:
            return
        self._discard_temp_recording()
        self.selected_path = path
        self._is_temp_recording = False
        self._recorded_wall_s = None
        self.lbl_file.setText(path)
        self.run_validation()

    @Slot()
    def start_recording(self):
        if self._thread is not None:
            QMessageBox.information(self, "Recording", "A recording is already running.")
            return
        self._discard_temp_recording()
        self._thread = QThread(self)
        self._worker = RecorderWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.tick.connect(lambda s: self.lbl_count.setText(f"Countdown: {s}s"))
        self._worker.preview.connect(self._on_preview)
        self._worker.hint.connect(lambda m: self.lbl_status.setText("Status: recording... " + m))
        self._worker.finished.connect(self._on_recorded)
        self._worker.error.connect(self._on_rec_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._teardown_thread)
        self.lbl_status.setText("Status: recording...")
        self._thread.start()

    @Slot()
    def stop_recording(self):
        if self._worker:
            self._worker.request_stop()

    @Slot()
    def cancel_recording(self):
        if self._worker:
            self._worker.request_cancel()
        else:
            self._discard_temp_recording()

    @Slot(object)
    def _on_preview(self, frame):
        try:
            self.lbl_preview.setPixmap(_bgr_to_pixmap(frame))
        except Exception:
            pass

    @Slot(str, float)
    def _on_recorded(self, path, wall_s=0.0):
        self.selected_path = path
        self._is_temp_recording = True
        # Keep recorder wall-clock for this take only; a re-upload clears it.
        self._recorded_wall_s = float(wall_s) if wall_s and wall_s > 0 else None
        self.lbl_file.setText("(webcam recording) " + path)
        self.lbl_status.setText("Status: recording saved - please confirm.")
        self.run_validation()

    @Slot(str)
    def _on_rec_error(self, msg):
        if msg == "__CANCELLED__":
            self.lbl_status.setText("Status: recording cancelled.")
            self.lbl_preview.setText("Camera preview appears here.")
            self.lbl_count.setText(f"Countdown: {VIDEO_TARGET_DURATION_S:.0f}s")
            return
        QMessageBox.warning(self, "Webcam", msg)
        self.lbl_status.setText("Status: error - " + msg)

    def _teardown_thread(self):
        if self._worker:
            self._worker.deleteLater()
        self._worker = None
        if self._thread:
            self._thread.deleteLater()
        self._thread = None

    @Slot()
    def run_validation(self):
        if not self.selected_path:
            return
        self.progress.setVisible(True)
        self.lbl_status.setText("Status: validating...")
        # Only a fresh webcam take carries wall-clock; uploads use None
        # so header duration applies.
        wall = self._recorded_wall_s if self._is_temp_recording else None
        res = validate_video_input(self.selected_path, duration_override_s=wall)
        self.progress.setVisible(False)
        if res.ok:
            d = res.details
            self.lbl_details.setText(
                os.path.basename(self.selected_path or "")
                + f" - face {d.get('usable')}/{d.get('total')} frames")
            self.lbl_validation.setText("Passed: " + res.message)
            self.lbl_status.setText("Status: valid - review, then Continue to Analysis.")
            self.btn_continue.setEnabled(True)
        else:
            self.lbl_validation.setText("Failed: " + res.message)
            self.lbl_status.setText("Status: invalid video.")
            self.btn_continue.setEnabled(False)
            QMessageBox.warning(self, "Validation", res.message)

    @Slot()
    def reset_selection(self):
        self._discard_temp_recording()
        self.selected_path = None
        self._recorded_wall_s = None
        self.lbl_file.setText("No file chosen.")
        self.lbl_details.setText("--")
        self.lbl_validation.setText("Not validated yet.")
        self.lbl_status.setText("Status: idle")
        self.btn_continue.setEnabled(False)

    @Slot()
    def continue_to_analysis(self):
        if not self.selected_path:
            return
        QMessageBox.information(
            self, "Confirmed",
            "Video confirmed and validated:\n" + str(self.selected_path)
            + "\n\nNext phase: frame > face > MediaPipe > CNN > VBS.")

    def _discard_temp_recording(self):
        if self._is_temp_recording and self.selected_path \
                and is_temp_mindsync_path(self.selected_path):
            try:
                if os.path.exists(self.selected_path):
                    os.remove(self.selected_path)
            except OSError:
                pass
        self._is_temp_recording = False

    def closeEvent(self, event):
        try:
            if self._worker:
                self._worker.request_cancel()
            if self._thread:
                self._thread.quit()
                self._thread.wait(3000)
        finally:
            super().closeEvent(event)

