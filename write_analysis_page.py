import os

path = r"c:\Users\User\Desktop\J26-IT-328\MindSync-Enterprise\app\ui\pages\video_analysis_page.py"

part1 = '''"""Video analysis results page — displays real pipeline outputs.

Memory only: frames, face crops, landmarks, and raw predictions are never
written to disk or database. All numbers come from the live pipeline run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.config import (
    EXPECTED_LANDMARK_COUNT, FACE_CROP_PADDING, FACE_MIN_SIZE_PX,
    FRAME_SAMPLE_FPS, FRAME_SAMPLE_MAX_FRAMES, MIN_FACE_COVERAGE,
    MIN_USABLE_FRAMES, MIN_VALID_FRAMES,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.classifier import (
    ModelNotAvailableError, aggregate_frame_probs, validate_probs)
from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (
    EMOTIONS, NUM_EMOTIONS)
from ai_services.video_facial_emotion_behavior.emotion_recognition.predict import (
    load_model as predict_load_model)
from ai_services.video_facial_emotion_behavior.inference.pipeline import (
    run_emotion_stage)
from ai_services.video_facial_emotion_behavior.landmarks import (
    run_face_stage)
from ai_services.video_facial_emotion_behavior.preprocessing.frames import (
    preprocess_video)
from ai_services.video_facial_emotion_behavior.preprocessing.video_loader import (
    validate_video_file)
from ai_services.video_facial_emotion_behavior.scoring.vbs import (
    VBSSpecMissingError, compute_vbs)


@dataclass
class AnalysisResult:
    """Collected pipeline outputs for display (memory only, no persistence).

    Every field is populated from the real pipeline run. Nothing is
    hard-coded or invented.
    """

    ok: bool = False
    error_message: str = ""

    validation_ok: bool = False
    validation_message: str = ""
    width: int = 0
    height: int = 0
    duration_s: float = 0.0

    sampled_count: int = 0
    scanned_count: int = 0
    dropped_count: int = 0

    face_total: int = 0
    face_with_face: int = 0
    face_without_face: int = 0
    face_coverage: float = 0.0
    face_errors: list = field(default_factory=list)
    face_warnings: list = field(default_factory=list)

    emotion_ok: bool = False
    emotion_frames: list = field(default_factory=list)
    emotion_skipped: int = 0
    emotion_errors: list = field(default_factory=list)

    aggregated_probs: list | None = None
    top_emotion: str | None = None
    top_prob: float = 0.0

    model_load_time_s: float = 0.0
    inference_time_s: float = 0.0
    total_time_s: float = 0.0

    vbs_available: bool = False
    vbs_message: str = ""

    disclaimer: str = (
        "Emotion predictions are experimental facial-expression outputs "
        "and are NOT a medical diagnosis."
    )
'''

with open(path, "w", encoding="utf-8") as f:
    f.write(part1)
part2a = '''

class AnalysisWorker(QObject):
    """Run the full video-analysis pipeline off the UI thread."""

    progress = Signal(int)
    done = Signal(object)

    def __init__(self, video_path: str, parent: QObject | None = None):
        super().__init__(parent)
        self._path = video_path
        self._cancel = False

    @Slot()
    def request_cancel(self):
        self._cancel = True

    def _is_cancelled(self) -> bool:
        return self._cancel

    @Slot()
    def run(self):
        t0 = time.perf_counter()
        result = AnalysisResult()

        def _pct(v: int):
            try:
                self.progress.emit(min(100, max(0, v)))
            except Exception:
                pass

        _pct(5)
        try:
            vr = validate_video_file(
                self._path,
                allowed_extensions=(".mp4", ".avi", ".mov"),
                min_duration_s=25.0, max_duration_s=35.0,
            )
            result.validation_ok = vr.ok
            result.validation_message = vr.message()
            if vr.props:
                result.width = vr.props.width
                result.height = vr.props.height
                result.duration_s = vr.props.duration_s
        except Exception as exc:
            result.validation_ok = False
            result.validation_message = f"Validation error: {exc}"
            result.error_message = result.validation_message
            _pct(100)
            self.done.emit(result)
            return

        if not vr.ok:
            _pct(100)
            self.done.emit(result)
            return

        _pct(20)
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part2a)
part2b = '''
        pre = None
        try:
            pre = preprocess_video(
                self._path,
                sample_fps=FRAME_SAMPLE_FPS,
                max_frames=FRAME_SAMPLE_MAX_FRAMES,
                min_valid_frames=MIN_VALID_FRAMES,
                is_cancelled=self._is_cancelled,
                on_progress=lambda kept, cap_max: _pct(
                    20 + int(15 * kept / max(cap_max, 1))),
            )
            result.sampled_count = pre.sampled_count
            result.scanned_count = pre.scanned_count
            result.dropped_count = pre.dropped_count
        except Exception as exc:
            result.error_message = f"Frame sampling failed: {exc}"
            _pct(100)
            self.done.emit(result)
            return

        if not pre.ok:
            result.error_message = pre.message()
            _pct(100)
            self.done.emit(result)
            return

        _pct(35)
        face = None
        try:
            face = run_face_stage(
                pre.frames,
                min_size=FACE_MIN_SIZE_PX,
                padding=FACE_CROP_PADDING,
                min_usable=MIN_USABLE_FRAMES,
                min_coverage=MIN_FACE_COVERAGE,
                expected_landmarks=EXPECTED_LANDMARK_COUNT,
                is_cancelled=self._is_cancelled,
            )
            result.face_total = face.total
            result.face_with_face = face.with_face
            result.face_without_face = face.without_face
            result.face_coverage = face.coverage
            result.face_errors = list(face.errors)
            result.face_warnings = list(face.warnings)
        except Exception as exc:
            result.error_message = f"Face stage failed: {exc}"
            _pct(100)
            self.done.emit(result)
            return

        if not face.ok:
            result.error_message = face.message()
            _pct(100)
            self.done.emit(result)
            return

        _pct(50)
        model_load_t0 = time.perf_counter()
        try:
            predict_load_model()
            result.model_load_time_s = time.perf_counter() - model_load_t0
        except ModelNotAvailableError as exc:
            result.error_message = f"Model not available: {exc}"
            _pct(100)
            self.done.emit(result)
            return
        except Exception as exc:
            result.error_message = f"Model load failed: {exc}"
            _pct(100)
            self.done.emit(result)
            return
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part2b)
part2c = '''
        _pct(60)
        inference_t0 = time.perf_counter()
        try:
            emo = run_emotion_stage(face.usable, is_cancelled=self._is_cancelled)
            result.inference_time_s = time.perf_counter() - inference_t0
            result.emotion_ok = emo.ok
            result.emotion_skipped = emo.skipped
            result.emotion_errors = list(emo.errors)
            result.emotion_frames = [
                {"timestamp_s": f.timestamp_s, "probs": [float(v) for v in f.probs]}
                for f in emo.frames
            ]
        except Exception as exc:
            result.inference_time_s = time.perf_counter() - inference_t0
            result.error_message = f"Emotion inference failed: {exc}"
            _pct(100)
            self.done.emit(result)
            return

        if not emo.ok:
            result.error_message = emo.message()
            _pct(100)
            self.done.emit(result)
            return

        _pct(80)
        try:
            probs_list = [fr["probs"] for fr in result.emotion_frames]
            agg = aggregate_frame_probs(probs_list)
            result.aggregated_probs = [float(v) for v in agg]
            top_idx = int(max(range(NUM_EMOTIONS), key=lambda i: agg[i]))
            result.top_emotion = EMOTIONS[top_idx]
            result.top_prob = float(agg[top_idx])
        except ValueError as exc:
            result.error_message = f"Aggregation failed: {exc}"
            _pct(100)
            self.done.emit(result)
            return

        _pct(90)
        try:
            compute_vbs(result.aggregated_probs)
            result.vbs_available = True
            result.vbs_message = "VBS computed."
        except VBSSpecMissingError:
            result.vbs_available = False
            result.vbs_message = (
                "Not available — scoring method requires approval"
            )
        except Exception as exc:
            result.vbs_available = False
            result.vbs_message = f"VBS error: {exc}"

        result.total_time_s = time.perf_counter() - t0
        result.ok = True
        try:
            face.clear()
        except Exception:
            pass
        try:
            pre.frames.clear()
        except Exception:
            pass

        _pct(100)
        self.done.emit(result)
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part2c)
part3a = '''

class VideoAnalysisPage(QWidget):
    """Displays real video-analysis pipeline outputs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str | None = None
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._build_ui()

    def set_video_path(self, path: str):
        self._video_path = path
        self._reset_display()

    def _build_ui(self):
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        form = QVBoxLayout(container)

        self.lbl_title = QLabel("Video Analysis Results")
        self.lbl_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        form.addWidget(self.lbl_title)

        self.lbl_video = QLabel("No video selected.")
        self.lbl_video.setWordWrap(True)
        form.addWidget(self.lbl_video)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        form.addWidget(self.progress)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_analysis)
        form.addWidget(self.btn_cancel)

        grp_val = QGroupBox("Video Validation")
        fl = QGridLayout(grp_val)
        self.lbl_val_status = QLabel("\\u2014")
        self.lbl_val_message = QLabel("\\u2014")
        self.lbl_val_message.setWordWrap(True)
        self.lbl_val_dims = QLabel("\\u2014")
        self.lbl_val_duration = QLabel("\\u2014")
        fl.addWidget(QLabel("Status:"), 0, 0)
        fl.addWidget(self.lbl_val_status, 0, 1)
        fl.addWidget(QLabel("Message:"), 1, 0)
        fl.addWidget(self.lbl_val_message, 1, 1)
        fl.addWidget(QLabel("Dimensions:"), 2, 0)
        fl.addWidget(self.lbl_val_dims, 2, 1)
        fl.addWidget(QLabel("Duration:"), 3, 0)
        fl.addWidget(self.lbl_val_duration, 3, 1)
        form.addWidget(grp_val)

        grp_pipe = QGroupBox("Pipeline Summary")
        fl = QGridLayout(grp_pipe)
        self.lbl_pipe_sampled = QLabel("\\u2014")
        self.lbl_pipe_scanned = QLabel("\\u2014")
        self.lbl_pipe_dropped = QLabel("\\u2014")
        self.lbl_pipe_faces = QLabel("\\u2014")
        self.lbl_pipe_coverage = QLabel("\\u2014")
        self.lbl_pipe_classified = QLabel("\\u2014")
        self.lbl_pipe_skipped = QLabel("\\u2014")
        fl.addWidget(QLabel("Frames sampled:"), 0, 0)
        fl.addWidget(self.lbl_pipe_sampled, 0, 1)
        fl.addWidget(QLabel("Frames scanned:"), 1, 0)
        fl.addWidget(self.lbl_pipe_scanned, 1, 1)
        fl.addWidget(QLabel("Frames dropped:"), 2, 0)
        fl.addWidget(self.lbl_pipe_dropped, 2, 1)
        fl.addWidget(QLabel("Faces detected:"), 3, 0)
        fl.addWidget(self.lbl_pipe_faces, 3, 1)
        fl.addWidget(QLabel("Face coverage:"), 4, 0)
        fl.addWidget(self.lbl_pipe_coverage, 4, 1)
        fl.addWidget(QLabel("Crops classified:"), 5, 0)
        fl.addWidget(self.lbl_pipe_classified, 5, 1)
        fl.addWidget(QLabel("Frames skipped:"), 6, 0)
        fl.addWidget(self.lbl_pipe_skipped, 6, 1)
        form.addWidget(grp_pipe)
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part3a)
part3b = '''
        grp_emo = QGroupBox("Aggregated Emotion Probabilities")
        emo_gl = QGridLayout(grp_emo)
        self.lbl_top_emotion = QLabel("\\u2014")
        self.lbl_top_emotion.setStyleSheet("font-weight: bold;")
        emo_gl.addWidget(QLabel("Top emotion:"), 0, 0)
        emo_gl.addWidget(self.lbl_top_emotion, 0, 1)
        self._emotion_rows = []
        for i, name in enumerate(EMOTIONS):
            lbl = QLabel("\\u2014")
            emo_gl.addWidget(QLabel(f"{name}:"), i + 1, 0)
            emo_gl.addWidget(lbl, i + 1, 1)
            self._emotion_rows.append(lbl)
        form.addWidget(grp_emo)

        grp_frames = QGroupBox("Frame-Level Predictions")
        fl = QVBoxLayout(grp_frames)
        self.lbl_frame_hint = QLabel(
            "Frame-level predictions appear here when CNN inference succeeds.")
        self.lbl_frame_hint.setWordWrap(True)
        fl.addWidget(self.lbl_frame_hint)
        self.frame_table = QTableWidget(0, 1 + NUM_EMOTIONS)
        headers = ["Timestamp (s)"] + list(EMOTIONS)
        self.frame_table.setHorizontalHeaderLabels(headers)
        self.frame_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.frame_table.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)
        fl.addWidget(self.frame_table)
        form.addWidget(grp_frames)

        grp_time = QGroupBox("Processing Time")
        fl = QGridLayout(grp_time)
        self.lbl_time_model = QLabel("\\u2014")
        self.lbl_time_infer = QLabel("\\u2014")
        self.lbl_time_total = QLabel("\\u2014")
        fl.addWidget(QLabel("Model load:"), 0, 0)
        fl.addWidget(self.lbl_time_model, 0, 1)
        fl.addWidget(QLabel("CNN inference:"), 1, 0)
        fl.addWidget(self.lbl_time_infer, 1, 1)
        fl.addWidget(QLabel("Total processing:"), 2, 0)
        fl.addWidget(self.lbl_time_total, 2, 1)
        form.addWidget(grp_time)

        grp_vbs = QGroupBox("VBS Score")
        fl = QVBoxLayout(grp_vbs)
        self.lbl_vbs = QLabel("\\u2014")
        self.lbl_vbs.setWordWrap(True)
        fl.addWidget(self.lbl_vbs)
        form.addWidget(grp_vbs)

        self.lbl_disclaimer = QLabel()
        self.lbl_disclaimer.setWordWrap(True)
        self.lbl_disclaimer.setStyleSheet(
            "color: #b00020; font-style: italic; font-size: 12px;")
        form.addWidget(self.lbl_disclaimer)

        self.btn_back = QPushButton("Back to Check-in")
        self.btn_back.clicked.connect(self._go_back)
        form.addWidget(self.btn_back)

        scroll.setWidget(container)
        root.addWidget(scroll)
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part3b)
part3c = '''
    def _reset_display(self):
        self.lbl_video.setText(
            f"Video: {self._video_path}" if self._video_path else "No video selected.")

        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)

        self.lbl_val_status.setText("\\u2014")
        self.lbl_val_message.setText("\\u2014")
        self.lbl_val_dims.setText("\\u2014")
        self.lbl_val_duration.setText("\\u2014")

        self.lbl_pipe_sampled.setText("\\u2014")
        self.lbl_pipe_scanned.setText("\\u2014")
        self.lbl_pipe_dropped.setText("\\u2014")
        self.lbl_pipe_faces.setText("\\u2014")
        self.lbl_pipe_coverage.setText("\\u2014")
        self.lbl_pipe_classified.setText("\\u2014")
        self.lbl_pipe_skipped.setText("\\u2014")

        self.lbl_top_emotion.setText("\\u2014")
        for lbl in self._emotion_rows:
            lbl.setText("\\u2014")

        self.lbl_frame_hint.setText(
            "Frame-level predictions appear here when CNN inference succeeds.")
        self.frame_table.setRowCount(0)

        self.lbl_time_model.setText("\\u2014")
        self.lbl_time_infer.setText("\\u2014")
        self.lbl_time_total.setText("\\u2014")

        self.lbl_vbs.setText("\\u2014")

        self.lbl_disclaimer.setText(
            "Emotion predictions are experimental facial-expression outputs "
            "and are NOT a medical diagnosis.")

    @Slot()
    def _cancel_analysis(self):
        if self._worker:
            self._worker.request_cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)

    @Slot()
    def _go_back(self):
        self._cancel_analysis()
        stack = self.parentWidget()
        while stack is not None and not hasattr(stack, 'widget'):
            stack = stack.parentWidget()
        if stack is not None:
            try:
                stack.setCurrentWidget(stack.widget(0))
            except Exception:
                pass

    @Slot(int)
    def _on_progress(self, v: int):
        self.progress.setVisible(True)
        self.progress.setValue(int(v))
        self.btn_cancel.setVisible(True)
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part3c)
part3d = '''
    @Slot(object)
    def _on_done(self, res: AnalysisResult):
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self._thread = None
        self._worker = None

        self.lbl_val_status.setText(
            "Passed" if res.validation_ok else "Failed")
        self.lbl_val_message.setText(res.validation_message or "\\u2014")
        self.lbl_val_dims.setText(
            f"{res.width} x {res.height}" if res.width and res.height else "\\u2014")
        self.lbl_val_duration.setText(
            f"{res.duration_s:.1f}s" if res.duration_s > 0 else "\\u2014")

        self.lbl_pipe_sampled.setText(str(res.sampled_count))
        self.lbl_pipe_scanned.setText(str(res.scanned_count))
        self.lbl_pipe_dropped.setText(str(res.dropped_count))
        self.lbl_pipe_faces.setText(
            f"{res.face_with_face}/{res.face_total}")
        self.lbl_pipe_coverage.setText(f"{res.face_coverage:.0%}")
        self.lbl_pipe_classified.setText(
            str(len(res.emotion_frames)))
        self.lbl_pipe_skipped.setText(str(res.emotion_skipped))

        if res.aggregated_probs is not None:
            self.lbl_top_emotion.setText(
                f"{res.top_emotion} ({res.top_prob:.3f})")
            for lbl, prob in zip(self._emotion_rows, res.aggregated_probs):
                lbl.setText(f"{prob:.4f}")
        else:
            self.lbl_top_emotion.setText("\\u2014")
            for lbl in self._emotion_rows:
                lbl.setText("\\u2014")

        self.frame_table.setRowCount(0)
        if res.emotion_frames:
            self.lbl_frame_hint.setText(
                f"{len(res.emotion_frames)} frame-level predictions below.")
            self.frame_table.setRowCount(len(res.emotion_frames))
            for r, fr in enumerate(res.emotion_frames):
                ts_item = QTableWidgetItem(f"{fr['timestamp_s']:.2f}")
                ts_item.setFlags(ts_item.flags() & ~Qt.ItemIsEditable)
                self.frame_table.setItem(r, 0, ts_item)
                for c, prob in enumerate(fr["probs"]):
                    item = QTableWidgetItem(f"{prob:.4f}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.frame_table.setItem(r, c + 1, item)
        else:
            self.lbl_frame_hint.setText(
                "No frame-level predictions available.")

        self.lbl_time_model.setText(
            f"{res.model_load_time_s:.3f}s" if res.model_load_time_s > 0 else "\\u2014")
        self.lbl_time_infer.setText(
            f"{res.inference_time_s:.3f}s" if res.inference_time_s > 0 else "\\u2014")
        self.lbl_time_total.setText(
            f"{res.total_time_s:.3f}s" if res.total_time_s > 0 else "\\u2014")

        self.lbl_vbs.setText(res.vbs_message or "\\u2014")

        if res.error_message and not res.ok:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Analysis", res.error_message)

        self.btn_back.setEnabled(True)

    def showEvent(self, event):
        super().showEvent(event)
        if self._video_path and self._thread is None and self._worker is None:
            self._start_analysis()

    def _start_analysis(self):
        if not self._video_path:
            return
        self.btn_back.setEnabled(False)
        self._thread = QThread(self)
        self._worker = AnalysisWorker(self._video_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)
        self._worker.done.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
'''

with open(path, "a", encoding="utf-8") as f:
    f.write(part3d)
print("Part 3d written")
print("All parts written. Running script to generate video_analysis_page.py...")
exec(open(path).read().replace('import os\\n', '').replace('path = ', '_path = ').split("print(\"Part 1 written\")")[0].split("part1 = ")[1].rsplit("'''", 1)[0] + "'''")
print("Done")