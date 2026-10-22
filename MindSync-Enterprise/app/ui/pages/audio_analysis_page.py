"""Audio analysis page for the MindSync Enterprise desktop application."""

from pathlib import Path

import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QFrame,
    QProgressBar,
    QGridLayout,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
)


API_URL = "http://127.0.0.1:8000/predict-audio"

SUPPORTED_VIDEO_TYPES = (
    "Video files (*.mp4 *.mov *.avi *.mkv *.webm);;"
    "All files (*.*)"
)


class AnalysisWorker(QThread):
    """Send a video to the FastAPI backend without blocking the UI."""

    analysis_finished = Signal(dict)
    analysis_failed = Signal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            with open(self.video_path, "rb") as video_file:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            Path(self.video_path).name,
                            video_file,
                            "application/octet-stream",
                        )
                    },
                    timeout=180,
                )

            response.raise_for_status()
            result = response.json()

            if result.get("status") != "success":
                raise ValueError(
                    result.get("message", "The analysis was not successful.")
                )

            self.analysis_finished.emit(result)

        except requests.exceptions.ConnectionError:
            self.analysis_failed.emit(
                "Cannot connect to the MindSync backend.\n\n"
                "Please start the FastAPI server first:\n"
                "uvicorn app.main:app --reload"
            )
        except requests.exceptions.Timeout:
            self.analysis_failed.emit(
                "The analysis took too long. Please try again with a shorter video."
            )
        except requests.exceptions.HTTPError as exc:
            message = "The backend returned an error."
            try:
                message = exc.response.json().get("detail", message)
            except (ValueError, AttributeError):
                pass
            self.analysis_failed.emit(str(message))
        except (OSError, ValueError, requests.exceptions.RequestException) as exc:
            self.analysis_failed.emit(str(exc))


class AudioAnalysisPage(QWidget):
    """Video upload, speech emotion prediction and acoustic feature display."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.video_path = None
        self.worker = None
        self.probability_bars = {}
        self.acoustic_value_labels = {}

        self.setObjectName("AudioAnalysisPage")
        self._build_ui()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(28, 24, 28, 24)
        root_layout.setSpacing(18)

        # Scroll area keeps the page usable on smaller displays.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(20)

        # Page heading
        heading = QLabel("Speech Emotion Analysis")
        heading.setObjectName("PageTitle")

        subtitle = QLabel(
            "Analyze a video check-in using speech emotion probabilities "
            "and interpretable acoustic features."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        content_layout.addWidget(heading)
        content_layout.addWidget(subtitle)

        # Upload panel
        upload_card = self._create_card()
        upload_layout = QVBoxLayout(upload_card)
        upload_layout.setContentsMargins(22, 20, 22, 20)
        upload_layout.setSpacing(12)

        upload_title = QLabel("Video Check-in")
        upload_title.setObjectName("SectionTitle")

        upload_description = QLabel(
            "Select a video containing speech. Supported formats: "
            "MP4, MOV, AVI, MKV and WEBM. Maximum upload size: 100 MB."
        )
        upload_description.setWordWrap(True)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)

        self.file_label = QLabel("No video selected")
        self.file_label.setObjectName("FileLabel")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.browse_button = QPushButton("Choose Video")
        self.browse_button.setObjectName("SecondaryButton")
        self.browse_button.clicked.connect(self._choose_video)

        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.browse_button)

        self.analyze_button = QPushButton("Analyze Audio")
        self.analyze_button.setObjectName("PrimaryButton")
        self.analyze_button.setMinimumHeight(42)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._start_analysis)

        self.status_label = QLabel("Select a video to begin.")
        self.status_label.setWordWrap(True)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("AnalysisProgressBar")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.hide()

        upload_layout.addWidget(upload_title)
        upload_layout.addWidget(upload_description)
        upload_layout.addLayout(file_row)
        upload_layout.addWidget(self.analyze_button)
        upload_layout.addWidget(self.progress_bar)
        upload_layout.addWidget(self.status_label)

        content_layout.addWidget(upload_card)

        # Prediction panel
        results_row = QHBoxLayout()
        results_row.setSpacing(18)

        prediction_card = self._create_card()
        prediction_layout = QVBoxLayout(prediction_card)
        prediction_layout.setContentsMargins(22, 20, 22, 20)
        prediction_layout.setSpacing(12)

        prediction_title = QLabel("Emotion Prediction")
        prediction_title.setObjectName("SectionTitle")

        self.predicted_emotion = QLabel("—")
        self.predicted_emotion.setObjectName("PredictedEmotion")
        self.predicted_emotion.setWordWrap(True)

        self.prediction_description = QLabel(
            "The predicted emotion will appear here after analysis."
        )
        self.prediction_description.setWordWrap(True)

        prediction_layout.addWidget(prediction_title)
        prediction_layout.addWidget(self.predicted_emotion)
        prediction_layout.addWidget(self.prediction_description)
        prediction_layout.addStretch()

        probabilities_card = self._create_card()
        probabilities_layout = QVBoxLayout(probabilities_card)
        probabilities_layout.setContentsMargins(22, 20, 22, 20)
        probabilities_layout.setSpacing(12)

        probabilities_title = QLabel("Emotion Probabilities")
        probabilities_title.setObjectName("SectionTitle")
        probabilities_layout.addWidget(probabilities_title)

        emotions = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
        ]

        for emotion in emotions:
            row = QHBoxLayout()
            row.setSpacing(10)

            name_label = QLabel(emotion.capitalize())
            name_label.setMinimumWidth(75)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("")
            bar.setTextVisible(False)
            bar.setMinimumHeight(15)

            value_label = QLabel("—")
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            value_label.setMinimumWidth(55)

            row.addWidget(name_label)
            row.addWidget(bar, 1)
            row.addWidget(value_label)

            probabilities_layout.addLayout(row)
            self.probability_bars[emotion] = (bar, value_label)

        results_row.addWidget(prediction_card, 1)
        results_row.addWidget(probabilities_card, 2)
        content_layout.addLayout(results_row)

        # Acoustic features panel
        acoustic_card = self._create_card()
        acoustic_layout = QVBoxLayout(acoustic_card)
        acoustic_layout.setContentsMargins(22, 20, 22, 20)
        acoustic_layout.setSpacing(14)

        acoustic_title = QLabel("Acoustic Features")
        acoustic_title.setObjectName("SectionTitle")

        acoustic_description = QLabel(
            "Descriptive measurements extracted from the speech signal."
        )
        acoustic_description.setObjectName("PageSubtitle")
        acoustic_description.setWordWrap(True)

        acoustic_grid = QGridLayout()
        acoustic_grid.setHorizontalSpacing(16)
        acoustic_grid.setVerticalSpacing(16)

        feature_names = [
            (
                "mean_internal_pause_seconds",
                "Mean Internal Pause",
                "seconds",
            ),
            (
                "energy_peak_rate_per_min_est",
                "Estimated Energy Peaks",
                "per minute",
            ),
            (
                "pitch_std_hz",
                "Pitch Variation",
                "Hz",
            ),
            (
                "jitter_local",
                "Local Jitter",
                "measurement",
            ),
        ]

        for index, (key, display_name, unit) in enumerate(feature_names):
            feature_card = QFrame()
            feature_card.setObjectName("FeatureCard")

            feature_layout = QVBoxLayout(feature_card)
            feature_layout.setContentsMargins(16, 14, 16, 14)
            feature_layout.setSpacing(8)

            name_label = QLabel(display_name)
            name_label.setObjectName("FeatureName")
            name_label.setWordWrap(True)

            value_label = QLabel("—")
            value_label.setObjectName("FeatureValue")

            unit_label = QLabel(unit)
            unit_label.setObjectName("FeatureUnit")

            feature_layout.addWidget(name_label)
            feature_layout.addWidget(value_label)
            feature_layout.addWidget(unit_label)
            feature_layout.addStretch()

            row_index = index // 2
            column_index = index % 2

            acoustic_grid.addWidget(
                feature_card, row_index, column_index
            )

            self.acoustic_value_labels[key] = value_label

        acoustic_layout.addWidget(acoustic_title)
        acoustic_layout.addWidget(acoustic_description)
        acoustic_layout.addLayout(acoustic_grid)

        content_layout.addWidget(acoustic_card)


        # Responsible-use notice
        self.notice_label = QLabel(
            "Results are model estimates, not a diagnosis or proof of a "
            "person's emotional state. Acoustic features are descriptive "
            "measurements and are not a validated clinical distress score."
        )
        self.notice_label.setObjectName("NoticeLabel")
        self.notice_label.setWordWrap(True)
        content_layout.addWidget(self.notice_label)

        content_layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)

    @staticmethod
    def _create_card():
        card = QFrame()
        card.setObjectName("ContentCard")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        return card

    def _choose_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video check-in",
            "",
            SUPPORTED_VIDEO_TYPES,
        )

        if not file_path:
            return

        if Path(file_path).suffix.lower() not in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm"
        }:
            QMessageBox.warning(
                self,
                "Unsupported file",
                "Please select an MP4, MOV, AVI, MKV or WEBM video.",
            )
            return

        if Path(file_path).stat().st_size > 100 * 1024 * 1024:
            QMessageBox.warning(
                self,
                "File too large",
                "Please select a video smaller than or equal to 100 MB.",
            )
            return

        self.video_path = file_path
        self.file_label.setText(Path(file_path).name)
        self.status_label.setText("Video selected. Ready to analyze.")
        self.analyze_button.setEnabled(True)

    def _start_analysis(self):
        if not self.video_path:
            QMessageBox.information(
                self,
                "Select a video",
                "Please choose a video before starting the analysis.",
            )
            return

        self.analyze_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(
          "Analyzing speech... Please wait while the model processes your video."
        )
        self.status_label.setStyleSheet("color: #2563EB; font-weight: 600;")

        self.worker = AnalysisWorker(self.video_path)
        self.worker.analysis_finished.connect(self._show_results)
        self.worker.analysis_failed.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _show_results(self, result: dict):
        emotion = str(result.get("predicted_class", "Unknown"))
        self.predicted_emotion.setText(emotion.replace("_", " ").title())
        self.prediction_description.setText(
            "The result below represents the model's estimated probabilities "
            "for the supported emotion classes."
        )

        # Populate probability bars.
        for key, (bar, value_label) in self.probability_bars.items():
            bar.setValue(0)
            value_label.setText("—")

        for item in result.get("emotions", []):
            key = str(item.get("emotion", "")).lower()
            if key not in self.probability_bars:
                continue

            probability = item.get("probability_percent", 0)
            try:
                percent = max(0.0, min(100.0, float(probability)))
            except (TypeError, ValueError):
                percent = 0.0

            bar, value_label = self.probability_bars[key]
            bar.setValue(round(percent))
            value_label.setText(f"{percent:.2f}%")

        # Populate the four acoustic features.
        acoustic = result.get("acoustic_features", {})

        for key, value_label in self.acoustic_value_labels.items():
            value = acoustic.get(key)

            if value is None:
                value_label.setText("Unavailable")
                continue

            try:
                value_label.setText(f"{float(value):.4f}")
            except (TypeError, ValueError):
                value_label.setText(str(value))

        notice = result.get("notice")
        if notice:
            self.notice_label.setText(str(notice))

        self.progress_bar.hide()
        self.status_label.setText("Analysis completed successfully.")
        self.status_label.setStyleSheet("color: #15803D; font-weight: 600;")

    def _show_error(self, message: str):
        self.status_label.setText(
            "Analysis failed. Check the message for details and try again."
        )
        self.status_label.setStyleSheet("color: #DC2626; font-weight: 600;")
        QMessageBox.warning(
            self,
            "Audio Analysis Error",
            message,
        )

    def _worker_finished(self):
        self.analyze_button.setEnabled(self.video_path is not None)
        self.browse_button.setEnabled(True)
        self.progress_bar.hide()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None