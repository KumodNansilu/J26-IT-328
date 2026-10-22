
"""Audio extraction and preprocessing for MindSync speech analysis."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip

SAMPLE_RATE = 16000

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def extract_audio(video_path: str | Path, output_wav: str | Path) -> Path:
    """Extract mono audio from a video and save it at 16 kHz."""

    video_path = Path(video_path)
    output_wav = Path(output_wav)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if video_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video format.")

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            raise ValueError("The selected video does not contain audio.")

        video.audio.write_audiofile(
            str(output_wav),
            fps=SAMPLE_RATE,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )

    return output_wav


def preprocess_audio(
    input_wav: str | Path,
    output_wav: str | Path,
) -> np.ndarray:
    """Convert audio to mono, 16 kHz, finite and peak-normalized samples."""

    input_wav = Path(input_wav)
    output_wav = Path(output_wav)

    if not input_wav.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_wav}")

    audio, sample_rate = sf.read(input_wav, dtype="float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio is empty or has an unsupported shape.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numeric values.")

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio is empty or invalid after resampling.")

    peak = float(np.max(np.abs(audio)))

    if peak < 1e-6:
        raise ValueError("Extracted audio is silent.")

    audio = audio / peak

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, audio, SAMPLE_RATE)

    return audio