
"""Extract interpretable acoustic features from preprocessed audio."""

import numpy as np
import librosa
import parselmouth


SAMPLE_RATE = 16000


def extract_acoustic_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, float]:
    """Extract four acoustic features from a mono audio signal.

    The input audio should already be preprocessed and mono.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty mono signal.")

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numeric values.")

    duration = len(audio) / sample_rate

    if duration <= 0:
        raise ValueError("Audio duration must be positive.")

    # 1. Estimate speech regions and internal pauses.
    intervals = librosa.effects.split(
        audio,
        top_db=30,
        frame_length=2048,
        hop_length=512,
    )

    speech_regions = [
        (int(start), int(end))
        for start, end in intervals
        if end > start
    ]

    pauses = []

    for index in range(1, len(speech_regions)):
        previous_end = speech_regions[index - 1][1]
        next_start = speech_regions[index][0]

        pause_seconds = (next_start - previous_end) / sample_rate

        if pause_seconds > 0:
            pauses.append(pause_seconds)

    mean_pause = (
        float(np.mean(pauses))
        if pauses
        else 0.0
    )

    # 2. Estimate energy peaks per minute.
    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if rms.size > 0:
        peak_threshold = float(np.mean(rms) + np.std(rms))
        peak_indices = np.where(
            (rms[1:-1] > rms[:-2])
            & (rms[1:-1] >= rms[2:])
            & (rms[1:-1] > peak_threshold)
        )[0] + 1

        peak_rate = float(len(peak_indices) / duration * 60.0)
    else:
        peak_rate = 0.0

    # 3. Estimate pitch standard deviation and 4. local jitter.
    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sample_rate,
    )

    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    voiced_pitch = pitch_values[pitch_values > 0]

    pitch_std = (
        float(np.std(voiced_pitch))
        if voiced_pitch.size > 1
        else 0.0
    )

    point_process = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        500,
    )

    try:
        jitter_local = float(
            parselmouth.praat.call(
                point_process,
                "Get jitter (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
            )
        )
    except Exception:
        jitter_local = 0.0

    if not np.isfinite(jitter_local):
        jitter_local = 0.0

    return {
        "mean_internal_pause_seconds": mean_pause,
        "energy_peak_rate_per_min_est": peak_rate,
        "pitch_std_hz": pitch_std,
        "jitter_local": jitter_local,
    }

