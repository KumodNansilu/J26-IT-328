
"""Application service for speech emotion analysis."""

from pathlib import Path
from uuid import uuid4

from ai_services.audio_analysis.preprocessing import (
    extract_audio,
    preprocess_audio,
)
from ai_services.audio_analysis.acoustic_features import (
    extract_acoustic_features,
)
from ai_services.audio_analysis.speech_model import (
    extract_xlsr_embedding,
)
from ai_services.audio_analysis.inference import (
    predict_emotion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "prediction_output" / "audio_analysis"


def analyze_video(video_path: str | Path) -> dict:
    """Analyze speech emotion from a video using the hybrid DNN."""

    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    request_dir = OUTPUT_ROOT / str(uuid4())
    request_dir.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = request_dir / "extracted_audio.wav"
    processed_audio_path = request_dir / "processed_audio.wav"

    # 1. Extract audio from the video
    extract_audio(
        video_path,
        extracted_audio_path,
    )

    # 2. Preprocess audio: mono, 16 kHz, normalized
    audio = preprocess_audio(
        extracted_audio_path,
        processed_audio_path,
    )

    # 3. Extract the four acoustic features
    acoustic_features = extract_acoustic_features(
        audio,
        sample_rate=16000,
    )

    # 4. Extract the 1024-dimensional XLSR embedding
    embedding = extract_xlsr_embedding(audio)

    # 5. Predict emotion probabilities using the trained hybrid DNN
    predictions = predict_emotion(
        embedding=embedding,
        acoustic_features=acoustic_features,
        output_dir=request_dir,
    )

    # 6. Format emotion probabilities
    emotions = [
        {
            "emotion": emotion,
            "probability": float(probability),
            "probability_percent": round(
                float(probability) * 100,
                2,
            ),
        }
        for emotion, probability in predictions
    ]

    if not emotions:
        raise RuntimeError(
            "The hybrid model did not return any predictions."
        )

    # 7. Return results for API or application integration
    return {
        "status": "success",
        "filename": video_path.name,
        "predicted_class": emotions[0]["emotion"],
        "emotions": emotions,
        "acoustic_features": acoustic_features,
        "processed_audio": str(processed_audio_path),
        "probabilities_csv": str(
            request_dir / "emotion_probabilities.csv"
        ),
        "notice": (
            "Emotion probabilities are model estimates, not a "
            "diagnosis or proof of a person's emotional state. "
            "Acoustic features are descriptive measurements and "
            "are not a validated clinical distress score."
        ),
    }
