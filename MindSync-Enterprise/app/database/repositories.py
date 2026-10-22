"""Database repositories for MindSync."""

from datetime import datetime, timezone
from typing import Any

from app.database.connection import db


def save_audio_analysis(
    prediction: dict[str, Any],
    employee_id: str | None = None,
) -> str:
    """Save an audio analysis result to MongoDB."""

    collection = db["audio_analysis_results"]

    emotions = prediction.get("emotions", [])
    probabilities = {
        item["emotion"]: item["probability"]
        for item in emotions
        if "emotion" in item and "probability" in item
    }

    document = {
        "employee_id": employee_id,
        "analysis_type": "audio_emotion",
        "filename": prediction.get("filename"),
        "predicted_emotion": prediction.get("predicted_class"),
        "probabilities": probabilities,
        "emotions": emotions,
        "acoustic_features": prediction.get("acoustic_features", {}),
        "notice": prediction.get("notice"),
        "created_at": datetime.now(timezone.utc),
        "status": prediction.get("status", "completed"),
    }

    result = collection.insert_one(document)

    return str(result.inserted_id)
