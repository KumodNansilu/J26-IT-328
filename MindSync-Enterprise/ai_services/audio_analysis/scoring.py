
"""Format speech emotion predictions for MindSync integration."""

from typing import Any


def format_emotion_results(predictions: list[tuple[str, float]]) -> dict[str, Any]:
    """Format emotion probabilities without inferring clinical distress."""

    if not predictions:
        raise ValueError("No emotion predictions were provided.")

    formatted = []

    for emotion, probability in predictions:
        probability = float(probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Invalid probability for {emotion}: {probability}"
            )

        formatted.append(
            {
                "emotion": str(emotion),
                "probability": probability,
                "probability_percent": round(probability * 100, 2),
            }
        )

    formatted.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "predicted_class": formatted[0]["emotion"],
        "emotions": formatted,
        "notice": (
            "These are model-estimated emotion probabilities. "
            "They are not a diagnosis or a validated distress score."
        ),
    }
