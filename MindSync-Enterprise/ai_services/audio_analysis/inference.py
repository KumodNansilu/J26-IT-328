
"""Load the trained hybrid DNN and predict speech emotion probabilities."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "speech_emotion_hybrid.pt"
IMPUTER_PATH = MODEL_DIR / "hybrid_imputer.joblib"
SCALER_PATH = MODEL_DIR / "hybrid_scaler.joblib"
LABELS_PATH = MODEL_DIR / "hybrid_emotion_labels.joblib"

DEVICE = torch.device("cpu")

ACOUSTIC_COLUMNS = [
    "mean_internal_pause_seconds",
    "energy_peak_rate_per_min_est",
    "pitch_std_hz",
    "jitter_local",
]

EXPECTED_FEATURES = 1028

_model = None
_imputer = None
_scaler = None
_classes = None
_feature_columns = None


class EmotionDNN(nn.Module):
    """Architecture matching the trained hybrid checkpoint."""

    def __init__(self, input_size: int, number_of_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_trained_model():
    """Load the hybrid model and preprocessing artifacts once."""

    global _model, _imputer, _scaler, _classes, _feature_columns

    if all(
        item is not None
        for item in (
            _model, _imputer, _scaler, _classes, _feature_columns
        )
    ):
        return _model, _imputer, _scaler, _classes, _feature_columns

    required_files = [
        MODEL_PATH, IMPUTER_PATH, SCALER_PATH, LABELS_PATH
    ]
    missing = [str(p) for p in required_files if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing hybrid model files: " + ", ".join(missing)
        )

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    label_data = joblib.load(LABELS_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not isinstance(label_data, dict):
        raise ValueError("Hybrid label artifact must be a dictionary.")

    classes = label_data.get("emotion_classes")
    feature_columns = label_data.get("feature_columns")

    if not classes or not feature_columns:
        raise ValueError(
            "Hybrid label artifact is missing classes or feature_columns."
        )

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} saved feature columns, "
            f"got {len(feature_columns)}."
        )

    expected_columns = ACOUSTIC_COLUMNS + [
        f"xlsr_{i:04d}" for i in range(1024)
    ]
    if feature_columns != expected_columns:
        raise ValueError(
            "Saved hybrid feature order does not match the expected "
            "acoustic-first, XLSR-second order."
        )

    if checkpoint["input_size"] != EXPECTED_FEATURES:
        raise ValueError(
            f"Hybrid checkpoint expects {checkpoint['input_size']} "
            f"features, not {EXPECTED_FEATURES}."
        )

    if imputer.n_features_in_ != EXPECTED_FEATURES:
        raise ValueError("Hybrid imputer has an unexpected feature count.")

    if scaler.n_features_in_ != EXPECTED_FEATURES:
        raise ValueError("Hybrid scaler has an unexpected feature count.")

    model = EmotionDNN(
        input_size=checkpoint["input_size"],
        number_of_classes=len(classes),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _imputer = imputer
    _scaler = scaler
    _classes = classes
    _feature_columns = feature_columns

    return _model, _imputer, _scaler, _classes, _feature_columns


def predict_emotion(embedding, acoustic_features, output_dir=None):
    """Predict emotion from a 1024-D XLSR embedding and four acoustic features."""

    model, imputer, scaler, classes, _ = load_trained_model()

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (1024,):
        raise ValueError(
            f"Expected a 1024-dimensional embedding, got {embedding.shape}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError("XLSR embedding contains invalid numeric values.")

    if not isinstance(acoustic_features, dict):
        raise ValueError("Acoustic features must be supplied as a dictionary.")

    missing = [name for name in ACOUSTIC_COLUMNS if name not in acoustic_features]
    if missing:
        raise ValueError(f"Missing acoustic features: {missing}")

    acoustic = np.array(
        [acoustic_features[name] for name in ACOUSTIC_COLUMNS],
        dtype=np.float32,
    )

    # Missing acoustic measurements are allowed and handled by the
    # trained imputer. Infinite values are rejected.
    if np.isinf(acoustic).any():
        raise ValueError("Acoustic features contain infinite values.")

    # Training order: four acoustic values first, then 1024 XLSR values.
    hybrid_vector = np.concatenate([acoustic, embedding])

    if hybrid_vector.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Expected a {EXPECTED_FEATURES}-dimensional hybrid vector, "
            f"got {hybrid_vector.shape}."
        )

    features = hybrid_vector.reshape(1, -1)
    features = imputer.transform(features)
    features = scaler.transform(features)
    features = np.asarray(features, dtype=np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Hybrid features contain invalid values after preprocessing.")

    input_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    results = sorted(
        [
            (str(emotion), float(probability))
            for emotion, probability in zip(classes, probabilities)
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(
            results, columns=["emotion", "probability"]
        )
        results_df["probability_percent"] = (
            results_df["probability"] * 100
        ).round(2)
        results_df["predicted_class"] = results[0][0]
        results_df.to_csv(
            output_dir / "emotion_probabilities.csv", index=False
        )

    return results
