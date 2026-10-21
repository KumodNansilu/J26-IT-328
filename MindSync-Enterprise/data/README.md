# Data

> **Never commit real employee data here.** Only anonymized sample data and
> public, licensed datasets belong in this directory.

# FER2013 Dataset

Status: **downloaded and verified** (see `docs/fer2013_dataset_report.json`).

## Provenance

| Field | Value |
|---|---|
| Source (approved) | <https://www.kaggle.com/datasets/deadskull7/fer2013/data> |
| Kaggle handle | `deadskull7/fer2013` |
| Downloaded via | KaggleHub (`kagglehub` 1.0.2) — official Kaggle library |
| Licence | **CC0: Public Domain** (Kaggle dataset API) |
| Dataset last updated | 2018-05-26 |
| Local path | `data/fer2013/fer2013.csv` (git-ignored) |
| Size | 301,072,766 bytes |
| SHA-256 | `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b` |

The file size matches the `totalBytes` reported by the Kaggle dataset API
(301,072,766), confirming a complete, uncorrupted download.

## Structure (verified, not assumed)

- Header: `emotion,pixels,Usage` — matches `fer2013.load_fer_csv` exactly.
- 35,887 rows total, **0** invalid/missing rows, **0** skipped by the loader.
- Every row has exactly **2,304** integer pixel values (48×48 grayscale).
- Partitions: `Training` 28,709 · `PublicTest` 3,589 · `PrivateTest` 3,589.
- Labels are integers `0–6`, all seven present.

## Label order

The dataset's numeric labels map **identity** to the canonical order in
`emotion_recognition/labels.py` (`FER_CSV_TO_LABELS_INDEX = {i: i}`):

| Index | Emotion | Rows |
|---|---|---|
| 0 | Angry | 4,953 |
| 1 | Disgust | 547 |
| 2 | Fear | 5,121 |
| 3 | Happy | 8,989 |
| 4 | Sad | 6,077 |
| 5 | Surprise | 4,002 |
| 6 | Neutral | 6,198 |

Class counts match the canonical FER2013 distribution (Disgust is the
smallest class at 547 rows, Happy the largest at 8,989), which independently
confirms the label ordering is not shifted.

## Duplicate pixels

1,853 rows share an identical 2,304-pixel payload with an earlier row
(34,034 unique images). This is a known property of the FER2013 CSV and is
**left as-is** — the dataset is not modified. Note it when interpreting
metrics; the project's stratified split keeps duplicates in the same fold.

## Reproducing

```bash
python scripts/download_fer2013.py --verify
```

`scripts/download_fer2013.py` downloads only the approved handle into
`data/fer2013/`; `scripts/verify_fer2013.py` is read-only and reports the
structure, partitions, label mapping, and quality facts above.

## Attribution

FER2013 was introduced in *"Challenges in Representation Learning for Facial
Emotion Recognition"* by Goodfellow et al. (2013). The Kaggle copy is
distributed under CC0 (public domain). Attribute the dataset to the original
authors when publishing results.

