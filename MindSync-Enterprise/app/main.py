
"""MindSync Enterprise FastAPI entry point."""

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.services.audio_service import analyze_video
from app.database.repositories import save_audio_analysis

app = FastAPI(
    title="MindSync Audio Analysis API",
    description=(
        "Speech emotion estimation and acoustic feature extraction "
        "for the MindSync research prototype."
    ),
    version="1.0.0",
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


@app.get("/")
def root():
    """Return basic API information."""
    return {
        "service": "MindSync Audio Analysis API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Check that the API process is running."""
    return {
        "status": "healthy",
        "service": "audio_analysis",
    }


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """Accept a video upload and return speech analysis results."""

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. Upload a video in "
                "MP4, MOV, AVI, MKV, or WEBM format."
            ),
        )

    temporary_path = None

    try:
        # Save upload to a temporary location, with a size limit.
        with tempfile.NamedTemporaryFile(
            prefix="mindsync_upload_",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            total_bytes = 0

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the 100 MB limit.",
                    )

                temporary_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Run speech analysis.
        result = analyze_video(temporary_path)

        # Remove temporary file paths from the API response.
        result.pop("processed_audio", None)
        result.pop("probabilities_csv", None)

        # Include the original uploaded filename.
        result["filename"] = filename

        # Save the analysis result to MongoDB.
        try:
            analysis_id = save_audio_analysis(result)
            result["analysis_id"] = analysis_id
            result["saved_to_database"] = True

        except Exception as exc:
            logger.exception(
                "Failed to save audio analysis to MongoDB"
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analysis completed, but saving the result "
                    "to the database failed."
                ),
            ) from exc

        return result

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Audio analysis failed. Check the server logs "
                "for details."
            ),
        ) from exc

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
