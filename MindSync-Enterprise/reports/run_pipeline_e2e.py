"""
E2E video-analysis pipeline runner for MindSync Enterprise.

Uses ONLY the existing public API (no second pipeline, no mocks):
  - video_loader.validate_video_file        -> validation
  - preprocessing.preprocess_video          -> frame sampling
  - landmarks.run_face_stage                -> face detection + crops
  - emotion_recognition.predict.load_model  -> CNN load
  - emotion_recognition.predict.predict_probs -> per-crop inference
  - emotion_recognition.classifier.aggregate_frame_probs -> video-level aggregation
  - scoring.vbs.compute_vbs                 -> VBS boundary (expected to raise)

Creates a synthetic test video from FER2013 Training images in %TEMP%
(outside project source dirs), runs the real trained CNN, writes a JSON
report to reports/, prints a human-readable summary, and cleans up temp
files in a finally block.

Emotion label order (single source of truth from labels.py / metadata):
  Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_services.video_facial_emotion_behavior.preprocessing.video_loader import (
    validate_video_file,
)
from ai_services.video_facial_emotion_behavior.preprocessing.frames import (
    preprocess_video,
)
from ai_services.video_facial_emotion_behavior.landmarks import run_face_stage
from ai_services.video_facial_emotion_behavior.emotion_recognition.predict import (
    load_model as load_cnn,
    predict_probs as cnn_predict_probs,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.classifier import (
    aggregate_frame_probs,
)
from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (
    EMOTIONS,
    DEFAULT_CKPT,
    DEFAULT_META,
)
from ai_services.video_facial_emotion_behavior.scoring.vbs import (
    compute_vbs,
    VBSSpecMissingError,
)
from app.config import (
    FRAME_SAMPLE_FPS,
    FRAME_SAMPLE_MAX_FRAMES,
    MIN_VALID_FRAMES,
    MIN_USABLE_FRAMES,
    MIN_FACE_COVERAGE,
    FACE_MIN_SIZE_PX,
    FACE_CROP_PADDING,
    EXPECTED_LANDMARK_COUNT,
)

EMOTION_LABELS = list(EMOTIONS)
FER2013_CSV = os.path.join(PROJECT_ROOT, "data", "fer2013", "fer2013.csv")
MODEL_CKPT = DEFAULT_CKPT
MODEL_META = DEFAULT_META

TEMP_DIR = os.environ.get("TEMP", os.path.expanduser("~"))
SYNTH_VIDEO = os.path.join(TEMP_DIR, "j26_e2e_synthetic_test.mp4")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "j26_e2e_report.json")

VIDEO_DURATION_S = 30.0
VIDEO_FPS = 2.0
FRAME_W, FRAME_H = 480, 480
FACE_SIZE = 200
TOTAL_FRAMES = int(VIDEO_DURATION_S * VIDEO_FPS)


def build_synthetic_video() -> str:
    """
    Build a synthetic mp4 video from FER2013 Training images.

    Reads the Training split from fer2013.csv, upscales each 48x48 face to
    200x200 RGB, places it on a plain background, and writes ~60 frames
    at 2 fps for 30 seconds. No identifiable footage — only synthetic
    renderings of dataset training examples.
    """
    import cv2
    import numpy as np

    print(f"[video] Building synthetic video: {SYNTH_VIDEO}")
    training_rows = []
    with open(FER2013_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("Usage", "").strip() == "Training":
                training_rows.append(row)

    if not training_rows:
        raise RuntimeError("No Training rows found in FER2013 CSV.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SYNTH_VIDEO, fourcc, VIDEO_FPS, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open VideoWriter (check codec).")

    rng = np.random.default_rng(42)
    chosen = rng.choice(training_rows, size=TOTAL_FRAMES, replace=True)

    for row in chosen:
        pixels = [int(v) for v in row["pixels"].split()]
        face = np.asarray(pixels, dtype=np.uint8).reshape(48, 48)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        face_big = cv2.resize(face_rgb, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_NEAREST)

        frame = np.full((FRAME_H, FRAME_W, 3), 200, dtype=np.uint8)
        x0 = (FRAME_W - FACE_SIZE) // 2
        y0 = (FRAME_H - FACE_SIZE) // 2
        frame[y0:y0 + FACE_SIZE, x0:x0 + FACE_SIZE] = face_big
        writer.write(frame)

    writer.release()
    print(f"[video] Wrote {TOTAL_FRAMES} frames ({VIDEO_DURATION_S}s @ {VIDEO_FPS} fps).")
    return SYNTH_VIDEO


def format_probs(probs: list[float]) -> str:
    """Human-readable 7-emotion probability line."""
    parts = [f"{EMOTION_LABELS[i]}={p:.4f}" for i, p in enumerate(probs)]
    return ", ".join(parts)


def main() -> int:
    t_total_start = time.perf_counter()

    try:
        SYNTH_VIDEO = build_synthetic_video()
        report: dict = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": PROJECT_ROOT,
        "model_checkpoint": MODEL_CKPT,
        "model_metadata": MODEL_META,
        "emotion_label_order": EMOTION_LABELS,
        "input_video": SYNTH_VIDEO,
        "synthetic_note": "Synthetic FER2013 Training renders; not identifiable footage.",
    }

        print("\n=== Step 1: Video validation ===")
        vr = validate_video_file(
            SYNTH_VIDEO,
            allowed_extensions=(".mp4", ".avi", ".mov"),
            min_duration_s=25.0,
            max_duration_s=35.0,
        )
        print(f"  ok={vr.ok}")
        print(f"  errors={vr.errors}")
        if vr.props:
            print(f"  props: width={vr.props.width}, height={vr.props.height}, "
                  f"fps={vr.props.fps:.2f}, frame_count={vr.props.frame_count}, "
                  f"duration_s={vr.props.duration_s:.3f}, readable={vr.props.readable}")
        report["validation"] = {
            "ok": vr.ok,
            "errors": vr.errors,
            "message": vr.message(),
            "props": {
                "path": vr.props.path,
                "width": vr.props.width,
                "height": vr.props.height,
                "fps": vr.props.fps,
                "frame_count": vr.props.frame_count,
                "duration_s": vr.props.duration_s,
                "readable": vr.props.readable,
            } if vr.props else None,
        }
        if not vr.ok:
            print("  -> Validation failed; aborting.")
            report["aborted_reason"] = "validation_failed"
            return 1

        print("\n=== Step 2: Frame sampling ===")
        pre = preprocess_video(
            SYNTH_VIDEO,
            sample_fps=FRAME_SAMPLE_FPS,
            max_frames=FRAME_SAMPLE_MAX_FRAMES,
            min_valid_frames=MIN_VALID_FRAMES,
        )
        print(f"  ok={pre.ok}")
        print(f"  sampled={pre.sampled_count}, scanned={pre.scanned_count}, dropped={pre.dropped_count}")
        print(f"  errors={pre.errors}")
        print(f"  src_fps={pre.src_fps:.2f}, duration_s={pre.duration_s:.3f}")
        report["preprocessing"] = {
            "ok": pre.ok,
            "sampled_count": pre.sampled_count,
            "scanned_count": pre.scanned_count,
            "dropped_count": pre.dropped_count,
            "src_fps": pre.src_fps,
            "duration_s": pre.duration_s,
            "errors": pre.errors,
            "message": pre.message(),
        }
        if not pre.ok:
            print("  -> Preprocessing failed; aborting.")
            report["aborted_reason"] = "preprocessing_failed"
            return 1

        sampled_frames = pre.frames

        print("\n=== Step 3: Face detection (largest-box policy) ===")
        face_res = run_face_stage(
            sampled_frames,
            min_size=FACE_MIN_SIZE_PX,
            padding=FACE_CROP_PADDING,
            min_usable=MIN_USABLE_FRAMES,
            min_coverage=MIN_FACE_COVERAGE,
            expected_landmarks=EXPECTED_LANDMARK_COUNT,
        )
        print(f"  ok={face_res.ok}")
        print(f"  total={face_res.total}, with_face={face_res.with_face}, "
              f"without_face={face_res.without_face}")
        print(f"  coverage={face_res.coverage:.2%}")
        print(f"  errors={face_res.errors}")
        print(f"  warnings={face_res.warnings}")
        report["face_stage"] = {
            "ok": face_res.ok,
            "total": face_res.total,
            "with_face": face_res.with_face,
            "without_face": face_res.without_face,
            "coverage": face_res.coverage,
            "errors": face_res.errors,
            "warnings": face_res.warnings,
            "message": face_res.message(),
        }

        face_frames = list(face_res.usable)
        cnn_inference_start = time.perf_counter()
        frame_results: list[dict] = []

        if face_res.ok:
            print("\n=== Step 4: CNN emotion prediction ===")
            t_cnn_load_start = time.perf_counter()
            abs_ckpt = os.path.join(PROJECT_ROOT, MODEL_CKPT)
            status = load_cnn(abs_ckpt)
            t_cnn_load = time.perf_counter() - t_cnn_load_start
            print(f"  model loaded: {status}")
            report["cnn"] = {
                "model_status": status,
                "checkpoint": MODEL_CKPT,
                "metadata": MODEL_META,
                "load_time_s": round(t_cnn_load, 4),
                "frame_results": [],
            }

            for ff in face_frames:
                try:
                    probs = cnn_predict_probs(ff.crop)
                    frame_results.append({
                        "index": ff.index,
                        "timestamp_s": round(ff.timestamp_s, 4),
                        "probs": probs,
                        "top_emotion": EMOTION_LABELS[int(max(range(len(probs)), key=lambda i: probs[i]))],
                        "top_prob": round(float(max(probs)), 4),
                    })
                except Exception as exc:
                    print(f"  frame {ff.index}: inference failed: {exc}")
                    frame_results.append({
                        "index": ff.index,
                        "timestamp_s": round(ff.timestamp_s, 4),
                        "error": str(exc),
                    })
            report["cnn"]["frame_results"] = frame_results
            t_cnn_inference = time.perf_counter() - cnn_inference_start

            valid_frame_results = [r for r in frame_results if "probs" in r]
            print(f"  classified_crops={len(valid_frame_results)} "
                  f"(skipped={face_res.with_face - len(valid_frame_results)})")
            print(f"  CNN inference time: {t_cnn_inference:.4f}s")
            report["cnn"]["classified_crops"] = len(valid_frame_results)
            report["cnn"]["skipped_crops"] = face_res.with_face - len(valid_frame_results)
            report["cnn"]["inference_time_s"] = round(t_cnn_inference, 4)

            print("\n  Representative frame-level probabilities:")
            show = valid_frame_results[:3] + valid_frame_results[-3:] if len(valid_frame_results) > 6 else valid_frame_results
            shown = set()
            for r in show:
                if r["index"] in shown:
                    continue
                shown.add(r["index"])
                print(f"    frame {r['index']} (@{r['timestamp_s']}s): {format_probs(r['probs'])} "
                      f"| top={r['top_emotion']} ({r['top_prob']})")

            print("\n=== Step 5: Video-level probability aggregation ===")
            all_probs = [r["probs"] for r in valid_frame_results]
            try:
                agg = aggregate_frame_probs(all_probs)
                top_idx = int(max(range(len(agg)), key=lambda i: agg[i]))
                print(f"  aggregated probs: {format_probs(agg)}")
                print(f"  top emotion: {EMOTION_LABELS[top_idx]} ({agg[top_idx]:.4f}) "
                      f"(MODEL PREDICTION — not a mental-health conclusion)")
                report["aggregation"] = {
                    "ok": True,
                    "n_frames": len(all_probs),
                    "probs": agg,
                    "top_emotion": EMOTION_LABELS[top_idx],
                    "top_prob": round(float(agg[top_idx]), 4),
                    "note": "Mean of per-frame softmax distributions; model prediction only.",
                }
            except ValueError as exc:
                print(f"  aggregation failed: {exc}")
                report["aggregation"] = {"ok": False, "error": str(exc)}
        else:
            print("  -> Face stage failed; skipping CNN inference and aggregation.")
            report["cnn"] = {"skipped": "face_stage_failed"}
            report["aggregation"] = {"skipped": "face_stage_failed"}

        print("\n=== Step 6: VBS integration boundary ===")
        agg_probs = report.get("aggregation", {}).get("probs")
        try:
            vbs_result = compute_vbs(agg_probs)
            print(f"  VBS computed: {vbs_result}")
            report["vbs"] = {"computed": True, "result": vbs_result}
        except VBSSpecMissingError as exc:
            print(f"  VBSSpecMissingError: {exc}")
            report["vbs"] = {
                "computed": False,
                "error": "VBSSpecMissingError",
                "message": str(exc),
                "note": "VBS scoring formula not approved; compute_vbs always raises.",
            }
        except Exception as exc:
            print(f"  unexpected VBS error: {exc}")
            report["vbs"] = {"computed": False, "error": type(exc).__name__, "message": str(exc)}

    finally:
        try:
            from ai_services.video_facial_emotion_behavior.landmarks import close as mesh_close
            mesh_close()
        except Exception:
            pass
        try:
            if 'face_res' in locals():
                face_res.clear()
            if 'pre' in locals() and hasattr(pre, 'frames'):
                pre.frames.clear()
        except Exception:
            pass

        try:
            if os.path.exists(SYNTH_VIDEO):
                os.remove(SYNTH_VIDEO)
                print(f"\n[cleanup] Removed temp video: {SYNTH_VIDEO}")
        except Exception as exc:
            print(f"\n[cleanup] Could not remove temp video: {exc}")

    t_total = time.perf_counter() - t_total_start
    report["timing"] = {
        "total_time_s": round(t_total, 4),
    }
    print(f"\n[total] Total pipeline time: {t_total:.4f}s")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[report] JSON report saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())