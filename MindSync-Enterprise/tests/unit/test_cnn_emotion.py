"""Step-4 tests: loader + CNN + inference (synthetic fixtures, no download)."""
import os
import json
import shutil
import tempfile

import pytest

numpy = pytest.importorskip("numpy")
import numpy as np

torch = pytest.importorskip("torch")

from ai_services.video_facial_emotion_behavior.emotion_recognition.fer2013 import (
    FER_CSV_TO_LABELS_INDEX, IMG_PIXELS, load_fer_csv, load_fer_folders,
    lock_test_set, split_train_val)
from ai_services.video_facial_emotion_behavior.emotion_recognition.labels import (
    EMOTIONS, NUM_EMOTIONS)
from ai_services.video_facial_emotion_behavior.emotion_recognition.predict import (
    ModelNotAvailableError, load_model, predict_probs)
from ai_services.video_facial_emotion_behavior.emotion_recognition.cnn_model import (
    MODEL_INPUT_SHAPE, build_model, preprocess_crop)

# Temporary directory for synthetic weights/checkpoints + meta files used
# by every metadata-compatibility test below. Cleaned between tests.
_SYNTH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_services",
                      "video_facial_emotion_behavior",
                      "emotion_recognition", ".synth_nn")


def _csv_text(rows):
    lines = ["emotion,pixels,Usage"]
    for emo, usage, fill in rows:
        lines.append(f"{emo},{' '.join([str(fill)] * IMG_PIXELS)},{usage}")
    return "\n".join(lines) + "\n"


def test_label_map_matches_labels_py():
    assert [FER_CSV_TO_LABELS_INDEX[i] for i in range(7)] == list(range(7))
    assert EMOTIONS == ("Angry", "Disgust", "Fear", "Happy", "Sad",
                        "Surprise", "Neutral")
    assert NUM_EMOTIONS == 7


def test_csv_loader_good_and_bad_rows(tmp_path):
    p = tmp_path / "mini.csv"
    good = _csv_text([(3, "Training", 128), (6, "PrivateTest", 10)])
    bad_label = "9," + " ".join(["0"] * IMG_PIXELS) + ",Training\n"
    bad_pix = "1,1 2 3,Training\n"
    p.write_text(good + bad_label + bad_pix)
    recs, rep = load_fer_csv(str(p))
    assert len(recs) == 2 and rep.kept == 2 and rep.skipped == 2
    assert recs[0].label == 3 and recs[0].image.shape == (48, 48)
    assert len(rep.problems) == 2


def test_csv_bad_header_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_fer_csv(str(p))


def test_folder_loader_and_unknown_folder(tmp_path):
    import cv2

    (tmp_path / "train" / "Happy").mkdir(parents=True)
    (tmp_path / "train" / "Nope").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "train" / "Happy" / "a.png"),
                np.full((60, 60), 200, dtype=np.uint8))
    recs, rep = load_fer_folders(str(tmp_path))
    assert len(recs) == 1 and recs[0].label == EMOTIONS.index("Happy")
    assert recs[0].image.shape == (48, 48)
    assert any("Nope" in m for m in rep.problems)


def test_split_stratified_and_test_locked():
    from ai_services.video_facial_emotion_behavior.emotion_recognition.fer2013 import (
        FerRecord)

    recs = []
    for c in range(3):
        for i in range(10):
            recs.append(FerRecord(
                image=np.zeros((48, 48), dtype=np.uint8), label=c,
                usage="Training", source=f"t{c}-{i}"))
    recs.append(FerRecord(image=np.zeros((48, 48), dtype=np.uint8),
                          label=0, usage="PrivateTest", source="secret"))
    tr, va = split_train_val(recs, val_fraction=0.2, seed=1)
    assert len(tr) + len(va) == 30
    assert {id(r) for r in tr}.isdisjoint({id(r) for r in va})
    assert lock_test_set(recs) == []  # locked by default
    assert len(lock_test_set(recs, allow_test=True)) == 1


def test_cnn_forward_shape_and_preprocess():
    from ai_services.video_facial_emotion_behavior.emotion_recognition.cnn_model import (
        MODEL_INPUT_SHAPE, build_model, preprocess_crop)

    m = build_model()
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 1, 48, 48))
    assert tuple(out.shape) == (2, 7)
    bgr = np.full((90, 70, 3), 150, dtype=np.uint8)
    t = preprocess_crop(bgr)
    assert tuple(t.shape) == MODEL_INPUT_SHAPE
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    with pytest.raises(ValueError):
        preprocess_crop(None)


def test_predict_missing_model_raises():
    from ai_services.video_facial_emotion_behavior.emotion_recognition.classifier import (
        ModelNotAvailableError)
    from ai_services.video_facial_emotion_behavior.emotion_recognition.predict import (
        load_model)

    with pytest.raises(ModelNotAvailableError):
        load_model(str("/nonexistent_dir/no_model.pt"))


def test_emotion_stage_skips_invalid():
    from ai_services.video_facial_emotion_behavior.inference.pipeline import (
        run_emotion_stage)

    class _FF:
        def __init__(self, i, crop):
            self.index, self.timestamp_s, self.crop = i, float(i), crop

    r = run_emotion_stage([], is_cancelled=None)
    assert not r.ok
    # No model loaded -> all frames skipped, honest failure (no fake zeros).

# --- metadata compatibility: training and inference share one path ---


def _clean():
    import shutil

    if os.path.exists(_SYNTH):
        shutil.rmtree(_SYNTH)
    os.makedirs(_SYNTH)


def _tmp_dir():
    import tempfile

    return tempfile.mkdtemp(prefix="fer2013_meta_")


def _write_meta(path, label_order=None, input_shape=None):
    meta = {
        "label_order": label_order if label_order is not None else list(EMOTIONS),
        "input": input_shape if input_shape is not None else list(MODEL_INPUT_SHAPE),
        "normalize": "x/255.0",
        "seed": 42,
        "epochs_run": 1,
        "best_val_loss": 0.5,
        "smoke": False,
        "class_counts": {EMOTIONS[c]: 1 for c in range(NUM_EMOTIONS)},
        "torch": "99.9",
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


def _make_dummy_ckpt(path):
    from ai_services.video_facial_emotion_behavior.emotion_recognition.cnn_model import (
        build_model)

    model = build_model()
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, *MODEL_INPUT_SHAPE[1:]))
    torch.save({"state_dict": {
        k: v.clone() for k, v in model.state_dict().items()
    }}, path)


def _rm(d):
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def _load_from(synth_dir):
    from ai_services.video_facial_emotion_behavior.emotion_recognition import predict as P

    old_ckpt, old_meta = P.DEFAULT_CKPT, P.DEFAULT_META
    P.DEFAULT_CKPT = os.path.join(synth_dir, "w.pt")
    P.DEFAULT_META = os.path.join(synth_dir, "w.json")
    try:
        return load_model(), {"loaded": True,
                              "path": P.DEFAULT_CKPT}
    finally:
        P.DEFAULT_CKPT, P.DEFAULT_META = old_ckpt, old_meta


def test_same_metadata_path_used_by_train_and_infer():
    from ai_services.video_facial_emotion_behavior.emotion_recognition import labels as L

    assert os.path.dirname(L.DEFAULT_META) == os.path.dirname(L.DEFAULT_CKPT)
    assert os.path.basename(L.DEFAULT_CKPT) == "fer2013_cnn_best.pt"
    assert os.path.basename(L.DEFAULT_META) == "fer2013_cnn_meta.json"


def test_correct_meta_is_accepted():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    _write_meta(os.path.join(d, "w.json"))
    try:
        _, info = _load_from(d)
        assert info["loaded"] is True
    finally:
        _rm(d)


def test_wrong_label_order_is_rejected():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    _write_meta(os.path.join(d, "w.json"),
                label_order=["Neutral", "Happy", "Fear", "Sad",
                             "Surprise", "Angry", "Disgust"])
    try:
        with pytest.raises(ModelNotAvailableError) as cm:
            _load_from(d)
        assert "label order" in str(cm.value)
    finally:
        _rm(d)


def test_wrong_input_shape_is_rejected():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    _write_meta(os.path.join(d, "w.json"), input_shape=[1, 64, 64])
    try:
        with pytest.raises(ModelNotAvailableError) as cm:
            _load_from(d)
        assert "input" in str(cm.value)
    finally:
        _rm(d)


def test_missing_meta_fails_clearly():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    try:
        with pytest.raises(ModelNotAvailableError) as cm:
            _load_from(d)
        assert "Metadata file missing" in str(cm.value)
    finally:
        _rm(d)


def test_malformed_meta_fails_clearly():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    with open(os.path.join(d, "w.json"), "w", encoding="utf-8") as fh:
        fh.write("{not valid json!!!")
    try:
        with pytest.raises(ModelNotAvailableError) as cm:
            _load_from(d)
        assert "not valid JSON" in str(cm.value)
    finally:
        _rm(d)


def test_meta_not_a_dict_fails_clearly():
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    with open(os.path.join(d, "w.json"), "w", encoding="utf-8") as fh:
        fh.write("42")
    try:
        with pytest.raises(ModelNotAvailableError) as cm:
            _load_from(d)
        assert "not a JSON object" in str(cm.value)
    finally:
        _rm(d)


def test_inference_still_returns_seven_probs():
    # Existing behaviour (predict_probs -> 7 probs in EMOTIONS order) is
    # untouched; metadata checks run before weights are used.
    _clean()
    d = _tmp_dir()
    ckpt = os.path.join(d, "w.pt")
    _make_dummy_ckpt(ckpt)
    _write_meta(os.path.join(d, "w.json"))
    try:
        _, _ = _load_from(d)
        bgr = np.full((90, 70, 3), 150, dtype=np.uint8)
        probs = predict_probs(bgr)
        assert isinstance(probs, list) and len(probs) == NUM_EMOTIONS
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    finally:
        _rm(d)

