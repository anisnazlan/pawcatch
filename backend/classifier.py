"""Animal image classifier.

Uses the model trained by training/train.py, preferring the small TFLite version
(model/animal_model.tflite) so it fits on Render's free server. Falls back to the
full Keras model (model/animal_model.keras) if TensorFlow is installed.
With no model, it runs in demo mode and the resident picks the animal manually.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
TFLITE_PATH = MODEL_DIR / "animal_model.tflite"
KERAS_PATH = MODEL_DIR / "animal_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

_predict = None
_labels = None
_tried = False


def _load_tflite():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def predict(x):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])[0]
    return predict


def _load_keras():
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_PATH)
    return lambda x: model.predict(x, verbose=0)[0]


def _load():
    global _predict, _labels, _tried
    if _tried:
        return
    _tried = True
    if not LABELS_PATH.exists():
        return
    try:
        if TFLITE_PATH.exists():
            _predict = _load_tflite()
        elif KERAS_PATH.exists():
            _predict = _load_keras()
        else:
            return
        _labels = json.loads(LABELS_PATH.read_text())
        print(f"[classifier] model loaded, classes: {_labels}")
    except Exception as e:
        print(f"[classifier] could not load model, running in demo mode: {e}")
        _predict = None


def model_ready() -> bool:
    _load()
    return _predict is not None


def classify(image_path: str) -> dict:
    """Returns {"animal": str|None, "confidence": float, "confident": bool, "mode": str}."""
    if not model_ready():
        return {"animal": None, "confidence": 0.0, "confident": False, "mode": "demo"}

    import numpy as np
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB").resize((224, 224))
    x = np.expand_dims(np.asarray(img, dtype="float32"), 0)
    probs = _predict(x)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return {
        "animal": _labels[idx],
        "confidence": round(conf, 3),
        "confident": conf >= CONFIDENCE_THRESHOLD,
        "mode": "model",
    }
