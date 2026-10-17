"""Train the animal classifier with transfer learning (MobileNetV2).

Expected dataset layout (one folder per class, any image names):
    dataset/
      monkey/  dog/  snake/  wild_boar/  other/

Run (Google Colab with a GPU is easiest):
    pip install tensorflow scikit-learn matplotlib
    python training/train.py --data dataset --epochs 8

Outputs:
    model/animal_model.tflite  small model used by the server (commit this to GitHub)
    model/animal_model.keras   full model, for retraining
    model/labels.json          class order
    model/report.txt           accuracy + per-class precision/recall
    model/confusion_matrix.png chart for the pitch deck
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG = (224, 224)
ROOT = Path(__file__).resolve().parent.parent


def main(data_dir: str, epochs: int, fine_tune_epochs: int, batch: int, weights: str):
    out = ROOT / "model"
    out.mkdir(exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42, image_size=IMG, batch_size=batch)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42, image_size=IMG, batch_size=batch)
    labels = train_ds.class_names
    print("Classes:", labels)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])
    base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False,
                                             weights=None if weights == "none" else weights)
    base.trainable = False

    # The model takes raw 0-255 RGB pixels, so the backend needs no extra preprocessing.
    inputs = tf.keras.Input(shape=IMG + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[stop])

    if fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=[stop])

    model.save(out / "animal_model.keras")
    (out / "labels.json").write_text(json.dumps(labels))

    # Small TFLite copy for the server (fits Render's free plan; no TensorFlow needed there).
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    (out / "animal_model.tflite").write_bytes(converter.convert())

    # Evaluation for the pitch deck
    from sklearn.metrics import classification_report, confusion_matrix
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        y_true.extend(yb.numpy())
        y_pred.extend(np.argmax(model.predict(xb, verbose=0), axis=1))
    report = classification_report(y_true, y_pred, target_names=labels, digits=3, zero_division=0)
    print(report)
    (out / "report.txt").write_text(report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Animal classifier: confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    print("Saved model and evaluation to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fine-tune-epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--weights", default="imagenet", help="'none' only for quick pipeline tests")
    a = p.parse_args()
    main(a.data, a.epochs, a.fine_tune_epochs, a.batch, a.weights)
