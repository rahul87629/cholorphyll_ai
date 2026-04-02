"""
predictor.py — Fixed CSV saving + ensemble prediction (CNN + Ridge corrector)
"""

import os
import datetime
import numpy as np
import pandas as pd
import cv2
import joblib
import tensorflow as tf

from color_features import extract_color_features, preprocess_image_for_cnn

MODEL_PATH   = os.path.join("saved_model", "hybrid_cnn_final.keras")
SCALER_PATH  = os.path.join("saved_model", "scaler.pkl")
RIDGE_PATH   = os.path.join("saved_model", "ridge_corrector.pkl")
DATASET_IMG  = "rice_images"
DATASET_CSV  = "rice_labels.csv"

_model  = None
_scaler = None
_ridge  = None


def load_model_and_scaler():
    global _model, _scaler, _ridge
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found. Run: python train.py")
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _ridge is None and os.path.exists(RIDGE_PATH):
        _ridge = joblib.load(RIDGE_PATH)
    return _model, _scaler


def predict_spad(image_bgr: np.ndarray) -> dict:
    model, scaler = load_model_and_scaler()

    img_input    = preprocess_image_for_cnn(image_bgr)
    color_vec    = extract_color_features(image_bgr)
    color_scaled = scaler.transform(color_vec.reshape(1, -1))

    # CNN prediction
    cnn_pred = float(model.predict([img_input, color_scaled], verbose=0)[0, 0])

    # Ridge corrector (if available)
    if _ridge is not None:
        correction = float(_ridge.predict(color_vec.reshape(1, -1))[0])
        spad_val   = float(np.clip(cnn_pred + 0.5 * correction, 0.0, 80.0))
    else:
        spad_val = float(np.clip(cnn_pred, 0.0, 80.0))

    spad_val = round(spad_val, 2)

    if spad_val < 15:
        note = "Critical — severe nitrogen deficiency. Apply fertilizer immediately."
    elif spad_val < 25:
        note = "Poor — low chlorophyll, nitrogen stress detected."
    elif spad_val < 35:
        note = "Moderate — sub-optimal. Light nitrogen topdress recommended."
    elif spad_val < 45:
        note = "Good — healthy chlorophyll. Continue current schedule."
    else:
        note = "Excellent — high chlorophyll. No action needed."

    feat_names = ["Norm-R", "Norm-G", "Norm-B", "Hue", "Saturation", "Value", "DGCI"]
    color_dict = {n: round(float(v), 4) for n, v in zip(feat_names, color_vec)}

    return {"spad_predicted": spad_val, "color_features": color_dict, "confidence_note": note}


def save_new_sample(image_bgr: np.ndarray, original_filename: str, spad_value: float) -> str:
    """Save image + append row to CSV using EXACT same column names as original file."""
    os.makedirs(DATASET_IMG, exist_ok=True)

    ext      = os.path.splitext(original_filename)[1] or ".jpg"
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"leaf_{ts}{ext}"
    cv2.imwrite(os.path.join(DATASET_IMG, new_name), image_bgr)

    if os.path.exists(DATASET_CSV):
        existing = pd.read_csv(DATASET_CSV)
        img_col  = next((c for c in existing.columns
                         if any(k in c.lower() for k in ["image","file","name"])), "image_filename")
        spad_col = next((c for c in existing.columns
                         if any(k in c.lower() for k in ["spad","value","label"])), "spad_value")
        new_row  = pd.DataFrame({img_col: [new_name], spad_col: [round(spad_value, 2)]})
        updated  = pd.concat([existing[[img_col, spad_col]], new_row], ignore_index=True)
    else:
        updated = pd.DataFrame({"image_filename": [new_name], "spad_value": [round(spad_value, 2)]})

    updated.to_csv(DATASET_CSV, index=False)
    return new_name


def get_dataset_stats() -> dict:
    if not os.path.exists(DATASET_CSV):
        return {}
    try:
        df = pd.read_csv(DATASET_CSV)
    except Exception:
        return {}
    spad_col = next((c for c in df.columns if "spad" in c.lower()), None)
    if not spad_col:
        return {"total": len(df)}
    vals = pd.to_numeric(df[spad_col], errors="coerce").dropna()
    return {
        "total": len(df),
        "mean":  round(float(vals.mean()), 2),
        "min":   round(float(vals.min()),  2),
        "max":   round(float(vals.max()),  2),
        "std":   round(float(vals.std()),  2),
    }