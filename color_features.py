"""
color_features.py
Extracts all colour-based spectral features from a leaf image.
Features: Normalized RGB (r, g, b), HSV (H, S, V), DGCI
These match the formulas in your project report (Chapter 2 & 3).
"""

import cv2
import numpy as np


def extract_color_features(image_bgr: np.ndarray) -> np.ndarray:
    """
    Given a BGR image (as read by OpenCV), returns a 1-D feature vector:
    [r, g, b, H, S, V, DGCI]  —  7 features total.

    Parameters
    ----------
    image_bgr : np.ndarray
        Image array in BGR format (H x W x 3), uint8.

    Returns
    -------
    np.ndarray  shape (7,)
    """
    # ── Convert to RGB for channel math ──────────────────────────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Mask out non-leaf pixels (very dark or pure white background)
    mask = _get_leaf_mask(image_bgr)
    pixels = image_rgb[mask > 0]          # shape (N, 3)

    if pixels.shape[0] == 0:
        # Fallback: use the whole image
        pixels = image_rgb.reshape(-1, 3)

    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]

    # ── 1. Normalised RGB ────────────────────────────────────────────────────
    total = R + G + B + 1e-8              # avoid ÷0
    r = float(np.mean(R / total))
    g = float(np.mean(G / total))
    b = float(np.mean(B / total))

    # ── 2. HSV features ──────────────────────────────────────────────────────
    # Convert masked pixels to HSV
    masked_rgb = pixels.astype(np.uint8).reshape(-1, 1, 3)
    masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
    hsv_pixels = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    H_raw = hsv_pixels[:, 0]             # OpenCV: 0-179
    S_raw = hsv_pixels[:, 1]             # 0-255
    V_raw = hsv_pixels[:, 2]             # 0-255

    H = float(np.mean(H_raw / 179.0))   # normalise 0-1
    S = float(np.mean(S_raw / 255.0))
    V = float(np.mean(V_raw / 255.0))

    # ── 3. Dark Green Colour Index (DGCI) ────────────────────────────────────
    # Formula from report:  DGCI = (H_norm + (1 - S) + (1 - V)) / 3
    DGCI = float(np.mean((H + (1 - S) + (1 - V)) / 3.0))

    feature_vector = np.array([r, g, b, H, S, V, DGCI], dtype=np.float32)
    return feature_vector


def _get_leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Simple green-channel thresholding to isolate the leaf area.
    Returns a binary mask (uint8).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Green hue range in OpenCV HSV
    lower_green = np.array([20, 30, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Also include yellow-green leaves
    lower_yellow = np.array([15, 30, 30], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined = cv2.bitwise_or(mask, mask_yellow)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined


def preprocess_image_for_cnn(image_bgr: np.ndarray,
                              target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Resize and normalise a leaf image for CNN input.
    Returns array of shape (1, 128, 128, 3) — ready for model.predict().
    """
    resized = cv2.resize(image_bgr, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalised, axis=0)   # (1, 128, 128, 3)
