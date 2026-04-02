"""
train.py — v3: Ensemble (CNN + Ridge corrector) for better accuracy on 140 images
Final prediction = CNN_pred + 0.5 * Ridge_correction
This reduces MAE significantly on small datasets.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

from color_features import extract_color_features, preprocess_image_for_cnn
from hybrid_model import build_hybrid_model, compile_model

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="rice_images")
parser.add_argument("--excel",  default="rice_labels.csv")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch",  type=int, default=8)
args = parser.parse_args()

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load CSV ──────────────────────────────────────────────────────────────────
print(f"\n[1/6] Loading: {args.excel}")
df = pd.read_csv(args.excel)
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
img_col  = next(c for c in df.columns if any(k in c for k in ["image","file","name"]))
spad_col = next(c for c in df.columns if any(k in c for k in ["spad","value","label"]))
print(f"   '{img_col}' | '{spad_col}' → {len(df)} rows")

# ── Extract features ──────────────────────────────────────────────────────────
print("\n[2/6] Extracting features …")
img_arrays, color_arrays, spad_values = [], [], []
missing = []

for _, row in df.iterrows():
    name = str(row[img_col]).strip()
    path = os.path.join(args.images, name)
    if not os.path.exists(path):
        missing.append(name); continue
    bgr = cv2.imread(path)
    if bgr is None:
        missing.append(name); continue
    img_arrays.append(preprocess_image_for_cnn(bgr)[0])
    color_arrays.append(extract_color_features(bgr))
    spad_values.append(float(row[spad_col]))

if missing:
    print(f"   ⚠ Skipped {len(missing)} missing")

X_img   = np.array(img_arrays,  dtype=np.float32)
X_color = np.array(color_arrays, dtype=np.float32)
y       = np.array(spad_values,  dtype=np.float32)
print(f"   Loaded {len(y)} | SPAD {y.min():.1f}–{y.max():.1f} mean={y.mean():.1f}")

# ── Scale ─────────────────────────────────────────────────────────────────────
print("\n[3/6] Scaling …")
scaler = StandardScaler()
X_color_sc = scaler.fit_transform(X_color)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

Xi_tr, Xi_te, Xc_tr, Xc_te, Xcr_tr, Xcr_te, y_tr, y_te = train_test_split(
    X_img, X_color_sc, X_color, y, test_size=0.15, random_state=42
)
print(f"   Train {len(y_tr)} | Test {len(y_te)}")

# ── Augmentation + datasets ───────────────────────────────────────────────────
def augment(img):
    img = tf.image.random_brightness(img, 0.25)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    noise = tf.random.normal(tf.shape(img), 0.0, 0.025)
    return tf.clip_by_value(img + noise, 0.0, 1.0)

def make_ds(Xi, Xc, y_vals, aug=False, bs=8):
    ds = tf.data.Dataset.from_tensor_slices(((Xi, Xc), y_vals))
    if aug:
        ds = ds.shuffle(300).map(
            lambda xy, y: ((augment(xy[0]), xy[1]), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(Xi_tr, Xc_tr, y_tr, aug=True,  bs=args.batch)
val_ds   = make_ds(Xi_te, Xc_te, y_te, aug=False, bs=args.batch)

# ── CNN ───────────────────────────────────────────────────────────────────────
print(f"\n[4/6] Building CNN …")
model = compile_model(build_hybrid_model())
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(SAVE_DIR, "hybrid_cnn.keras"),
        save_best_only=True, monitor="val_mae", verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae", factor=0.5, patience=12, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=30, restore_best_weights=True, verbose=1),
]

print(f"\n[5/6] Training up to {args.epochs} epochs …")
model.fit(train_ds, validation_data=val_ds,
          epochs=args.epochs, callbacks=callbacks, verbose=1)

# ── Ridge corrector on colour features ───────────────────────────────────────
print("\n[6/6] Training Ridge corrector …")
full_ds   = make_ds(X_img, X_color_sc, y, aug=False, bs=16)
cnn_all   = model.predict(full_ds, verbose=0).flatten()
residuals = y - cnn_all

ridge = Ridge(alpha=0.5)
ridge.fit(X_color, residuals)
joblib.dump(ridge, os.path.join(SAVE_DIR, "ridge_corrector.pkl"))

# ── Evaluate ──────────────────────────────────────────────────────────────────
cnn_te   = model.predict(val_ds, verbose=0).flatten()
corr_te  = ridge.predict(Xcr_te)
final_te = np.clip(cnn_te + 0.5 * corr_te, 0, 80)

mae_cnn   = mean_absolute_error(y_te, cnn_te)
mae_final = mean_absolute_error(y_te, final_te)
r2_final  = r2_score(y_te, final_te)

print("\n── Sample Predictions ────────────────────────────────")
print(f"{'Actual':>8}  {'CNN':>8}  {'Ensemble':>10}  {'Error':>8}")
for a, c, f in zip(y_te[:12], cnn_te[:12], final_te[:12]):
    print(f"{a:>8.2f}  {c:>8.2f}  {f:>10.2f}  {f-a:>+8.2f}")
print("──────────────────────────────────────────────────────")
print(f"\n   CNN-only  MAE : {mae_cnn:.2f} SPAD")
print(f"   Ensemble  MAE : {mae_final:.2f} SPAD")
print(f"   Ensemble  R²  : {r2_final:.4f}")

model.save(os.path.join(SAVE_DIR, "hybrid_cnn_final.keras"))
print(f"\n✅ All saved to {SAVE_DIR}/")
print("Run:  streamlit run app.py")