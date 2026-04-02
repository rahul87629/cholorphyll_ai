# 🌿 LeafSense — Complete Setup Guide
### SPAD Chlorophyll Prediction Web App (PyCharm + Streamlit)

---

## STEP 0 — Final Folder Structure

```
chlorophyll_app/
│
├── app.py                  ← Main Streamlit web app (run this)
├── train.py                ← Training script (run once before app)
├── hybrid_model.py         ← Hybrid CNN architecture
├── color_features.py       ← Colour feature extraction (RGB, HSV, DGCI)
├── predictor.py            ← Inference + dataset update logic
├── requirements.txt        ← All Python packages
│
├── dataset/
│   ├── images/             ← YOUR leaf image folder goes here
│   └── spad_data.xlsx      ← YOUR Excel file goes here
│
└── saved_model/            ← Created automatically after training
    ├── hybrid_cnn_final.keras
    └── scaler.pkl
```

---

## STEP 1 — Install Python 3.10

TensorFlow 2.15 works best with Python 3.10.

1. Go to https://www.python.org/downloads/release/python-31011/
2. Download **Python 3.10.11** (Windows installer 64-bit)
3. Install — tick **"Add Python to PATH"** during installation
4. Verify: `python --version`  →  should show `Python 3.10.11`

---

## STEP 2 — Open in PyCharm + Create Virtual Environment

1. Open PyCharm → File → Open → select `chlorophyll_app` folder
2. In the bottom terminal:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Mac/Linux
   ```
   You'll see `(venv)` at the start of your terminal prompt.

---

## STEP 3 — Install All Packages

```bash
pip install -r requirements.txt
```
TensorFlow is large (~500 MB). This takes 5–10 minutes.

---

## STEP 4 — Prepare Your Dataset

Your Excel file must have columns like:

| image_name   | spad_value |
|--------------|-----------|
| leaf_001.jpg | 38.5      |
| leaf_002.jpg | 42.1      |

Place files:
```
dataset/images/leaf_001.jpg  (all ~140 images)
dataset/spad_data.xlsx
```
Image filenames in Excel must exactly match filenames in the folder.

---

## STEP 5 — Train the Model

```bash
python train.py --images dataset/images --excel dataset/spad_data.xlsx --epochs 80
```

Training takes ~5–15 minutes on CPU. Saves model to `saved_model/`.

---

## STEP 6 — Run the Web App

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## STEP 7 — Retraining (Continuous Learning)

After auto-saving 10+ new predictions:
```bash
python train.py --images dataset/images --excel dataset/spad_data.xlsx
```
Then restart: `Ctrl+C` → `streamlit run app.py`

---

## TROUBLESHOOTING

| Error | Fix |
|-------|-----|
| "Model not found" | Run `python train.py` first |
| "No module named tensorflow" | Activate venv, then `pip install -r requirements.txt` |
| Images not found in training | Check filenames in Excel match exactly (case-sensitive) |
| Excel column not detected | Ensure headers contain "image"/"name" and "spad"/"value" |
| App slow on first prediction | Normal — model loads once, then stays fast |

---

## FILE SUMMARY

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — upload, predict, charts, farmer guide |
| `train.py` | Load dataset, extract features, train CNN, save model |
| `hybrid_model.py` | CNN + colour feature fusion architecture |
| `color_features.py` | Extract r,g,b,H,S,V,DGCI from any image |
| `predictor.py` | Inference, dataset update, stats |
| `requirements.txt` | All pip packages |
