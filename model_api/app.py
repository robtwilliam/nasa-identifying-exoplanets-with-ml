from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, create_model
import numpy as np
import pandas as pd
import joblib
import json

# --- Load model + metadata ---
pipe = joblib.load("../model_export/exoplanet_model_tuned_all_2025-10-06T032756Z.pkl")
with open("../model_export/exoplanet_model_tuned_all_2025-10-06T032756Z.meta.json") as f:
    meta = json.load(f)
FEATURES = meta["features"]
LABEL_MAP = {int(k): v for k, v in meta["label_map"].items()}


# ---------- FastAPI setup ----------
app = FastAPI(title="Exoplanet Classifier", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Dynamic input schema ----------
# Create Pydantic model dynamically (every feature Optional[float] or Optional[int])
field_defs = {}
for f in FEATURES:
    if f.endswith("_miss"):
        # binary flags; allow 0/1
        field_defs[f] = (Optional[int], Field(default=None, ge=0, le=1, description=f"Missingness flag for {f[:-5]}"))
    else:
        field_defs[f] = (Optional[float], Field(default=None, description=f"Feature {f}"))

FeatureInput = create_model("FeatureInput", **field_defs)

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "exoplanet-classifier",
        "num_features": len(FEATURES),
        "features": FEATURES,
        "classes": LABEL_MAP,
    }

@app.post("/predict")
def predict(payload: FeatureInput):
    row = {f: None for f in FEATURES}
    data = payload.model_dump()   # <-- instead of payload.dict()
    for k, v in data.items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row], columns=FEATURES)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    try:
        proba = pipe.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
    except AttributeError:
        pred_idx = int(pipe.predict(X)[0])
        proba = None

    result = {
        "prediction_index": pred_idx,
        "prediction_label": LABEL_MAP[pred_idx],
    }
    if proba is not None:
        result["probabilities"] = {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)}
    result["inputs"] = row
    return result

# ---------- Optional simple HTML form ----------
@app.get("/form", response_class=HTMLResponse)
def form():
    form_html = """
    <html>
      <head><title>Exoplanet Classifier</title></head>
      <body>
        <h2>Enter Exoplanet Features</h2>
        <form id="predictForm">
    """
    for f in FEATURES:
        form_html += f'<label>{f}: <input id="{f}" type="number" step="any"></label><br/>'

    form_html += """
          <button type="submit">Predict</button>
        </form>
        <pre id="output"></pre>
        <script>
          const form = document.getElementById('predictForm');
          form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {};
            form.querySelectorAll('input').forEach(input => {
              const val = input.value;
              data[input.id] = val ? parseFloat(val) : null;
            });
            const res = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(data)
            });
            const json = await res.json();
            document.getElementById('output').textContent = JSON.stringify(json, null, 2);
          });
        </script>
      </body>
    </html>
    """
    return form_html