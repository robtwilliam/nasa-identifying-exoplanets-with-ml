# Exoplanet Classifier API

A machine learning web service that predicts whether an observed astronomical object is a Confirmed Exoplanet, a Planet Candidate, or a False Positive, based on data from NASA’s Kepler, TESS, and Exoplanet Archive missions.

This project combines a tuned scikit-learn HistGradientBoostingClassifier model with a FastAPI web backend, making it easy to serve predictions through a web form or JSON API.

## Overview

### What it does:
The Exoplanet Classifier analyzes a set of planetary and stellar features (for example, orbital period, transit depth, stellar temperature) to classify the observation into one of three categories:

### Label Description
0 – FALSE POSITIVE	The signal is likely not planetary (e.g., eclipsing binary, instrument noise).<br>
1 – CANDIDATE	The object shows a planet-like signal but lacks full confirmation.<br>
2 – CONFIRMED	The object is a validated exoplanet.<br>

### Model Summary:

Type: HistGradientBoostingClassifier

Best Params:
max_depth = 8
learning_rate = 0.08
l2_regularization = 0.0
max_iter = 1000
early_stopping = True
validation_fraction = 0.1
n_iter_no_change = 20
random_state = 42

Performance:
Validation F1 (macro): 0.769
Test F1 (macro): 0.789

## Project Structure

```
project_root/
├── model_api/
│   ├── app.py                  # FastAPI web app
│   └── ...
├── model_export/               # trained model + metadata
│   ├── exoplanet_model_tuned_all_<timestamp>.pkl
│   └── exoplanet_model_tuned_all_<timestamp>.meta.json
├── start_app.py                # main launcher (cross-platform)
├── ml_model_development.ipynb  # model training notebook
├── requirements.txt
└── README.md
```

## Setup Instructions

### Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate (on macOS/Linux)
venv\Scripts\activate (on Windows)

### Install dependencies:

pip install -r requirements.txt

Confirm your model files exist under:

```
model_export/
├── exoplanet_model_tuned_all_<timestamp>.pkl
└── exoplanet_model_tuned_all_<timestamp>.meta.json
```

### Launching the App

Run using the Python launcher (recommended):

python start_app.py

This will:

Activate your virtual environment (if it exists)

Start the FastAPI server at http://127.0.0.1:8000

Manually go to http://127.0.0.1:8000/form in your browser to test it.

You can enter planetary and stellar parameters (such as period_days, snr_model, star_teff_k) and get an instant classification with class probabilities.

### API Usage

POST /predict
Send a JSON payload with feature values.

Example:

curl -X POST "http://127.0.0.1:8000/predict
"
-H "Content-Type: application/json"
-d '{"period_days": 12.3, "duration_hr": 3.1, "depth_ppm": 500,
"snr_model": 12.0, "planet_radius_re": 2.1, "star_teff_k": 5300,
"mag_mission": 10.8, "distance_pc": 150}'

Example response:

{
"prediction_index": 1,
"prediction_label": "CANDIDATE",
"probabilities": {
"FALSE POSITIVE": 0.18,
"CANDIDATE": 0.61,
"CONFIRMED": 0.21
}
}

## Notes

Missing inputs are handled automatically via median imputation.

Feature scaling and preprocessing are baked into the trained pipeline.

The app supports both form-based and API-based usage.

## AI Acknowledgement

This code and project was developed with the assistance of Open AI's ChatGPT 5.

## License

MIT License © 2025
Developed with Python, FastAPI, and scikit-learn.