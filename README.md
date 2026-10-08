# Loan Approval Prediction System (FastAPI + scikit-learn)

ML-powered loan eligibility checker. Pandas for data cleaning, a scikit-learn
pipeline for modeling, a FastAPI REST backend, and an HTML/CSS/JS front end.

## Project structure
```
train.py            # trains model, prints and saves accuracy/precision/recall
app/main.py         # FastAPI app: /predict, /metrics, /health, /docs
static/index.html   # front end
data/loan_data.csv  # YOU add this (Kaggle Loan Prediction train.csv)
model/              # model.joblib + metrics.json (created by train.py)
render.yaml         # one-click deploy config for Render
```

## Dataset
`data/sample_loan_data.csv` is **synthetic sample data** so the project runs out of the box.
For real results, download the Kaggle "Loan Prediction Problem Dataset", save its train file as
`data/loan_data.csv`, and rerun `python train.py`. Only quote metrics from the real dataset.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                  # needs data/loan_data.csv
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (Swagger).

## Dashboard features
- Live prediction: results update as you move any slider (debounced API calls)
- Animated approval-probability gauge and recent-prediction history
- EMI calculator showing EMI-to-income ratio
- Model dashboard: accuracy, precision, recall, F1, confusion matrix, top features
- Responsive, light/dark theme, accessible controls

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | /predict | Returns approval decision and probability |
| GET | /metrics | Accuracy, precision, recall, F1, confusion matrix |
| GET | /health | Service and model status |

## Model results
Fill in from `model/metrics.json` after training on the real dataset.
