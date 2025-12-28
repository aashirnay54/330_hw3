from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("census_model.pkl")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    """
    features should be a JSON object with the SAME
    column names you used in training.
    """
    X = pd.DataFrame([features])
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
