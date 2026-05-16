from fastapi import FastAPI
from pydantic import BaseModel

import pickle

from src.preprocess import clean_text

# Load model
model = pickle.load(
    open("models/model.pkl", "rb")
)

# Load vectorizer
vectorizer = pickle.load(
    open("models/vectorizer.pkl", "rb")
)

# Create app
app = FastAPI(title="Sentiment Analysis API")

# Input schema
class TextData(BaseModel):
    text: str

# Home route
@app.get("/")
def home():

    return {
        "message": "Sentiment API Running"
    }

# Prediction route
@app.post("/predict")
def predict(data: TextData):

    cleaned = clean_text(data.text)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    return {
        "sentiment": prediction
    }