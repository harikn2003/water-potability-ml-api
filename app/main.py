from fastapi import FastAPI
from app.schema import WaterInput
from app.model import load_model, load_scaler
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

model = load_model()
scaler = load_scaler()

@app.get("/")
def home():
    return {"message": "Water Potability ML API Running"}

@app.post("/predict")
def predict(data: WaterInput):
    try:
        logging.info(f"Received input: {data}")

        input_data = np.array([[
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity
        ]])

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        logging.info(f"Prediction: {prediction}, Probability: {probability}")

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "result": "Safe" if prediction == 1 else "Not Safe"
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}