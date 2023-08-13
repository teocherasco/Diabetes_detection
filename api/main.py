from fastapi import FastAPI
import uvicorn
import json
import pickle
import numpy as np

app = FastAPI()

mod = None
dc = None


@app.get("/ping")
async def ping():
    return "Server running"


def load_data():
    global mod, dc
    with open("../diabetes_prediction.pickle", "rb") as f:
        mod = pickle.load(f)

    with open("../columns.json", "r") as f:
        dc = json.load(f)["data_columns"]


def get_label(prediction):
    if prediction == 0:
        return "No Diabetes"
    elif prediction == 1:
        return "Diabetes"
    else:
        return "Error"


@app.post("/predict")
async def predict(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, gender, smoking_history):
    try:
        gender_index = dc.index(gender.lower())
    except Exception:
        gender_index = -1
    try:
        smoke_index = dc.index(smoking_history.lower())
    except Exception:
        smoke_index = -1

    x = np.zeros(len(dc))
    x[0] = age
    x[1] = hypertension
    x[2] = heart_disease
    x[3] = np.log(int(bmi))
    x[4] = HbA1c_level
    x[5] = blood_glucose_level
    if gender_index >= 0:
        x[gender_index] = 1
    if smoke_index >= 0:
        x[smoke_index] = 1

    prediction = mod.predict([x])[0]
    return get_label(prediction)


if __name__ == "__main__":
    load_data()
    uvicorn.run(app, host="localhost", port=8000)
