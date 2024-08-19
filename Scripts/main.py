from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, conint, confloat
from typing import List
import pandas as pd
import config
from preprocess import Pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize the pipeline
pipeline = Pipeline(target=config.TARGET,
                    num_reciprocal=config.NUM_RECIPROCAL,
                    num_yeo_johnson=config.NUM_YEO_JOHNSON,
                    features=config.FEATURES)

# Load and fit the model with the existing data
data = pd.read_csv(config.DATASET_PATH)
pipeline.fit(data)

class InputData(BaseModel):
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trtbps: confloat(ge=20, le=300)
    chol: confloat(ge=100, le=800)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalachh: confloat(ge=50, le=300)
    exng: conint(ge=0, le=1)
    oldpeak: confloat(ge=0, le=7)
    slp: conint(ge=0, le=2)
    caa: conint(ge=0, le=4)
    thall: conint(ge=0, le=3)

class InputDataList(BaseModel):
    data: List[InputData]

@app.post("/predict")
def predict(input_data: InputDataList):
    input_df = pd.DataFrame([item.dict() for item in input_data.data])
    predictions = pipeline.predict(input_df)
    prediction_value = predictions.tolist()[0]
    message = "The model predicts that the person is at risk of a heart attack." if prediction_value == 1 else "The model predicts that the person is not at risk of a heart attack."
    return {"prediction": prediction_value, "message": message}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "message": None})

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    sex: conint(ge=0, le=1) = Form(...),
    cp: conint(ge=0, le=3) = Form(...),
    trtbps: confloat(ge=20, le=300) = Form(...),
    chol: confloat(ge=100, le=800) = Form(...),
    fbs: conint(ge=0, le=1) = Form(...),
    restecg: conint(ge=0, le=2) = Form(...),
    thalachh: confloat(ge=50, le=300) = Form(...),
    exng: conint(ge=0, le=1) = Form(...),
    oldpeak: confloat(ge=0, le=7) = Form(...),
    slp: conint(ge=0, le=2) = Form(...),
    caa: conint(ge=0, le=4) = Form(...),
    thall: conint(ge=0, le=3) = Form(...)
):
    input_df = pd.DataFrame([{
        "sex": sex,
        "cp": cp,
        "trtbps": trtbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalachh": thalachh,
        "exng": exng,
        "oldpeak": oldpeak,
        "slp": slp,
        "caa": caa,
        "thall": thall
    }])
    
    prediction = pipeline.predict(input_df).tolist()[0]
    message = "The model predicts that the person is at risk of a heart attack." if prediction == 1 else "The model predicts that the person is not at risk of a heart attack."
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "message": message})

# main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
