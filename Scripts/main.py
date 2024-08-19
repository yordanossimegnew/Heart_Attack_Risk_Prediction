# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import config
from preprocess import Pipeline

app = FastAPI()

# Initialize the pipeline
pipeline = Pipeline(target=config.TARGET,
                    num_reciprocal=config.NUM_RECIPROCAL,
                    num_yeo_johnson=config.NUM_YEO_JOHNSON,
                    features=config.FEATURES)

# Load and fit the model with the existing data
data = pd.read_csv(config.DATASET_PATH)
pipeline.fit(data)


class InputData(BaseModel):
    sex: int
    cp: int
    trtbps: float
    chol: float
    fbs: int
    restecg: int
    thalachh: float
    exng: int
    oldpeak: float
    slp: int
    caa: int
    thall: int
   
    
class InputDataList(BaseModel):
    data: List[InputData]
    
    
@app.post("/predict")
def predict(input_data: InputDataList):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([item.dict() for item in input_data.data])
    
    # Make predictions
    predictions = pipeline.predict(input_df)
    
    return {"predictions": predictions.tolist()}


@app.get("/")
def read_root():
    return {"message": "Heart Attack Risk Prediction API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)