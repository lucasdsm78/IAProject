import codecs
import csv
import pandas as pd
import os
import pickle
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:3000/"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)


class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: float
    used_chip: float
    used_pin_number: float
    online_order: float

@app.post("/predict")
def predict(transaction: Transaction):
    input_data = [[
        transaction.distance_from_home,
        transaction.ratio_to_median_purchase_price,
        transaction.online_order
    ]]

    model_file_path = "random_forest_model.pkl"
    if os.path.exists(model_file_path):
        # Load the trained model
        with open("random_forest_model.pkl", "rb") as model_file:
            random_forest_model = pickle.load(model_file)

        prediction = random_forest_model.predict(input_data)
        print(prediction)

        return {"fraud_prediction": bool(int(prediction))}
    else:
        return {"fraud_prediction": "No Model found, please try to train one"}


@app.post("/uploadModel")
async def create_model(file: UploadFile = File(...)):
    return {"filename": file.filename, "message": "File uploaded successfully"}



@app.get("/")
async def root():
    return {"message": "Hello World"}
