import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load the trained KNN model
with open("random_forest_model.pkl", "rb") as model_file:
    random_forest_model = pickle.load(model_file)

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

    prediction = random_forest_model.predict(input_data)
    print(prediction)

    return {"fraud_prediction": bool(int(prediction))}


@app.get("/")
async def root():
    return {"message": "Hello World"}
