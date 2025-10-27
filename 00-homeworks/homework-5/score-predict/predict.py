import pickle
from fastapi import FastAPI
from pydantic import BaseModel

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

class customeritem(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()
@app.post('/predict')
async def create_item(customer: customeritem):
    response = predict(customer.dict())
    return response

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn),
    }

    return result