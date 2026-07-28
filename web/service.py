import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from pyexpat import features


class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool

app = FastAPI()
model = joblib.load("../model.pkl")

@app.post("/score")
def score(data: ClientData):
    # approved = data.income > 50 or (data.education and data.work)
    features = [data.age, data.income, data.education, data.work, data.car]
    approved = not model.predict([features])[0].item()
    return {"approved": approved}

# uvicorn service:app