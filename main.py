from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()
#app = Flask(__name__)

class ProductRecommendation(BaseModel):
    condition1:int
    condition2:int
    condition3:int
    condition4:int
    condition5:int
    condition6:int
    condition7:int
    condition8:int
    condition9:int
    condition10:int
    condition11:int
    condition12:int
    condition13:int
    condition14:int
    condition15:int
    condition16:int
    condition17:int
    condition18:int
    condition19:int
    condition20:int
    condition21:int
    condition22:int
    condition23:int
    condition24:int
    condition25:int
    condition26:int
    condition27:int
    condition28:int

with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ProductRecommendation):
    df= pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    return {"prediction":int(yhat)}