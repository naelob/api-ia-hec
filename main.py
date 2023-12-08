from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
import random

class DataModel(BaseModel):
    stock_id: int
    date_id: int
    seconds_in_bucket: int
    imbalance_size: int
    imbalance_buy_sell_flag: int
    reference_price: float
    matched_size: float
    far_price: float
    near_price: float
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    wap: float
    time_id: int
    row_id: str

app = FastAPI()

@app.get("/")
async def root():
    return {"welcome": "You are on the Optiver API!"}

@app.post("/predict")
async def root(data : DataModel):

    def price_fit(x, A, B): 
        y = 1/(A + (B*x)**2)
        return y 
    
    fit_A_near = 0.004416741927896185
    fit_B_near = 24.805478617495247

    fit_A_far = 0.00695765304579013
    fit_B_far = 22.97869636334778

    df = pd.DataFrame([data.model_dump()])

    df['near_price_norm'] = df.near_price
    N_near_price = df['near_price_norm'].isna().sum()
    x = np.arange(0, 0.3, step=0.0001)
    prob = price_fit(x, fit_A_near, fit_B_near)
    df.loc[df['near_price_norm'].isna(), 'near_price_norm'] = random.choices(x, weights=prob, k=N_near_price)
    
    df['far_price_norm'] = df.far_price
    N_far_price = df['far_price_norm'].isna().sum()
    x = np.arange(0, 0.3, step=0.0001)
    prob = price_fit(x, fit_A_far, fit_B_far)
    df.loc[df['far_price_norm'].isna(), 'far_price_norm'] = random.choices(x, weights=prob, k=N_far_price)
    
    df['volume'] = df['ask_size'] + df['bid_size']
    df['ask_ref_ratio'] = df['ask_price']/df['reference_price']
    df['bid_ref_ratio'] = df['bid_price']/df['reference_price']

    with open('./models/lgbm_model.pkl', 'rb') as file:
        model = pickle.load(file)

    features = ['imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'matched_size', 
            'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'near_price_norm', 'far_price_norm',
            'volume', 'ask_ref_ratio', 'bid_ref_ratio']    

    y = model.predict(df[features])
    predictions = y.tolist() if isinstance(y, np.ndarray) else y

    return {"target": predictions}