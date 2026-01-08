# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import tensorflow as tf
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Load saved models and scalers
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(MODEL_DIR,"stacked_xgboost_model.json"))

lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR,"lstm_model.keras"))

with open(os.path.join(MODEL_DIR,"lstm_feature_scaler.pkl"), "rb") as f:
    lstm_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR,"xgb_feature_scaler.pkl"), "rb") as f:
    xgb_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR,"features.json"), "r") as f:
    features = json.load(f)

app = FastAPI(title="Player Value Prediction API")

class PlayerInput(BaseModel):
    data: list  # [{"goals_per_match":0.3,...}, ...]

@app.post("/predict")
def predict_player_value(input: PlayerInput):
    df_input = pd.DataFrame(input.data)

    # --- Set defaults for missing features ---
    default_values = {
        "age_clean": 25,
        "assists_per_match": 0.2,
        "xG_per_match": 0.25,
        "xG_performance": 0.05,
        "goals_yoy_change": 0.1,
        "assists_yoy_change": 0.05,
        "pos_Defender": 0,
        "pos_Forward": 1,
        "pos_Goalkeeper": 0,
        "pos_Midfielder": 0,
        "post_count": 5  # <-- added missing feature
    }
    
    for col, val in default_values.items():
        if col not in df_input.columns:
            df_input[col] = val

    # --- Compute derived features ---
    df_input['age_x_goals'] = df_input['age_clean'] * df_input['goals_per_match']
    df_input['age_squared'] = df_input['age_clean'] ** 2
    df_input['attack_contribution'] = df_input['goals_per_match'] + df_input['assists_per_match']

    # --- LSTM ---
    lstm_features = [f for f in features if f != 'lstm_prediction']

    # Safety check: if any column is still missing, fill with 0
    for col in lstm_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input[lstm_features] = lstm_scaler.transform(df_input[lstm_features])
    X_seq = df_input[lstm_features].values.reshape((df_input.shape[0],1,len(lstm_features)))
    lstm_pred_log = lstm_model.predict(X_seq, verbose=0)
    df_input['lstm_prediction'] = lstm_pred_log.flatten()

    # --- XGBoost ---
    X_xgb = df_input[features]
    X_xgb_scaled = xgb_scaler.transform(X_xgb)
    pred_log = xgb_model.predict(X_xgb_scaled)
    pred_euros = np.expm1(pred_log)
    pred_euros[pred_euros<0] = 0
    return {"predictions_eur": pred_euros.tolist()}
