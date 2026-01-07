from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load("dt.pkl")

# -------------------------------------------------
# Feature order (CRITICAL)
# -------------------------------------------------
FEATURES = [
    "total_assists",
    "total_nb_in_group",
    "total_matches",
    "total_subed_in",
    "total_penalty_goals",
    "total_goals",
    "career_total_injuries",
    "total_yellow_cards",
    "total_nb_on_pitch",
    "total_minutes_played",
    "height",
    "total_subed_out",
    "total_own_goals",
    "total_second_yellow_cards",
    "total_direct_red_cards",
    "main_position_encoded_midfield"
]

# -------------------------------------------------
# Feature scaling params (hard-coded)
# -------------------------------------------------
MEAN = {
    "total_assists": 23.347878,
    "total_nb_in_group": 320.060681,
    "total_matches": 24.144641,
    "total_subed_in": 49.541427,
    "total_penalty_goals": 2.828382,
    "total_goals": 39.106686,
    "career_total_injuries": 4.931014,
    "total_yellow_cards": 34.002082,
    "total_nb_on_pitch": 268.586098,
    "total_minutes_played": 7865.713424,
    "height": 184.620068,
    "total_subed_out": 59.720806,
    "total_own_goals": 0.540886,
    "total_second_yellow_cards": 0.859453,
    "total_direct_red_cards": 0.894848
}

STD = {
    "total_assists": 32.092319,
    "total_nb_in_group": 216.067471,
    "total_matches": 36.231985,
    "total_subed_in": 45.503593,
    "total_penalty_goals": 7.418137,
    "total_goals": 60.167491,
    "career_total_injuries": 5.541989,
    "total_yellow_cards": 31.799643,
    "total_nb_on_pitch": 192.149683,
    "total_minutes_played": 9017.090235,
    "height": 53.775656,
    "total_subed_out": 59.735423,
    "total_own_goals": 1.137524,
    "total_second_yellow_cards": 1.381741,
    "total_direct_red_cards": 1.427938
}

# -------------------------------------------------
# Target scaling params
# -------------------------------------------------
TARGET_STD  = 193542928.53
TARGET_MEAN = 9621216.77


BINARY_FEATURES = {"main_position_encoded_midfield"}

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Market Value Prediction API (Scaled Target)")

# -------------------------------------------------
# Input schema
# -------------------------------------------------
class PlayerInput(BaseModel):
    total_assists: float
    total_nb_in_group: float
    total_matches: float
    total_subed_in: float
    total_penalty_goals: float
    total_goals: float
    career_total_injuries: float
    total_yellow_cards: float
    total_nb_on_pitch: float
    total_minutes_played: float
    height: float
    total_subed_out: float
    total_own_goals: float
    total_second_yellow_cards: float
    total_direct_red_cards: float
    main_position_encoded_midfield: int  # 0 or 1

# -------------------------------------------------
# Scaling logic
# -------------------------------------------------
def scale_input(data: PlayerInput):
    scaled = []
    for feature in FEATURES:
        value = getattr(data, feature)

        if feature in BINARY_FEATURES:
            scaled.append(value)
        else:
            scaled.append((value - MEAN[feature]) / STD[feature])

    return np.array([scaled])

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: PlayerInput):
    X_scaled = scale_input(data)

    # Model outputs scaled target
    y_scaled = model.predict(X_scaled)[0]
    print("scaled input: ", X_scaled)
    print("scaled value: ", y_scaled)

    # Inverse transform → ORIGINAL market value
    y_original = y_scaled * TARGET_STD + TARGET_MEAN

    return {
        "predicted_market_value": float(y_original)
    }
