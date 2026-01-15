# Model2.py — Complete Modified Version
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import pickle
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "final_master_dataset.csv")

def create_sequences(data, feature_cols, target_col, sequence_length=1):
    X, y, player_ids, season_names = [], [], [], []
    for player_id, group in data.groupby('player_id'):
        features = group.sort_values('season_name')[feature_cols].values
        targets = group.sort_values('season_name')[target_col].values
        if len(features) > sequence_length:
            for i in range(len(features) - sequence_length):
                X.append(features[i:i + sequence_length])
                y.append(targets[i + sequence_length])
                player_ids.append(player_id)
                season_names.append(group.sort_values('season_name').iloc[i + sequence_length]['season_name'])
    return np.array(X), np.array(y), player_ids, season_names

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        exit()
    df = pd.read_csv(dataset_path)
    print("✅ Dataset loaded successfully!")

    # Feature columns
    feature_columns = [
        'age_clean', 'goals_per_match', 'assists_per_match', 'shots_per_match',
        'xG_per_match', 'xG_performance', 'goals_yoy_change', 'assists_yoy_change',
        'total_days_missed', 'injury_count', 'average_sentiment', 'post_count',
        'pos_Defender', 'pos_Forward', 'pos_Goalkeeper', 'pos_Midfielder'
    ]
    target_column = 'market_value_eur'

    df_model = df[['player_id', 'season_name'] + feature_columns + [target_column]].copy().dropna()
    df_model['age_x_goals'] = df_model['age_clean'] * df_model['goals_per_match']
    df_model['age_squared'] = df_model['age_clean'] ** 2
    df_model['attack_contribution'] = df_model['goals_per_match'] + df_model['assists_per_match']
    engineered_features = ['age_x_goals', 'age_squared', 'attack_contribution']
    lstm_feature_columns = feature_columns + engineered_features

    # --- Train LSTM ---
    feature_scaler_lstm = MinMaxScaler()
    df_lstm = df_model.copy()
    df_lstm[lstm_feature_columns] = feature_scaler_lstm.fit_transform(df_lstm[lstm_feature_columns])
    df_lstm[target_column] = np.log1p(df_lstm[target_column])

    X_seq, y_seq, _, _ = create_sequences(df_lstm, lstm_feature_columns, target_column)
    if len(X_seq) == 0:
        print("Not enough data for sequences. Exiting.")
        exit()

    X_train_lstm, _, y_train_lstm, _ = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=1)

    # Generate LSTM predictions as feature
    lstm_predictions_log = lstm_model.predict(X_seq, verbose=0)
    df_predictions = pd.DataFrame({
        'player_id': df_model['player_id'].iloc[1:len(lstm_predictions_log)+1].values,
        'season_name': df_model['season_name'].iloc[1:len(lstm_predictions_log)+1].values,
        'lstm_prediction': lstm_predictions_log.flatten()
    })
    df_model = pd.merge(df_model, df_predictions, on=['player_id','season_name'], how='left')

    # --- Train XGBoost ---
    final_feature_columns = lstm_feature_columns + ['lstm_prediction']
    df_xgb = df_model.copy()
    df_xgb[target_column] = np.log1p(df_xgb[target_column])
    df_xgb.dropna(subset=['lstm_prediction'], inplace=True)

    X = df_xgb[final_feature_columns]
    y = df_xgb[target_column]
    feature_scaler_xgb = MinMaxScaler()
    X_scaled = feature_scaler_xgb.fit_transform(X)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_scaled, y)

    # --- Save models & scalers ---
    os.makedirs(os.path.join(BASE_DIR,"saved_models"), exist_ok=True)
    xgb_model.save_model(os.path.join(BASE_DIR,"saved_models","stacked_xgboost_model.json"))
    lstm_model.save(os.path.join(BASE_DIR,"saved_models","lstm_model.keras"))

    with open(os.path.join(BASE_DIR,"saved_models","lstm_feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler_lstm, f)
    with open(os.path.join(BASE_DIR,"saved_models","xgb_feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler_xgb, f)
    with open(os.path.join(BASE_DIR,"saved_models","features.json"), "w") as f:
        json.dump(final_feature_columns, f)

    print("✅ Models and scalers saved successfully!")
