# frontend.py
import streamlit as st
import requests

st.title("Player Market Value Prediction")
st.write("Enter player stats:")

# Only user-input fields
goals_per_match = st.number_input("Goals per match", min_value=0.0, max_value=5.0, value=0.3)
shots_per_match = st.number_input("Shots per match", min_value=0.0, max_value=10.0, value=1.5)
injury_count = st.number_input("Injury count", min_value=0, max_value=20, value=1)
total_days_missed = st.number_input("Total days missed", min_value=0, max_value=365, value=10)
average_sentiment = st.number_input("Average sentiment", min_value=-1.0, max_value=1.0, value=0.2)

output = st.empty()  # placeholder for prediction output

if st.button("Predict Value"):
    payload = {"data": [{
        "goals_per_match": goals_per_match,
        "shots_per_match": shots_per_match,
        "injury_count": injury_count,
        "total_days_missed": total_days_missed,
        "average_sentiment": average_sentiment
    }]}
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["predictions_eur"][0]
            st.success(f"Predicted Player Market Value: €{prediction:,.2f}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

