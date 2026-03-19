import streamlit as st
import joblib
import pandas as pd
import os

MODEL_PATH = "model/best_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Check the path.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Phone Addiction Predictor", page_icon="📱")

st.title("📱 Phone Addiction Level Predictor")
st.subheader("Enter Student Details")

# Inputs
daily_usage = st.number_input("Daily Usage Hours", 0.0, 24.0, 2.0)
sleep = st.number_input("Sleep Hours", 0.0, 15.0, 7.0)
anxiety = st.number_input("Anxiety Level (0-10)", 0, 10, 5)
academic = st.number_input("Academic Performance (0-100)", 0, 100, 75)
age = st.number_input("Age", 5, 25, 16)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

if st.button("Predict Addiction Level"):

    if model is None:
        st.stop()

    # Encode Gender to match training data
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    gender_encoded = gender_map[gender]

    input_data = pd.DataFrame([{
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep,
        "Anxiety_Level": anxiety,
        "Academic_Performance": academic,
        "Age": age,
        "Gender": gender_encoded
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Addiction Level: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })
            st.subheader("Prediction Confidence")
            st.bar_chart(prob_df.set_index("Class"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")