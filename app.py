import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("Mental_health_prediction.pkl")

model = load_model()

# ---------------- HEADER ----------------
st.title("🧠 Mental Health Score Predictor")
st.markdown("Predict mental health score based on lifestyle & social media usage")

# ---------------- INPUT FORM ----------------
st.subheader("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 60, 22)
    study_hours = st.slider("Study Hours", 0.0, 12.0, 4.0)
    usage_hours = st.slider("Daily Social Media Usage (hrs)", 0.0, 15.0, 5.0)
    unlocks = st.number_input("Daily Phone Unlocks", 0, 500, 50)

with col2:
    physical = st.slider("Physical Activity Hours", 0.0, 10.0, 1.0)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 6.0)

# ---------------- CATEGORICAL ----------------
stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High', 'Very High'])

gender = st.selectbox("Gender", ['Male', 'Female'])

academic = st.selectbox("Academic Level", [
    'High School', 'Undergraduate', 'Postgraduate'
])

platform = st.selectbox("Most Used Platform", [
    'Instagram', 'Facebook', 'YouTube', 'Twitter', 'LinkedIn', 'Other'
])

purpose = st.selectbox("Purpose of Use", [
    'Entertainment', 'Education', 'Socializing', 'Work'
])

country = st.selectbox("Country", [
    'India','USA','UK','Canada','Australia','Germany','France','Other'
])

# ---------------- CREATE DATAFRAME ----------------
input_df = pd.DataFrame([{
    'Study_Hours': study_hours,
    'Age': age,
    'Avg_Daily_Usage_Hours': usage_hours,
    'Daily_Unlocks': unlocks,
    'Physical_Activity_Hours': physical,
    'Sleep_Hours_Per_Night': sleep,
    'Stress_Level': stress,
    'Gender': gender,
    'Academic_Level': academic,
    'Most_Used_Platform': platform,
    'Purpose_Of_Use': purpose,
    'grouped_country': country
}])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Mental Health Score"):
    try:
        prediction = model.predict(input_df)[0]

        st.success(f"🧠 Predicted Mental Health Score: {prediction:.2f}")

        # Interpretation
        if prediction < 40:
            st.error("⚠️ Poor Mental Health")
        elif prediction < 70:
            st.warning("⚠️ Moderate Mental Health")
        else:
            st.success("✅ Good Mental Health")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | ML Model")