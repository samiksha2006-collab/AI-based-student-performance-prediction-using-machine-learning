import streamlit as st
#used to create a website
import joblib
#used to load ai model that trained in colab
import numpy as np

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Load the trained model
model = joblib.load("student_performance_model.pkl")

# Sidebar
st.sidebar.title("📌 Navigation")
st.sidebar.write("This app predicts student performance based on study habits and academic details.")

# Title
st.title("🎓 Student Performance Predictor")
st.markdown("---")

# About section
st.subheader("📌 About This Project")
st.write("""
This application predicts a student's performance based on their study habits and academic activities.

It uses a Machine Learning model trained on student data to estimate a performance score.

The goal is to help students understand how their habits affect their academic results.
""")

st.markdown("---")

# Input section
st.subheader("📥 Enter Student Details Below")

hours = st.number_input("Hours Studied", min_value=0.0)
previous_score = st.number_input("Previous Score", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0.0)
activities = st.number_input(
    "Extracurricular Activities (0 or 1)",
    min_value=0,
    max_value=1
)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Performance"):

    input_data = np.array([[hours, previous_score, activities, sleep_hours, sample_papers]])

    prediction = model.predict(input_data)

    # ✅ FIX: convert numpy array → float
    score = float(prediction.ravel()[0])

    st.subheader("📊 Prediction Result")

    if score >= 80:
        st.success(f"Predicted Performance Index: {score:.2f} → Excellent Performance 🎉")
    elif score >= 70:
        st.info(f"Predicted Performance Index: {score:.2f} → Good Performance 👍")
    elif score >= 60:
        st.warning(f"Predicted Performance Index: {score:.2f} → Average Performance 📚")
    else:
        st.error(f"Predicted Performance Index: {score:.2f} → Needs Improvement ⚠️")

st.markdown("---")
st.caption("© 2026 Student Performance Predictor")
st.caption("Developed by Samiksha")
st.caption("Built using Python, Scikit-learn, Machine Learning, and Streamlit")
st.markdown("---")
st.subheader("📌 Conclusion")

st.write("""
This project uses a Linear Regression machine learning model to predict student performance based on study habits and academic factors.

The model helps estimate a student's Performance Index by analyzing inputs such as study hours, previous scores, sleep hours, sample question paper practice, and extracurricular activities.

This project demonstrates how Artificial Intelligence and Machine Learning can support educational decision-making and encourage students to improve their study habits.
""")