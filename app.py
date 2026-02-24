import streamlit as st
import joblib
import numpy as np

# Load trained model

model = joblib.load("student_performance_model.pkl")

# Page configuration

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title & Description

st.title("🎓 Student Performance Prediction")
st.markdown(
    """
    This application predicts the **probability of a student failing**
    based on **academic performance, behavioral patterns, and social factors**.
    The goal is to assist institutions in **early identification of at-risk students**.
    """
)

st.markdown("---")

# Sidebar Inputs

st.sidebar.header("📌 Student Information")

g1 = st.sidebar.slider("G1 Marks (Mid-term)", 0, 20, 10)
g2 = st.sidebar.slider("G2 Marks (Pre-final)", 0, 20, 10)

studytime = st.sidebar.slider(
    "Study Time (hours per week)",
    1, 4, 2,
    help="1: <2 hrs, 2: 2–5 hrs, 3: 5–10 hrs, 4: >10 hrs"
)

failures = st.sidebar.slider("Past Academic Failures", 0, 4, 0)
absences = st.sidebar.slider("Class Absences", 0, 50, 5)

famrel = st.sidebar.slider(
    "Family Relationship Quality",
    1, 5, 3,
    help="1: Very poor, 5: Excellent"
)

goout = st.sidebar.slider(
    "Social Activity (Going Out)",
    1, 5, 3,
    help="1: Very low, 5: Very high"
)

health = st.sidebar.slider(
    "Health Status",
    1, 5, 3,
    help="1: Very poor, 5: Very good"
)

st.markdown("---")

# Prediction

if st.button("🔮 Predict Student Performance"):

    input_data = np.array([[g1, g2, studytime, failures, absences, famrel, goout, health]])
    fail_probability = model.predict_proba(input_data)[0][1]

    # Risk classification
    if fail_probability < 0.3:
        risk = "LOW RISK"
        color = "🟢"
        action = "Student is performing well. No immediate intervention required."
    elif fail_probability < 0.6:
        risk = "MEDIUM RISK"
        color = "🟠"
        action = "Monitor the student and provide academic support if necessary."
    else:
        risk = "HIGH RISK"
        color = "🔴"
        action = "Immediate counseling and additional academic assistance is required."

    # Results Display

    st.markdown("## 📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fail Probability", f"{fail_probability:.2f}")
    col2.metric("Risk Level", risk)
    col3.metric("Status", "AT RISK" if risk != "LOW RISK" else "SAFE")

    st.markdown("---")

    st.markdown(
        f"""
        ### 🧠 Interpretation
        - **Risk Category:** {color} **{risk}**
        - **Recommended Action:**  
          *{action}*
        """
    )

# Worst Case Scenario

st.markdown("---")
st.markdown("## 🚨 Worst Case Scenario Simulation")

st.write(
    "This simulation demonstrates how the model behaves under extreme academic stress conditions."
)

if st.button("Simulate Worst Case Student"):
    worst_case = np.array([[5, 5, 1, 4, 40, 1, 5, 1]])
    prob = model.predict_proba(worst_case)[0][1]

    st.error(
        f"""
        **Fail Probability:** {prob:.2f}  
        This student falls under **EXTREME HIGH RISK** and requires **immediate institutional intervention**.
        """
    )

st.markdown("---")
st.caption("📘 End-to-End Machine Learning Project | Streamlit Deployment")
