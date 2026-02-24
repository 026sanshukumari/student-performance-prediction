import streamlit as st
import pickle
import numpy as np

# Load trained model

with open("student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)

# App Title & Description

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write(
    """
    This app predicts the **probability of a student failing** based on
    academic, behavioral, and social factors.
    """
)

st.markdown("---")

# Sidebar Inputs

st.sidebar.header("Enter Student Details")

g1 = st.sidebar.slider("G1 Marks", 0, 20, 10)
g2 = st.sidebar.slider("G2 Marks", 0, 20, 10)

studytime = st.sidebar.slider(
    "Study Time (hours per week)",
    1, 4, 2,
    help="1: <2 hrs, 2: 2–5 hrs, 3: 5–10 hrs, 4: >10 hrs"
)

failures = st.sidebar.slider("Past Failures", 0, 4, 0)

absences = st.sidebar.slider("Absences", 0, 50, 5)

famrel = st.sidebar.slider(
    "Family Relationship Quality",
    1, 5, 3,
    help="1: Very bad, 5: Excellent"
)

goout = st.sidebar.slider(
    "Going Out with Friends",
    1, 5, 3,
    help="1: Very low, 5: Very high"
)

health = st.sidebar.slider(
    "Health Status",
    1, 5, 3,
    help="1: Very poor, 5: Very good"
)

st.markdown("---")

# Prediction Button

if st.button("Predict Performance"):

    # Prepare input for model
    input_data = np.array([[g1, g2, studytime, failures, absences, famrel, goout, health]])

    # Predict probability
    fail_probability = model.predict_proba(input_data)[0][1]

    # Risk level
    if fail_probability < 0.3:
        risk = "LOW RISK"
        color = "green"
        action = "No intervention needed. Student is performing well."
    elif fail_probability < 0.6:
        risk = "MEDIUM RISK"
        color = "orange"
        action = "Monitor progress and provide academic support."
    else:
        risk = "HIGH RISK"
        color = "red"
        action = "Immediate counseling and extra academic help required."

    # Display results
    st.subheader("Prediction Result")

    st.markdown(
        f"""
        **Fail Probability:** `{fail_probability:.2f}`  
        **Risk Level:** <span style="color:{color}; font-weight:bold;">{risk}</span>  
        **Suggested Action:** {action}
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Worst Case Scenario Demo

st.subheader("Worst Case Scenario Demo")

if st.button("Simulate Worst Case Student"):
    worst_case = np.array([[5, 5, 1, 4, 40, 1, 5, 1]])
    prob = model.predict_proba(worst_case)[0][1]

    st.error(
        f"""
        ⚠️ **Worst Case Fail Probability:** {prob:.2f}  
        This student is at **very high risk of failing**.
        """
    )

st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")
