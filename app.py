import streamlit as st
import pandas as pd
import joblib

# Page configuration

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained model (JOBLIB)

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

model = load_model()

# App Title & Description

st.title("🎓 Student Performance Prediction")
st.write(
    """
    This application predicts the **probability of a student failing**
    based on academic performance, study habits, and social factors.
    """
)

# Input Section

st.header("📥 Enter Student Details")

G1 = st.slider("G1 Marks", 0, 20, 10)
G2 = st.slider("G2 Marks", 0, 20, 10)

studytime = st.slider(
    "Study Time (hours per week)",
    1, 4, 2,
    help="1 = very low, 4 = very high"
)

failures = st.slider(
    "Past Failures",
    0, 4, 0,
    help="Number of past subject failures"
)

absences = st.slider(
    "Absences",
    0, 50, 5,
    help="Total classes missed"
)

famrel = st.slider(
    "Family Relationship Quality",
    1, 5, 3,
    help="1 = very bad, 5 = excellent"
)

goout = st.slider(
    "Going Out Frequency",
    1, 5, 3,
    help="1 = very low, 5 = very high"
)

health = st.slider(
    "Health Status",
    1, 5, 3,
    help="1 = very poor, 5 = very good"
)

# Prediction Button

if st.button("🔍 Predict Performance"):

    input_data = pd.DataFrame([{
        "G1": G1,
        "G2": G2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "famrel": famrel,
        "goout": goout,
        "health": health
    }])

    # Probability of FAILING (class 0)
    fail_prob = model.predict_proba(input_data)[0][0]

    # Risk interpretation

    if fail_prob < 0.30:
        risk = "LOW RISK"
        status = "ON TRACK"
        action = "Maintain current study routine."
        color = "green"
    elif fail_prob < 0.60:
        risk = "MODERATE RISK"
        status = "NEEDS ATTENTION"
        action = "Monitor progress and improve study consistency."
        color = "orange"
    else:
        risk = "HIGH RISK"
        status = "LIKELY TO FAIL"
        action = "Immediate counseling and academic support required."
        color = "red"

    # Output Section

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Fail Probability",
        value=f"{fail_prob:.2f}"
    )

    st.markdown(
        f"### 🚨 Risk Level: **:{color}[{risk}]**"
    )

    st.markdown(
        f"**Status:** {status}"
    )

    st.markdown(
        f"🛑 **Recommended Action:** {action}"
    )

    # -----------------------------
    # Worst Case Scenario (ONLY if high risk)
    # -----------------------------
    if fail_prob >= 0.60:
        with st.expander("⚠️ Worst Case Scenario (High Failure Risk)"):
            st.write(
                """
                This scenario illustrates what could happen
                **if no corrective action is taken**.
                """
            )
            st.markdown("❗ **Very likely to fail final exams**")
            st.markdown("🆘 **Urgent academic recovery plan required**")
