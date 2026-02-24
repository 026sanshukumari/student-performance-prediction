import streamlit as st
import numpy as np
import joblib

# Page config

st.set_page_config(
    page_title="Student Performance Prediction",
    layout="centered"
)

# Load model

model = joblib.load("model.joblib")

# Title & description

st.title("🎓 Student Performance Prediction")
st.write(
    "This application predicts the probability of a student failing "
    "based on academic performance, study habits, and social factors."
)

# Input section

st.subheader("📥 Enter Student Details")

G1 = st.slider("G1 Marks", 0, 20, 10)
G2 = st.slider("G2 Marks", 0, 20, 10)
studytime = st.slider("Study Time (hours per week)", 1, 4, 2)
failures = st.slider("Past Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 50, 5)
family = st.slider("Family Relationship Quality", 1, 5, 3)
goout = st.slider("Going Out Frequency", 1, 5, 3)
health = st.slider("Health Status", 1, 5, 3)

# Prediction

input_data = np.array([[G1, G2, studytime, failures, absences, family, goout, health]])
fail_probability = model.predict_proba(input_data)[0][0]

st.markdown("---")
st.subheader("📊 Prediction Result")

st.metric("Fail Probability", f"{fail_probability:.2f}")

# Risk logic

if fail_probability >= 0.75:
    st.error("🚨 Risk Level: HIGH RISK")
    st.write("**Status:** LIKELY TO FAIL")
    st.write(
        "🛑 **Recommended Action:** Immediate counseling, parent involvement, "
        "and extra academic support."
    )

elif fail_probability >= 0.40:
    st.warning("⚠️ Risk Level: MODERATE RISK")
    st.write("**Status:** NEEDS ATTENTION")
    st.write(
        "📘 **Recommended Action:** Improve study consistency and monitor progress closely."
    )

else:
    st.success("✅ Risk Level: LOW RISK")
    st.write("**Status:** ON TRACK")
    st.write(
        "🎯 **Recommended Action:** Maintain current study routine and consistency."
    )

# Worst Case Scenario

if "show_worst" not in st.session_state:
    st.session_state.show_worst = False

# Worst case ONLY when failure risk is high
if fail_probability >= 0.75:
    st.markdown("---")
    st.subheader("⚠️ Worst Case Scenario (High Failure Risk)")
    st.write(
        "This scenario shows how severe the outcome could be "
        "if no corrective action is taken."
    )

    if st.button("🚨 Simulate Worst Case"):
        st.session_state.show_worst = True

    if st.session_state.show_worst:
        worst_case = np.array([[0, 0, 1, 4, 50, 1, 5, 1]])
        worst_fail_prob = model.predict_proba(worst_case)[0][0]

        st.metric("Fail Probability", f"{worst_fail_prob:.2f}")
        st.error("❗ Risk Level: EXTREME RISK")
        st.write("**Status:** VERY LIKELY TO FAIL")
        st.write(
            "🆘 **Action Required:** Immediate intervention and a structured academic recovery plan."
        )
