import streamlit as st
import joblib
import numpy as np

# Page configuration

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained model (JOBLIB)

model = joblib.load("student_performance_model.pkl")

# Title & Description

st.title("🎓 Student Performance Prediction")

st.write(
    """
    This application predicts the **probability of a student failing**
    based on academic performance, study habits, and social factors.
    """
)

# Input Section

st.subheader("📥 Enter Student Details")

G1 = st.slider("G1 Marks", 0, 20, 10)
G2 = st.slider("G2 Marks", 0, 20, 10)

studytime = st.slider(
    "Study Time (hours per week)",
    1, 4, 2,
    help="1 = very low, 4 = very high"
)

failures = st.slider("Past Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 50, 5)

famrel = st.slider(
    "Family Relationship Quality",
    1, 5, 3,
    help="1 = very poor, 5 = excellent"
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

# Prediction

if st.button("🔍 Predict Performance"):

    input_data = np.array([[
        G1, G2, studytime, failures,
        absences, famrel, goout, health
    ]])

    # IMPORTANT:
    # class 0 = FAIL, class 1 = PASS
    fail_probability = model.predict_proba(input_data)[0][0]

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Fail Probability",
        value=f"{fail_probability:.2f}"
    )

    # Risk Interpretation
    
    if fail_probability >= 0.75:
        st.error("🚨 Risk Level: HIGH RISK")
        st.write("**Status:** LIKELY TO FAIL")
        st.write(
            "🛑 **Recommended Action:** Immediate counseling, "
            "parent involvement, and extra academic support."
        )

    elif fail_probability >= 0.40:
        st.warning("⚠️ Risk Level: MODERATE RISK")
        st.write("**Status:** NEEDS ATTENTION")
        st.write(
            "📌 **Recommended Action:** Monitor performance, "
            "improve study habits, and reduce absences."
        )

    else:
        st.success("✅ Risk Level: LOW RISK")
        st.write("**Status:** ON TRACK")
        st.write(
            "🎯 **Recommended Action:** Maintain consistency "
            "and current study routine."
        )

    # Worst Case Scenario (ONLY if student likely to fail)
    
    if fail_probability >= 0.75:

        st.markdown("---")
        st.subheader("⚠️ Worst Case Scenario (High Failure Risk)")

        st.write(
            "This scenario shows how severe the outcome could be "
            "if no corrective action is taken."
        )

        if st.button("🚨 Simulate Worst Case"):
            worst_case = np.array([[0, 0, 1, 4, 50, 1, 5, 1]])
            worst_fail_prob = model.predict_proba(worst_case)[0][0]

            st.metric("Fail Probability", f"{worst_fail_prob:.2f}")
            st.error("❗ Risk Level: EXTREME RISK")
            st.write("**Status:** VERY LIKELY TO FAIL")
            st.write(
                "🆘 **Action Required:** Immediate intervention and structured academic recovery plan."
            )
