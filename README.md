# 🎓 Student Performance Prediction System

This project is a **Machine Learning–based web application** that predicts the **probability of a student failing** based on academic performance, study habits, and social factors.

The goal of this system is **early risk detection**, so that timely academic intervention can be provided to students who are likely to fail.

##  Features

- Predicts **failure probability** using a trained ML model
- Categorizes students into:
  - LOW RISK
  - MODERATE RISK
  - HIGH RISK
- Displays **recommended academic actions**
- Shows **Worst Case Scenario** only for high-risk students
- Clean and interactive UI using **Streamlit**
- Deployed on **Streamlit Cloud**

---

## Machine Learning Model

- **Algorithm:** Random Forest Classifier  
- **Reason:** Handles non-linear relationships well and is robust to noise  
- **Model File:** `student_performance_model.pkl` (saved using joblib)

---

## 📊 Input Features

 Feature - Description 

 G1 : First period marks 
 G2 : Second period marks 
 studytime : Weekly study time (1–4 scale) 
 failures : Number of past failures 
 absences : Total class absences 
 famrel : Family relationship quality (1–5) 
 goout : Going out frequency (1–5) 
 health : Health status (1–5) 
 

## 📈 Output

- **Fail Probability** (0 to 1)
- **Risk Level** (Low / Moderate / High)
- **Student Status**
- **Recommended Academic Action**
- **Worst Case Scenario** (shown only for high risk students)


## 🖥️ Web Application

The application is built using **Streamlit**, providing:
- Sliders for easy input
- Real-time predictions
- Clear visual interpretation of results


## 📦 Project Structure
