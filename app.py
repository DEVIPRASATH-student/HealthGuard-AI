# ============================================
# HealthGuard AI – Intelligent Multi-Disease Prediction System
# Streamlit Web Application
# ============================================

import streamlit as st
import numpy as np
import joblib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 HealthGuard AI")
st.subheader("Intelligent Multi-Disease Prediction System")
st.write("Predict **Diabetes**, **Heart Disease**, and **Kidney Disease** using Machine Learning")

# ============================================
# LOAD MODELS & FEATURES
# ============================================
@st.cache_resource
def load_all():
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    heart_model = joblib.load("models/heart_model.pkl")
    heart_features = joblib.load("models/heart_features.pkl")
    kidney_model = joblib.load("models/kidney_model.pkl")
    kidney_features = joblib.load("models/kidney_features.pkl")
    return diabetes_model, heart_model, heart_features, kidney_model, kidney_features

diabetes_model, heart_model, heart_features, kidney_model, kidney_features = load_all()

# ============================================
# SELECT DISEASE
# ============================================
disease = st.selectbox(
    "Select Disease to Predict",
    ("Diabetes", "Heart Disease", "Kidney Disease")
)

# ============================================
# DIABETES PREDICTION
# ============================================
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ No Diabetes Detected")

# ============================================
# HEART DISEASE PREDICTION (FIXED)
# ============================================
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ("No", "Yes"))
    restecg = st.number_input("Rest ECG (0–2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))

    # Convert categorical to numeric
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):
        raw_inputs = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang
        }

        input_vector = np.zeros(len(heart_features))

        for i, col in enumerate(heart_features):
            if col in raw_inputs:
                input_vector[i] = raw_inputs[col]

        prediction = heart_model.predict([input_vector])[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ No Heart Disease Detected")

# ============================================
# KIDNEY DISEASE PREDICTION (FIXED)
# ============================================
elif disease == "Kidney Disease":
    st.header("🧪 Kidney Disease Prediction")

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
    al = st.number_input("Albumin (0–5)", 0, 5, 1)
    su = st.number_input("Sugar (0–5)", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 70, 500, 120)
    bu = st.number_input("Blood Urea", 10, 400, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.0)

    if st.button("Predict Kidney Disease"):
        raw_inputs = {
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "su": su,
            "bgr": bgr,
            "bu": bu,
            "sc": sc,
            "hemo": hemo
        }

        input_vector = np.zeros(len(kidney_features))

        for i, col in enumerate(kidney_features):
            if col in raw_inputs:
                input_vector[i] = raw_inputs[col]

        prediction = kidney_model.predict([input_vector])[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Kidney Disease")
        else:
            st.success("✅ No Kidney Disease Detected")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("⚠️ Educational purpose only. Not a medical diagnosis.")
