# ============================================
# HealthGuard AI – Interactive Web App
# ============================================

import streamlit as st
import numpy as np
import joblib

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------
# CUSTOM CSS (UI MAGIC ✨)
# --------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.result-safe {
    background-color: #0f5132;
    padding: 15px;
    border-radius: 10px;
    color: white;
}
.result-risk {
    background-color: #842029;
    padding: 15px;
    border-radius: 10px;
    color: white;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# LOAD MODELS
# --------------------------------------------
@st.cache_resource
def load_models():
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    heart_model = joblib.load("models/heart_model.pkl")
    heart_features = joblib.load("models/heart_features.pkl")
    kidney_model = joblib.load("models/kidney_model.pkl")
    kidney_features = joblib.load("models/kidney_features.pkl")
    return diabetes_model, heart_model, heart_features, kidney_model, kidney_features

diabetes_model, heart_model, heart_features, kidney_model, kidney_features = load_models()

# --------------------------------------------
# SIDEBAR
# --------------------------------------------
st.sidebar.title("🩺 HealthGuard AI")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🩸 Diabetes", "❤️ Heart Disease", "🧪 Kidney Disease", "ℹ️ About"]
)

# --------------------------------------------
# HOME
# --------------------------------------------
if menu == "🏠 Home":
    st.title("🩺 HealthGuard AI")
    st.subheader("Intelligent Multi-Disease Prediction System")

    st.markdown("""
    <div class="card">
    <h3>🔬 Diseases Covered</h3>
    <ul>
        <li>Diabetes</li>
        <li>Heart Disease</li>
        <li>Chronic Kidney Disease</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("⚠️ This application is for educational purposes only and not a medical diagnosis.")

# --------------------------------------------
# DIABETES
# --------------------------------------------
elif menu == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose Level", 0, 300, 120)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)

        with col2:
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)

        if st.button("🔍 Predict Diabetes"):
            data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            pred = diabetes_model.predict(data)[0]

            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ High Risk of Diabetes</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ No Diabetes Detected</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# HEART DISEASE
# --------------------------------------------
elif menu == "❤️ Heart Disease":
    st.title("❤️ Heart Disease Prediction")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        restecg = st.number_input("Rest ECG (0–2)", 0, 2, 1)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        if st.button("🔍 Predict Heart Disease"):
            raw = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg,
                "thalach": thalach, "exang": exang
            }

            vector = np.zeros(len(heart_features))
            for i, col in enumerate(heart_features):
                if col in raw:
                    vector[i] = raw[col]

            pred = heart_model.predict([vector])[0]

            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ No Heart Disease Detected</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# KIDNEY DISEASE
# --------------------------------------------
elif menu == "🧪 Kidney Disease":
    st.title("🧪 Kidney Disease Prediction")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        age = st.number_input("Age", 1, 120, 45)
        bp = st.number_input("Blood Pressure", 50, 200, 80)
        sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
        al = st.number_input("Albumin", 0, 5, 1)
        su = st.number_input("Sugar", 0, 5, 0)
        bgr = st.number_input("Blood Glucose Random", 70, 500, 120)
        bu = st.number_input("Blood Urea", 10, 400, 40)
        sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.0)

        if st.button("🔍 Predict Kidney Disease"):
            raw = {
                "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
                "bgr": bgr, "bu": bu, "sc": sc, "hemo": hemo
            }

            vector = np.zeros(len(kidney_features))
            for i, col in enumerate(kidney_features):
                if col in raw:
                    vector[i] = raw[col]

            pred = kidney_model.predict([vector])[0]

            if pred == 1:
                st.markdown("<div class='result-risk'>⚠️ High Risk of Kidney Disease</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-safe'>✅ No Kidney Disease Detected</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------
# ABOUT
# --------------------------------------------
else:
    st.title("ℹ️ About HealthGuard AI")
    st.markdown("""
    <div class="card">
    <b>HealthGuard AI</b> is a machine-learning based web application that predicts
    the risk of Diabetes, Heart Disease, and Chronic Kidney Disease using
    patient medical data.

    <br><br>
    <b>Technologies Used:</b>
    <ul>
        <li>Python</li>
        <li>Scikit-learn</li>
        <li>Streamlit</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("<div class='footer'>⚠️ Educational use only. Not a medical diagnosis.</div>", unsafe_allow_html=True)
