# ==========================================================
# HealthGuard AI – Doctor Style Intelligent Dashboard
# ==========================================================

import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="HealthGuard AI – Doctor Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS (UI + ANIMATIONS)
# ==========================================================
st.markdown("""
<style>
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
.fade { animation: fadeIn 0.8s ease-in-out; }

.card {
  background: #161b22;
  padding: 20px;
  border-radius: 15px;
  margin-bottom: 15px;
}

.safe {
  background: #0f5132;
  padding: 15px;
  border-radius: 10px;
  color: white;
}

.risk {
  background: #842029;
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

# ==========================================================
# LOAD MODELS & FEATURES
# ==========================================================
@st.cache_resource
def load_models():
    return (
        joblib.load("models/diabetes_model.pkl"),
        joblib.load("models/heart_model.pkl"),
        joblib.load("models/heart_features.pkl"),
        joblib.load("models/kidney_model.pkl"),
        joblib.load("models/kidney_features.pkl")
    )

diabetes_model, heart_model, heart_features, kidney_model, kidney_features = load_models()

# ==========================================================
# SIDEBAR – USER PROFILE
# ==========================================================
st.sidebar.title("👤 Patient Profile")

if "profile" not in st.session_state:
    st.session_state.profile = {}

name = st.sidebar.text_input("Patient Name")
age_p = st.sidebar.number_input("Age", 1, 120, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

if st.sidebar.button("Save Profile"):
    st.session_state.profile = {
        "name": name,
        "age": age_p,
        "gender": gender,
        "date": datetime.now().strftime("%d-%m-%Y %H:%M")
    }
    st.sidebar.success("Profile saved")

# ==========================================================
# NAVIGATION
# ==========================================================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🩸 Diabetes", "❤️ Heart", "🧪 Kidney", "📄 Report"]
)

# ==========================================================
# DASHBOARD
# ==========================================================
if menu == "🏠 Dashboard":
    st.title("🩺 HealthGuard AI – Doctor Dashboard")
    st.markdown("<div class='fade card'>", unsafe_allow_html=True)

    if st.session_state.profile:
        st.write("### 👤 Patient Overview")
        st.write(st.session_state.profile)
    else:
        st.info("Create patient profile from sidebar")

    st.write("### 🧠 System Capabilities")
    st.write("- Multi-disease prediction")
    st.write("- AI-assisted clinical decision support")
    st.write("- Real-time analysis")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# DIABETES
# ==========================================================
elif menu == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        with st.spinner("Analyzing patient data..."):
            data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            pred = diabetes_model.predict(data)[0]
            prob = diabetes_model.predict_proba(data)[0][1] * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Diabetes Risk %"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig)

        if pred == 1:
            st.markdown("<div class='risk'>⚠️ High Risk of Diabetes</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='safe'>✅ Low Risk of Diabetes</div>", unsafe_allow_html=True)

# ==========================================================
# HEART
# ==========================================================
elif menu == "❤️ Heart":
    st.title("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("FBS > 120", ["No", "Yes"])
    restecg = st.number_input("Rest ECG (0-2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):
        raw = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg,
            "thalach": thalach, "exang": exang
        }

        vec = np.zeros(len(heart_features))
        for i, col in enumerate(heart_features):
            if col in raw:
                vec[i] = raw[col]

        prob = heart_model.predict_proba([vec])[0][1] * 100
        pred = heart_model.predict([vec])[0]

        st.progress(int(prob))

        if pred == 1:
            st.markdown("<div class='risk'>⚠️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='safe'>✅ Low Risk of Heart Disease</div>", unsafe_allow_html=True)

# ==========================================================
# KIDNEY
# ==========================================================
elif menu == "🧪 Kidney":
    st.title("🧪 Kidney Disease Prediction")

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
    al = st.number_input("Albumin", 0, 5, 1)
    su = st.number_input("Sugar", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 70, 500, 120)
    bu = st.number_input("Blood Urea", 10, 400, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.0)

    if st.button("Predict Kidney Disease"):
        raw = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "bgr": bgr, "bu": bu, "sc": sc, "hemo": hemo
        }

        vec = np.zeros(len(kidney_features))
        for i, col in enumerate(kidney_features):
            if col in raw:
                vec[i] = raw[col]

        prob = kidney_model.predict_proba([vec])[0][1] * 100
        pred = kidney_model.predict([vec])[0]

        fig = go.Figure(go.Indicator(
            mode="number+gauge",
            value=prob,
            title={"text": "Kidney Disease Risk %"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig)

        if pred == 1:
            st.markdown("<div class='risk'>⚠️ High Risk of Kidney Disease</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='safe'>✅ Low Risk of Kidney Disease</div>", unsafe_allow_html=True)

# ==========================================================
# REPORT DOWNLOAD
# ==========================================================
else:
    st.title("📄 Patient Report")

    if not st.session_state.profile:
        st.warning("Save patient profile first")
    else:
        report = f"""
HealthGuard AI – Medical Risk Report
----------------------------------
Name: {st.session_state.profile['name']}
Age: {st.session_state.profile['age']}
Gender: {st.session_state.profile['gender']}
Date: {st.session_state.profile['date']}

⚠️ This report is AI-generated for educational purposes only.
"""
        st.download_button(
            "⬇️ Download Report",
            report,
            file_name="HealthGuard_Report.txt"
        )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("<div class='footer'>⚠️ Educational use only – Not a medical diagnosis</div>", unsafe_allow_html=True)
