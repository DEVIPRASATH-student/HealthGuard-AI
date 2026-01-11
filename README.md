# 🩺 HealthGuard AI – Intelligent Multi-Disease Prediction System

HealthGuard AI is an **AI-powered healthcare support system** designed to assist in **early disease risk detection** using Machine Learning. The system predicts the likelihood of **Diabetes, Heart Disease, and Chronic Kidney Disease** based on user-provided clinical parameters through an **interactive doctor-style web dashboard**.

> ⚠️ *This project is intended for educational and research purposes only.*

---

## 🚀 Key Features

✅ Multi-disease prediction on a single platform
✅ Separate ML models for each disease
✅ Interactive **doctor-style dashboard UI**
✅ Animated progress indicators & charts
✅ Patient profile section
✅ Downloadable medical report
✅ Clean, user-friendly interface
✅ Real-time predictions
✅ GitHub & Streamlit-ready deployment

---

## 🧠 Diseases Covered

* **Diabetes Prediction**
* **Heart Disease Prediction**
* **Chronic Kidney Disease (CKD) Prediction**

Each disease is handled by an **independent machine learning model**, ensuring better accuracy and modularity.

---

## 📊 System Architecture

1. **Dataset Collection**

   * Public medical datasets from **Kaggle** and **UCI ML Repository**

2. **Data Preprocessing**

   * Missing value handling
   * Feature encoding
   * Data normalization
   * Categorical-to-numerical conversion

3. **Model Training**

   * Algorithm: **Logistic Regression**
   * Separate models for each disease
   * Models saved as `.pkl` files using Joblib

4. **Web Application**

   * Built using **Streamlit**
   * Interactive UI with charts and animations
   * Real-time prediction & report generation

---

## 🎯 Project Objectives

* Predict multiple diseases using machine learning
* Provide a **single unified healthcare prediction platform**
* Apply real-world medical datasets
* Deploy ML models via a web-based interface
* Demonstrate AI’s role in **preventive healthcare**

---

## 🛠️ Technologies Used

| Category             | Tools                                 |
| -------------------- | ------------------------------------- |
| Programming Language | Python                                |
| Machine Learning     | Scikit-learn (Logistic Regression)    |
| Web Framework        | Streamlit                             |
| Data Handling        | Pandas, NumPy                         |
| Model Persistence    | Joblib                                |
| Visualization        | Plotly                                |
| Deployment           | Localhost / Streamlit Community Cloud |
| Datasets             | Kaggle, UCI ML Repository             |

---

## 🖥️ User Interface Highlights

* 📌 Sidebar-based navigation
* 🧍 Patient profile input
* 📈 Animated risk indicators & charts
* 🧪 Disease-specific input forms
* 📄 Medical report download
* 🏥 Doctor-style dashboard layout

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DEVIPRASATH-student/HealthGuard-AI.git
cd HealthGuard-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models (Optional – Already Included)

```bash
python train_models.py
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```
---

## 📄 requirements.txt

```txt
streamlit
numpy
pandas
scikit-learn
joblib
plotly
```
---
## 📌 Applications
* Early disease risk assessment
* AI-assisted healthcare support systems
* Academic mini/major projects
* Research and experimentation
* AI-based medical decision support
---
## ⚠️ Disclaimer
This system is developed **only for educational and research purposes**.
It **does not replace professional medical diagnosis or treatment**.
Always consult a qualified healthcare professional for medical decisions.
---
## 👨‍💻 Author
**Prasath**
Computer Science Engineering – Undergraduate
AI & Machine Learning Enthusiast

---
⭐ *If you find this project useful, consider giving it a star on GitHub!*
---
