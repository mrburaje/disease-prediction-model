import streamlit as st
import pickle
import numpy as np


# Load models
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# Models for different diseases
diabetes_model = load_model("diabetes_model.pkl")
heart_disease_model = load_model("heart_disease_model.pkl")

# Title and description
st.title("Disease Prediction System")
st.write("Select a disease, enter the relevant parameters, and receive a prediction.")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Choose a disease to predict:", ["Diabetes", "Heart Disease", "Asthma"])

# Diabetes Prediction
if disease == "Diabetes":
    st.header("Diabetes Prediction")

    gender = st.selectbox("Gender:", ["Female", "Male"])
    age = st.number_input("Age:", min_value=1, max_value=120)
    hypertension = st.selectbox("Hypertension:", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease:", ["Yes", "No"])
    bmi = st.number_input("BMI:", min_value=10.0, max_value=100.0, step=0.1)
    HbA1c_level = st.number_input("HbA1c Level:", min_value=4.0, max_value=15.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=50, max_value=300)

    if st.button("Predict Diabetes"):
        try:
            # Encode categorical inputs
            gender_female = 1 if gender == "Female" else 0
            gender_male = 1 if gender == "Male" else 0
            hypertension_yes = 1 if hypertension == "Yes" else 0
            heart_disease_yes = 1 if heart_disease == "Yes" else 0

            # Prepare the feature vector for prediction
            features = np.array([[gender_female, gender_male, age, hypertension_yes, heart_disease_yes, bmi,
                                  HbA1c_level, blood_glucose_level]])

            # Ensure feature alignment with the model
            if features.shape[1] != diabetes_model.n_features_in_:
                raise ValueError("Input feature size does not match the model's expected feature size.")

            # Predict probabilities
            probability = diabetes_model.predict_proba(features)[0][1] * 100

            # Display the result
            st.write(f"Probability of Diabetes: {probability:.2f}%")

        except ValueError as e:
            st.error(f"Error: {e}")


# Heart Disease Prediction
elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    age = st.number_input("Age:", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex:", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp):", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps):", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (chol):", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs):", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results (restecg):", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach):", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang):", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope):", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca):", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal):", [0, 1, 2, 3])

    if st.button("Predict Heart Disease"):
        # Encode inputs
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make prediction
        probability = heart_disease_model.predict_proba(features)[0][1] * 100
        st.write(f"Probability of Heart Disease: {probability:.2f}%")

# Asthma Prediction
elif disease == "Asthma":
    st.header("Asthma Prediction")
    tiredness = st.selectbox("Tiredness:", ["Yes", "No"])
    dry_cough = st.selectbox("Dry Cough:", ["Yes", "No"])
    difficulty_breathing = st.selectbox("Difficulty in Breathing:", ["Yes", "No"])
    sore_throat = st.selectbox("Sore Throat:", ["Yes", "No"])
    none_symptom = st.selectbox("None Symptom:", ["Yes", "No"])
    pains = st.selectbox("Pains:", ["Yes", "No"])
    nasal_congestion = st.selectbox("Nasal Congestion:", ["Yes", "No"])
    runny_nose = st.selectbox("Runny Nose:", ["Yes", "No"])
    none_experiencing = st.selectbox("None Experiencing:", ["Yes", "No"])
    age_group = st.selectbox("Age Group:", ["0-9", "10-19", "20-24", "25-59", "60+"])
    gender = st.selectbox("Gender:", ["Female", "Male"])

    if st.button("Predict Asthma"):
        try:
            # Encode categorical inputs
            tiredness = 1 if tiredness == "Yes" else 0
            dry_cough = 1 if dry_cough == "Yes" else 0
            difficulty_breathing = 1 if difficulty_breathing == "Yes" else 0
            sore_throat = 1 if sore_throat == "Yes" else 0
            none_symptom = 1 if none_symptom == "Yes" else 0
            pains = 1 if pains == "Yes" else 0
            nasal_congestion = 1 if nasal_congestion == "Yes" else 0
            runny_nose = 1 if runny_nose == "Yes" else 0
            none_experiencing = 1 if none_experiencing == "Yes" else 0

            # Age group encoding
            age_0_9 = 1 if age_group == "0-9" else 0
            age_10_19 = 1 if age_group == "10-19" else 0
            age_20_24 = 1 if age_group == "20-24" else 0
            age_25_59 = 1 if age_group == "25-59" else 0
            age_60_plus = 1 if age_group == "60+" else 0

            # Gender encoding
            gender_female = 1 if gender == "Female" else 0
            gender_male = 1 if gender == "Male" else 0

            # Create the feature vector for prediction
            features = np.array([[tiredness, dry_cough, difficulty_breathing, sore_throat, none_symptom, pains,
                                  nasal_congestion, runny_nose, none_experiencing, age_0_9, age_10_19, age_20_24,
                                  age_25_59, age_60_plus, gender_female, gender_male]])

            # Ensure feature alignment with the model
            if features.shape[1] != asthma_model.n_features_in_:
                raise ValueError("Input feature size does not match the model's expected feature size.")

            # Predict probabilities for asthma severity
            probabilities = asthma_model.predict_proba(features)

            # Extract probability for each class (severity)
            severity_mild = probabilities[0][:, 1] * 100  # Mild severity probability
            severity_moderate = probabilities[1][:, 1] * 100  # Moderate severity probability
            severity_none = probabilities[2][:, 1] * 100  # None severity probability

            # Display the result
            st.write(f"Probability of Mild Asthma: {severity_mild:.2f}%")
            st.write(f"Probability of Moderate Asthma: {severity_moderate:.2f}%")
            st.write(f"Probability of No Asthma: {severity_none:.2f}%")

            # Graphical Representation
            st.bar_chart({
                "Mild": severity_mild,
                "Moderate": severity_moderate,
                "None": severity_none
            })

        except ValueError as e:
            st.error(f"Error: {e}")
