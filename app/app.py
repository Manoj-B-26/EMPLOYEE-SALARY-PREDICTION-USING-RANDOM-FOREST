import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load('../model/salary_predictor_pipeline.pkl')

# Set UI
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")

st.markdown("Predict whether an employee earns **>50K or <=50K** based on their profile.")

# Sidebar inputs
age = st.sidebar.slider("Age", 18, 70, 30)
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'])
education = st.sidebar.selectbox("Education", ['HS-grad', 'Bachelors', 'Some-college', 'Masters'])
marital_status = st.sidebar.selectbox("Marital Status", ['Married-civ-spouse', 'Never-married', 'Divorced'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Exec-managerial'])
relationship = st.sidebar.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Other'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
hours = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Mexico'])

if st.sidebar.button("Predict Salary"):
    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': 0,  # dummy
        'education': education,
        'educational-num': 0,  # dummy
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours,
        'native-country': native_country
    }])

    result = model.predict(input_df)[0]
    salary = ">50K" if result == 1 else "<=50K"

    st.subheader("🧮 Prediction Result")
    st.success(f"The predicted salary class is **{salary}**.")
