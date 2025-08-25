import streamlit as st
import pickle
import numpy as np
import joblib
# Load the trained model
model = joblib.load("model/DecisionTree_regression.pkl")

st.title("Decision Tree Model Deployment 🚀")

# Example: input fields for prediction
st.header("Enter Input Features")

# Replace with your dataset's features
feature1 = st.number_input("Percentage of females who are divorced (FemalePctDiv)", value=0.0)
feature2 = st.number_input("Percentage of households with public assistance income in 1989 (pctWPubAsst)", value=0.0)
feature3 = st.number_input("Percentage of households with investment / rent income in 1989 (pctWInvInc)", value=0.0)
feature4 = st.number_input("Percentage of population that is african american (racepctblack)", value=0.0)
feature5 = st.number_input("Percentage of population that is caucasian (racePctWhite)", value=0.0)
feature6 = st.number_input("Percentage of kids born to never married (PctIlleg)", value=0.0)
feature7 = st.number_input("Percentage of kids in family housing with two parents (PctKids2Par)", value=0.0)
feature8 = st.number_input("Population for community (population)", value=0.0)

# Convert inputs into numpy array
features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Prediction: {prediction[0]}")
