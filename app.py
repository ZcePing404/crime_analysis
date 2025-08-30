import streamlit as st
import pickle
import numpy as np
from dashboard import show_dashboard

page = st.sidebar.radio("Select Page", ["Dashboard", "Predict"])
# Load the model
model = pickle.load(open("model/DecisionTree_regression.pkl", "rb"))
# Load the trained model
if page == "Dashboard":
    st.header("📊 Dashboard")
    show_dashboard(model)
    

elif page == "Predict":
    st.header("🔮 Prediction")
    st.header("Enter Input Features")

    # Input fields for the new attributes
    PctKids2Par = st.number_input("Percentage of kids in family housing with two parents (PctKids2Par)", value=0.7, min_value=0.0, max_value=1.0)
    PctPersDenseHous = st.number_input("Percentage of persons in dense housing (PctPersDenseHous)", value=0.1, min_value=0.0, max_value=1.0)
    PctIlleg = st.number_input("Percentage of kids born to never married (PctIlleg)", value=0.1, min_value=0.0, max_value=1.0)
    NumStreet = st.number_input("Number of people living in areas classified as urban (NumStreet)", value=0.1, min_value=0.0, max_value=1.0)
    racepctblack = st.number_input("Percentage of population that is African American (racepctblack)", value=0.1, min_value=0.0, max_value=1.0)
    HousVacant = st.number_input("Number of vacant households (HousVacant)", value=0.05, min_value=0.0, max_value=1.0)
    MalePctDivorce = st.number_input("Percentage of males who are divorced (MalePctDivorce)", value=0.1, min_value=0.0, max_value=1.0)
    RentLowQ = st.number_input("Lower quartile of rent (RentLowQ)", value=0.2, min_value=0.0, max_value=1.0)
    pctWInvInc = st.number_input("Percentage of households with investment / rent income in 1989 (pctWInvInc)", value=0.05, min_value=0.0, max_value=1.0)
    # Convert inputs into numpy array
    features = np.array([[PctKids2Par, PctPersDenseHous, PctIlleg, NumStreet, racepctblack, HousVacant, MalePctDivorce, RentLowQ, pctWInvInc]])

    if st.button("Predict"):
        prediction = model.predict(features)
        st.success(f"Predicted Violent Crime Rate: {prediction[0]:.3f}")
