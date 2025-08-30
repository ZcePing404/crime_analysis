import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_dashboard(model):
    st.subheader("📈 Model Insights")

    # ===== Feature Importance =====
    st.write("### Feature Importance")
    if hasattr(model, 'feature_importances_'):
        try:
            feature_names = [
                "PctKids2Par", "PctPersDenseHous", "PctIlleg", "NumStreet",
                "racepctblack", "HousVacant", "MalePctDivorce", "RentLowQ",
                "pctWInvInc"
            ]
            importances = model.feature_importances_
            fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            fi = fi.sort_values("Importance", ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not compute feature importance: {str(e)}")
    else:
        st.warning("⚠️ This model (SVR) does not provide feature importance.")

    # ===== Predicted vs Actual =====
    st.write("### Predicted vs Actual")
    # Placeholder: Replace with actual test data
    y_test = np.random.rand(100)
    y_pred = y_test + np.random.normal(0, 0.05, size=100)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, alpha=0.6)
    ax2.plot([0, 1], [0, 1], "r--")  # perfect prediction line
    ax2.set_xlabel("Actual Violent Crime Rate")
    ax2.set_ylabel("Predicted Violent Crime Rate")
    st.pyplot(fig2)

    # ===== Interactive Sliders =====
    st.write("### 🔍 Explore What-if Scenarios")
    st.markdown("Adjust the sliders to simulate different conditions:")

    # Input sliders
    PctKids2Par = st.slider("Percentage of kids in family housing with two parents (PctKids2Par)", 0.0, 1.0, 0.7)
    PctPersDenseHous = st.slider("Percentage of persons in dense housing (PctPersDenseHous)", 0.0, 1.0, 0.1)
    PctIlleg = st.slider("Percentage of kids born to never married (PctIlleg)", 0.0, 1.0, 0.1)
    NumStreet = st.slider("Number of people living in areas classified as urban (NumStreet)", 0.0, 1.0, 0.1)
    racepctblack = st.slider("Percentage of population that is African American (racepctblack)", 0.0, 1.0, 0.1)
    HousVacant = st.slider("Number of vacant households (HousVacant)", 0.0, 1.0, 0.05)
    MalePctDivorce = st.slider("Percentage of males who are divorced (MalePctDivorce)", 0.0, 1.0, 0.1)
    RentLowQ = st.slider("Lower quartile of rent (RentLowQ)", 0.0, 1.0, 0.2)
    pctWInvInc = st.slider("Percentage of households with investment / rent income in 1989 (pctWInvInc)", 0.0, 1.0, 0.05)

    # Put into DataFrame for prediction
    X_new = pd.DataFrame([[
        PctKids2Par,
        PctPersDenseHous,
        PctIlleg,
        NumStreet,
        racepctblack,
        HousVacant,
        MalePctDivorce,
        RentLowQ,
        pctWInvInc
    ]], columns=[
        "PctKids2Par",
        "PctPersDenseHous",
        "PctIlleg",
        "NumStreet",
        "racepctblack",
        "HousVacant",
        "MalePctDivorce",
        "RentLowQ",
        "pctWInvInc"
    ])

    # Predict button
    if st.button("Predict"):
        pred = model.predict(X_new)[0]
        st.success(f"🔮 Predicted Violent Crime Rate: **{pred:.3f}**")

    # ===== Summary & Recommendations =====
    st.write("### 📝 Summary & Recommendations")
    st.markdown("""
    - **Higher % of kids with two parents → lower violent crime.**  
    - **More vacant households → higher violent crime risk.**  
    - **Higher % of investment income households → lower violent crime.**  
    - **Higher % of persons in dense housing → potential increase in crime risk.**
    """)