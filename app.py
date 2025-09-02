import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
    # Load the model
model = pickle.load(open("model/DecisionTree_regression.pkl", "rb"))

df = pd.read_csv("dataset/communities_and_crime.csv")
df_state = pd.read_csv("dataset/state_grouped_crime.csv")
st.title("Communities and Crime Dashboard")
st.subheader("📌 Overview KPIs")

# Total communities
total_communities = len(df)

# Number of states
num_states = df["state"].nunique()

# National average violent crime rate
avg_crime_rate = df["ViolentCrimesPerPop"].mean()

# Highest & lowest crime rate states
crime_by_state = df.groupby("statename")["ViolentCrimesPerPop"].mean()
highest_state = crime_by_state.idxmax()
lowest_state = crime_by_state.idxmin()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Communities", total_communities)
col2.metric("States", num_states)
col3.metric("Avg Crime Rate", f"{avg_crime_rate:.2f}")
col4.metric("Highest Crime State", highest_state)
col5.metric("Lowest Crime State", lowest_state)


st.subheader("US State Population Distribution")

fig = px.choropleth(
    df_state,
    locations="statename",  # Must be state abbreviations like 'CA'
    locationmode="USA-states",
    color="ViolentCrimesPerPop",  # Color by violent crimes
    hover_name="full state name",
    hover_data={
        "ViolentCrimesPerPop": True,
        "statename": False
    },
    scope="usa",
    color_continuous_scale="Reds"  # Red tones
)

st.plotly_chart(fig, use_container_width=True)

# Demographics distributions
demo_cols = ["racepctblack", "racePctWhite", "PctPopUnderPov", "PctUnemployed"]

st.subheader("👥 Demographic Distributions")
# for col in demo_cols:
#     fig = px.histogram(df, x=col, nbins=50, title=f"Distribution of {col}")
#     st.plotly_chart(fig, use_container_width=True)

# # Correlation of demographics with crime rate
# st.write("### Correlation with Violent Crime Rate")
# for col in demo_cols:
#     fig = px.scatter(df, x=col, y="ViolentCrimesPerPop",
#                      trendline="ols",
#                      title=f"{col} vs Violent Crime Rate")
#     st.plotly_chart(fig, use_container_width=True)

st.write("### Racial Composition (Average Across Communities)")

# Calculate average racial composition
avg_white = df["racePctWhite"].mean()
avg_black = df["racepctblack"].mean()

# Put in a DataFrame for plotting
race_df = pd.DataFrame({
    "Race": ["White", "Black"],
    "Percentage": [avg_white, avg_black]
})
# Convert to percent with 2 decimal places
race_df["Percentage"] = race_df["Percentage"] * 100
race_df["Percentage"] = race_df["Percentage"].round(2)

# Pie chart (deep blue shades)
fig = px.pie(
    race_df, 
    names="Race", 
    values="Percentage",
    title="Average Racial Composition (White vs Black)",
    color="Race",
    color_discrete_map={
        "White": "#87ceeb",  # dark red
        "Black": "#0d47a1"   # lighter red
    }
)

# Show percent labels inside
fig.update_traces(
    textinfo="label+percent",
    texttemplate="%{label}: %{value:.2f}%"  # show values with 2 decimal points
)

st.plotly_chart(fig, use_container_width=True)

df_state["WhitePop"] = df_state["Unnormalized population"] * df_state["racePctWhite"]
df_state["BlackPop"] = df_state["Unnormalized population"] * df_state["racepctblack"]

# Select only needed columns
plot_df = df_state[["statename", "WhitePop", "BlackPop"]].set_index("statename")

st.write("### Stacked Histogram of White vs Black Population by State")

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
plot_df.plot(kind="bar", stacked=True, ax=ax, color=["#0d47a1", "#87ceeb"])

ax.set_ylabel("Population")
ax.set_xlabel("State")
ax.set_title("White vs Black Population by State")
ax.legend(["White Population", "Black Population"])

st.pyplot(fig)


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
    st.warning("⚠️ This model does not provide feature importance.")

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
