import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import numpy as np
import pickle
    # Load the model
model = pickle.load(open("model/DecisionTree_regression.pkl", "rb"))

df = pd.read_csv("dataset/communities_and_crime.csv")
df_state = pd.read_csv("dataset/state_summary.csv")
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


df_state["Unnormalized population %"] = (
    df_state["population"] / df_state["population"].sum() * 100
).round(2)

st.subheader("US State Population Distribution")
fig_population = px.choropleth(
    df_state,
    locations="statename",           
    locationmode="USA-states",       
    color="population",        
    hover_name="full state name",
    hover_data={
        "statename": False,        
        "population": ":,.0f",
        "Unnormalized population %": ":.2f",
    },
    scope="usa",                     
    color_continuous_scale="Blues",   
    title="State Population Choropleth"
)

fig_crime = px.choropleth(
    df_state,
    locations="statename", 
    locationmode="USA-states",
    color="ViolentCrimesPerPop", 
    hover_name="full state name",
    hover_data={
        "ViolentCrimesPerPop": True,
        "statename": False
    },
    scope="usa",
    color_continuous_scale="Reds",
    title="Crime Distribution Choropleth"
)

st.plotly_chart(fig_population, use_container_width=True)
st.plotly_chart(fig_crime, use_container_width=True)

st.subheader("👥 Demographic Distributions")
st.subheader("Bar Chart of Normalized White Percentages by State")

fig1 = px.bar(
    df_state.sort_values("racePctWhite", ascending=False),
    hover_name="full state name",
    x="statename",
    y="racePctWhite",
    color="racePctWhite",
    color_continuous_scale="Blues",
    title="Percentage of Caucasian by State (racePctWhite)"
)
fig1.update_layout(xaxis_title="State", yaxis_title="Normalized White %")

st.plotly_chart(fig1, use_container_width=True)


st.subheader("Bar Chart of Normalized Black Percentages by State")
fig2 = px.bar(
    df_state.sort_values("racepctblack", ascending=False),
    hover_name="full state name",
    x="statename",
    y="racepctblack",
    color="racepctblack",
    color_continuous_scale="Blues",
    title="Percentage of African American by State (racepctblack)"
)
fig2.update_layout(xaxis_title="State", yaxis_title="Normalized Black %")

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Density of racePctWhite vs racepctblack")
x_white = df["racePctWhite"]
x_black = df["racepctblack"]

kde_white = gaussian_kde(x_white)
kde_black = gaussian_kde(x_black)

x_range = np.linspace(0, max(x_white.max(), x_black.max()), 200)

# Create traces
trace_white = go.Scatter(
    x=x_range,
    y=kde_white(x_range),
    fill='tozeroy',             # fill area under the curve
    fillcolor='rgba(13,71,161,0.5)',  # blue with 50% transparency
    line=dict(color='rgba(13,71,161,1)'),
    name='racePctWhite'
)

trace_black = go.Scatter(
    x=x_range,
    y=kde_black(x_range),
    fill='tozeroy', 
    fillcolor='rgba(135,206,235,0.5)',  # light blue / purple-like color
    line=dict(color='rgba(135,206,235,1)'),
    name='racepctblack'
)

# Combine traces
fig = go.Figure([trace_white, trace_black])
fig.update_layout(
    title="Density of racePctWhite vs racepctblack",
    xaxis_title="Percentage (normalized)",
    yaxis_title="Density",
    template="simple_white"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Socioeconomic Indicators")

socio_cols = ["PctIlleg", "PctUnemployed", "PctPopUnderPov", "PctKids2Par"]
socio_description = ["Percentage of Kids Born to Never Married",
                     "Percentage of People Unemployed", 
                     "Percentage of People Under the Poverty Level", 
                     "Percentage of Kids in with Two Parents"]

for col, desc in zip(socio_cols, socio_description):
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        title=f"Distribution of {desc} ({col})",
        color_discrete_sequence=["#1f77b4"],
        opacity=0.7
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))  # bar borders
    fig.update_layout(
        bargap=0.05,
        xaxis_title=col,
        yaxis_title="Count",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


st.subheader("🏠 Housing Conditions")

housing_cols = ["HousVacant", "PctPersDenseHous"]
housing_description = ["Number of Vacant Households", 
                     "Percent of Persons in Dense Housing"]

for col, desc in zip(housing_cols, housing_description):
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        title=f"Distribution of {desc} ({col})",
        color_discrete_sequence=["#1f77b4"],
        opacity=0.7
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="black")))
    fig.update_layout(
        bargap=0.05,
        xaxis_title=col,
        yaxis_title="Count",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Model Insights")
st.write("### Feature Importance")
if hasattr(model, 'feature_importances_'):
    try:
        feature_names = [
            "PctPersDenseHous", "PctKids2Par", "HousVacant", "racepctblack", 
            "PctIlleg"
        ]
        importances = model.feature_importances_
        print(importances)
        fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi = fi.sort_values("Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(data=fi, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Could not compute feature importance: {str(e)}")
else:
    st.warning("⚠️ This model does not provide feature importance.")


st.write("### 🔍 Explore What-if Scenarios")
st.markdown("Adjust the sliders to simulate different conditions:")

# Input sliders
PctKids2Par = st.slider("Percentage of kids in family housing with two parents (PctKids2Par)", 0.0, 1.0, 0.7)
PctPersDenseHous = st.slider("Percentage of persons in dense housing (PctPersDenseHous)", 0.0, 1.0, 0.1)
PctIlleg = st.slider("Percentage of kids born to never married (PctIlleg)", 0.0, 1.0, 0.1)
racepctblack = st.slider("Percentage of population that is African American (racepctblack)", 0.0, 1.0, 0.1)
HousVacant = st.slider("Number of vacant households (HousVacant)", 0.0, 1.0, 0.05)

# Put into DataFrame for prediction
X_new = pd.DataFrame([[
    PctPersDenseHous,
    PctKids2Par,
    HousVacant,
    racepctblack,
    PctIlleg,
]], columns=[
    "PctPersDenseHous",
    "PctKids2Par",
    "HousVacant",
    "racepctblack",
    "PctIlleg",
])

if st.button("Predict"):
    pred = model.predict(X_new)[0]
    st.success(f"🔮 Predicted Violent Crime Rate: **{pred * 100:.2f}%**")

st.write("### 📝 Summary & Recommendations")
st.markdown("""
- **Higher % of kids in two-parent households (PctKids2Par) → lower violent crime risk.**  
- **Higher % of persons in dense housing (PctPersDenseHous) → potential increase in violent crime.**  
- **Higher % of kids born to unmarried parents (PctIlleg) → associated with higher violent crime risk.**  
- **Higher % of African American population (racepctblack) → may correlate with crime in the dataset, but be cautious as this may reflect systemic bias rather than true causation.**  
- **More vacant households (HousVacant) → higher violent crime risk.**  
""")