import pandas as pd

# Load dataset
df = pd.read_csv("dataset/communities_and_crime.csv")

# Aggregate to state level
df_state = df.groupby(["state", "statename", "full state name"], as_index=False).agg({
    "Unnormalized population": "sum",        
    "racePctWhite": "mean",                  
    "racepctblack": "mean",                  
    "ViolentCrimesPerPop": "mean"           
})

df_state = df_state.round(2)

# Reorder columns to match your example
df_state = df_state[[
    "state",
    "statename",
    "full state name",
    "Unnormalized population",
    "racePctWhite",
    "racepctblack",
    "ViolentCrimesPerPop"
]]

# Save to new CSV
df_state.to_csv("dataset/state_summary.csv", index=False)

