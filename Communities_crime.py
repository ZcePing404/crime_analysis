import pandas as pd

communities_dataset = pd.read_excel("communities_and_crime.xlsx")
communities_dataset = communities_dataset.replace("?", 0)
df_columns = communities_dataset.columns


correlation_dict = {}
useless_columns = ["state", "county", "community", "communityname", "fold", "countyname", "ViolentCrimesPerPop"]

for column in df_columns:
    if column in useless_columns:
        continue
    correlation_dict.update({column: communities_dataset[column].corr(communities_dataset['ViolentCrimesPerPop']).item()})

counter = 0

for key, value in correlation_dict.items():
    if value < 0.40:
        continue
    print(key, " : ", value)
    counter += 1

print("Number of features = ", counter)