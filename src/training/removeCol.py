import pandas as pd

df = pd.read_csv("../data/padd_val.csv", index_col=0)
df = df.drop(columns=["Unnamed: 0"])
df.to_csv("../data/padd_val.csv")
print(df.columns)