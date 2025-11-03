import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv("../data/Finished_Output.csv")

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output= False)
ohetransform = ohe.fit_transform(df[["Gloss"]]) #remove head later this is just for testing!
feature_names = [name.replace('Gloss_', '') for name in ohe.get_feature_names_out(['Gloss'])]
encoded_df = pd.DataFrame(ohetransform, columns=feature_names)
df = df[["Video file"]].join(encoded_df)

df.to_csv('onehot.csv')
