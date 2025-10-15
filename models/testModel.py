import pandas as pd
import numpy as np
import os

csv_path = '../ASL_Citizen/landmarks_2/JEWISH.csv'

label = os.path.splitext(os.path.basename(csv_path))[0]
df = pd.read_csv(csv_path)
data = df.select_dtypes(include=[np.number]).to_numpy()

print(label, data.shape)