import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# load dataset
data = pd.read_csv("data/skills_dataset.csv")

X = data[['Year']]

models = {}

for skill in data.columns[1:]:
    y = data[skill]

    model = LinearRegression()
    model.fit(X, y)

    models[skill] = model

# save trained models
with open("skill_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("Models trained successfully")