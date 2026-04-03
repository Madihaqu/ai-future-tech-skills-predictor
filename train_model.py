import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Create data directory if it doesn't exist for the example
if not os.path.exists('data'):
    os.makedirs('data')

# Load dataset
try:
    data = pd.read_csv("data/skills_dataset.csv")
except FileNotFoundError:
    print("Error: data/skills_dataset.csv not found. Please ensure the file exists.")
    exit()

X = data[['Year']]
models = {}

# Train a model for each skill column
for skill in data.columns[1:]:
    y = data[skill]
    model = LinearRegression()
    model.fit(X, y)
    models[skill] = model

# Save trained models in the current directory to match app.py
with open("skill_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("Models trained successfully and saved as skill_models.pkl")