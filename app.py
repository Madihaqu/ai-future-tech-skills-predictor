import streamlit as st
import pickle
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="AI Future Skills Predictor",
    page_icon="🚀",
    layout="wide"
)

# Dark UI Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
}
h1, h3 {
    color: #38bdf8 !important;
}
.stMetric {
    background-color: rgba(56, 189, 248, 0.1);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Future Skills Predictor")
st.write("Predict which technologies will dominate the future based on historical trends.")

# --- LOAD MODELS ---
model_path = "skill_models.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please run train_model.py first!")
    st.stop()

with open(model_path, "rb") as f:
    models = pickle.load(f)

# --- INPUT & PREDICTION ---
year = st.number_input("Enter Future Year", 2025, 2040, value=2025)

# Calculate results immediately for the UI
results = {}
input_df = pd.DataFrame([[year]], columns=['Year'])

for skill, model in models.items():
    prediction = model.predict(input_df)
    # Ensure demand isn't negative and convert to int
    results[skill] = max(0, int(prediction[0]))

df_results = pd.DataFrame(list(results.items()), columns=["Skill", "Demand"])
top_skills = df_results.sort_values(by="Demand", ascending=False)

# --- MAIN DASHBOARD ---
col_m1, col_m2 = st.columns(2)
col_m1.metric("Selected Year", year)
col_m2.metric("Top Predicted Skill", top_skills.iloc[0]["Skill"])

st.subheader("📊 Predicted Skill Demand")
st.bar_chart(df_results.set_index("Skill"))

# --- HISTORICAL TRENDS ---
st.subheader("📈 Historical Technology Trends")
if os.path.exists("data/skills_dataset.csv"):
    historical_data = pd.read_csv("data/skills_dataset.csv")
    st.line_chart(historical_data.set_index("Year"))
else:
    st.info("Upload data/skills_dataset.csv to see historical trends.")

# --- INDIVIDUAL SKILL FORECAST ---
st.divider()
st.subheader("🔎 Deep Dive: Skill Growth Forecast")

skill_list = list(models.keys())
selected_skill = st.selectbox("Select a Skill to Forecast", skill_list)

# Generate forecast data (2024 to 2035)
future_range = list(range(2024, 2036))
forecast_values = []

for y in future_range:
    val = models[selected_skill].predict(pd.DataFrame([[y]], columns=['Year']))[0]
    forecast_values.append(max(0, int(val)))

forecast_df = pd.DataFrame({
    "Year": future_range,
    "Predicted Demand": forecast_values
})

st.line_chart(forecast_df.set_index("Year"))

if st.button("Get Exact Prediction"):
    specific_demand = results[selected_skill]
    st.success(f"The predicted demand for **{selected_skill}** in **{year}** is **{specific_demand}** units.")

# --- FOOTER ---
st.subheader("🏆 Top 3 Future Technologies")
top3 = top_skills.head(3)
t_col1, t_col2, t_col3 = st.columns(3)
t_col1.metric("Rank 1", top3.iloc[0]["Skill"], f"Score: {top3.iloc[0]['Demand']}")
t_col2.metric("Rank 2", top3.iloc[1]["Skill"], f"Score: {top3.iloc[1]['Demand']}")
t_col3.metric("Rank 3", top3.iloc[2]["Skill"], f"Score: {top3.iloc[2]['Demand']}")