import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Future Skills Predictor",
    page_icon="🚀",
    layout="wide"
)

# Dark UI
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#0f172a,#020617);
color: white;
}

h1 {
text-align: center;
color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 AI Future Skills Predictor")

st.write("Predict which technologies will dominate the future.")

# Input year
year = st.number_input("Enter Future Year", 2025, 2035)

# Load ML models
with open("../model/skill_models.pkl", "rb") as f:
    models = pickle.load(f)

results = {}

# Predict demand
for skill, model in models.items():
    prediction = model.predict([[year]])
    results[skill] = int(prediction[0])

# Show results
st.subheader("📊 Predicted Skill Demand")

df = pd.DataFrame(list(results.items()), columns=["Skill", "Demand"])

st.bar_chart(df.set_index("Skill"))

st.subheader("🔥 Top Future Skills")

top_skills = df.sort_values(by="Demand", ascending=False)

st.table(top_skills)

st.subheader("🤖 AI Insight")

top_skill = top_skills.iloc[0]["Skill"]

st.write(f"Our AI model predicts that **{top_skill}** will have the highest demand in the selected year.")

col1, col2 = st.columns(2)

col1.metric("Predicted Year", year)
col2.metric("Top Skill", top_skills.iloc[0]["Skill"])

st.subheader("📈 Technology Demand Trends")

data = pd.read_csv("data/skills_dataset.csv")

st.line_chart(data.set_index("Year"))

st.subheader("🔎 Check Future Demand for a Skill")

skill_list = list(results.keys())

selected_skill = st.selectbox("Select Skill", skill_list)

if st.button("Predict Demand"):
    demand = results[selected_skill]

    st.success(f"Predicted demand for {selected_skill} in {year}: {demand}")

st.subheader("🔮 Future Skill Growth Prediction")

future_years = list(range(2024, 2036))

forecast = []

for y in future_years:
    pred = models[selected_skill].predict([[y]])[0]
    forecast.append(int(pred))

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Demand": forecast
})

st.line_chart(forecast_df.set_index("Year"))

# Top 3 future technologies
st.subheader("🏆 Top 3 Future Technologies")

top3 = df.sort_values(by="Demand", ascending=False).head(3)

col1, col2, col3 = st.columns(3)

col1.metric(top3.iloc[0]["Skill"], top3.iloc[0]["Demand"])
col2.metric(top3.iloc[1]["Skill"], top3.iloc[1]["Demand"])
col3.metric(top3.iloc[2]["Skill"], top3.iloc[2]["Demand"])

st.line_chart(forecast_df.set_index("Year"))

