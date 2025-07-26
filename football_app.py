import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Dummy match data (You can replace with real-time API data)
match_data = {
    "teams": ["Real Madrid vs Barcelona", "Man City vs Liverpool", "PSG vs Bayern"],
    "score": ["2 - 1", "1 - 1", "0 - 2"],
    "status": ["Finished", "Ongoing", "Finished"],
    "possession": [("55%", "45%"), ("48%", "52%"), ("40%", "60%")],
    "shots": [(12, 8), (10, 11), (5, 14)],
    "xG": [(1.8, 1.1), (1.4, 1.5), (0.9, 2.0)]
}

# Simulated AI prediction function
def ai_predict(xg1, xg2, shots1, shots2, poss1, poss2):
    X = pd.DataFrame({
        "xG_diff": [xg1 - xg2],
        "shots_diff": [shots1 - shots2],
        "poss_diff": [poss1 - poss2]
    })
    model = RandomForestClassifier()
    # Simulated training (normally you'd train with real match history)
    np.random.seed(42)
    dummy_X = np.random.randn(100, 3)
    dummy_y = np.random.choice([0, 1, 2], 100)  # 0: Team A win, 1: Draw, 2: Team B win
    model.fit(dummy_X, dummy_y)
    pred = model.predict(X)[0]
    return ["Team A Wins", "Draw", "Team B Wins"][pred]

# App UI
st.title("⚽ Football Live Scores with AI Insights")

selected_match = st.selectbox("Select a match", match_data["teams"])
match_index = match_data["teams"].index(selected_match)

teamA, teamB = selected_match.split(" vs ")
score = match_data["score"][match_index]
status = match_data["status"][match_index]
possession = match_data["possession"][match_index]
shots = match_data["shots"][match_index]
xg = match_data["xG"][match_index]

st.subheader(f"{selected_match}")
st.write(f"**Score:** {score} | **Status:** {status}")
st.write(f"**Possession:** {teamA}: {possession[0]} | {teamB}: {possession[1]}")
st.write(f"**Shots:** {teamA}: {shots[0]} | {teamB}: {shots[1]}")
st.write(f"**Expected Goals (xG):** {teamA}: {xg[0]} | {teamB}: {xg[1]}")

# Convert percentages to numbers
poss1 = int(possession[0].replace("%", ""))
poss2 = int(possession[1].replace("%", ""))

# Prediction
st.markdown("### 🧠 AI Match Winner Prediction")
prediction = ai_predict(xg[0], xg[1], shots[0], shots[1], poss1, poss2)
st.success(f"Predicted Result: **{prediction}**")

# Visualization
st.markdown("### 📊 Match Analytics")

df_plot = pd.DataFrame({
    "Metric": ["Shots", "xG", "Possession"],
    teamA: [shots[0], xg[0], poss1],
    teamB: [shots[1], xg[1], poss2]
}).set_index("Metric")

st.bar_chart(df_plot)

