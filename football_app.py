import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Football Score Predictions", layout="wide")

# App title and description
st.title("Football Score Predictions")
st.write("A simple app that provides AI-powered football match predictions.")

# Sample teams data
teams = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool", 
    "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest", 
    "Southampton", "Tottenham", "West Ham", "Wolverhampton"
]

# Team strength ratings (1-10)
team_ratings = {
    team: random.randint(70, 95) for team in teams
}

# Function to generate match prediction
def predict_match(home_team, away_team):
    """
    Generate a match prediction based on team ratings and home advantage.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        
    Returns:
        Dictionary with prediction details
    """
    # Get team ratings
    home_rating = team_ratings.get(home_team, 80)
    away_rating = team_ratings.get(away_team, 80)
    
    # Add home advantage (5% boost)
    home_advantage = 5
    effective_home_rating = home_rating + home_advantage
    
    # Calculate win probabilities
    total_rating = effective_home_rating + away_rating
    home_win_prob = effective_home_rating / total_rating
    away_win_prob = away_rating / total_rating
    draw_prob = 1 - (home_win_prob * 0.8) - (away_win_prob * 0.8)  # Adjust to make draws possible
    
    # Normalize probabilities
    total_prob = home_win_prob + away_win_prob + draw_prob
    home_win_prob /= total_prob
    away_win_prob /= total_prob
    draw_prob /= total_prob
    
    # Generate score prediction
    home_expected_goals = (effective_home_rating / 20) * random.uniform(0.8, 1.2)
    away_expected_goals = (away_rating / 20) * random.uniform(0.8, 1.2)
    
    # Round to whole numbers for the most likely score
    home_score = max(0, round(home_expected_goals))
    away_score = max(0, round(away_expected_goals))
    
    # Generate alternative scores
    alt_scores = [
        (home_score, away_score),  # Most likely
        (home_score + 1, away_score),
        (home_score, away_score + 1),
        (home_score - 1 if home_score > 0 else 0, away_score),
        (home_score, away_score - 1 if away_score > 0 else 0)
    ]
    
    # Calculate result probabilities
    result_probs = {
        "Home Win": round(home_win_prob * 100, 1),
        "Draw": round(draw_prob * 100, 1),
        "Away Win": round(away_win_prob * 100, 1)
    }
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_score": f"{home_score} - {away_score}",
        "alternative_scores": alt_scores,
        "result_probabilities": result_probs,
        "home_rating": home_rating,
        "away_rating": away_rating,
        "analysis": generate_analysis(home_team, away_team, home_rating, away_rating, result_probs)
    }

# Function to generate match analysis
def generate_analysis(home_team, away_team, home_rating, away_rating, result_probs):
    """
    Generate a text analysis of the match prediction.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        home_rating: Rating of the home team
        away_rating: Rating of the away team
        result_probs: Dictionary of result probabilities
        
    Returns:
        String with match analysis
    """
    # Determine team form descriptions
    home_form = "excellent" if home_rating > 90 else "good" if home_rating > 80 else "average" if home_rating > 70 else "poor"
    away_form = "excellent" if away_rating > 90 else "good" if away_rating > 80 else "average" if away_rating > 70 else "poor"
    
    # Determine match difficulty
    rating_diff = abs(home_rating - away_rating)
    if rating_diff < 5:
        match_desc = "This looks to be a very close match between evenly matched teams."
    elif rating_diff < 10:
        stronger_team = home_team if home_rating > away_rating else away_team
        match_desc = f"{stronger_team} has a slight edge in this contest."
    else:
        stronger_team = home_team if home_rating > away_rating else away_team
        weaker_team = away_team if home_rating > away_rating else home_team
        match_desc = f"{stronger_team} is strongly favored against {weaker_team}."
    
    # Generate analysis text
    analysis = f"{home_team} is in {home_form} form, while {away_team} is showing {away_form} form. "
    analysis += match_desc + " "
    
    # Add probability insight
    most_likely = max(result_probs, key=result_probs.get)
    analysis += f"Our AI model suggests a {most_likely} is the most likely outcome at {result_probs[most_likely]}%."
    
    return analysis

# Sidebar for team selection
st.sidebar.header("Select Teams")

home_team = st.sidebar.selectbox("Home Team", teams, index=0)
away_team = st.sidebar.selectbox("Away Team", teams, index=1)

# Prevent same team selection
if home_team == away_team:
    st.error("Please select different teams for home and away.")
else:
    # Generate prediction when button is clicked
    if st.sidebar.button("Generate Prediction"):
        with st.spinner("AI analyzing match data..."):
            # Simulate AI processing time
            import time
            time.sleep(1)
            
            # Get prediction
            prediction = predict_match(home_team, away_team)
            
            # Display match header
            st.header(f"{home_team} vs {away_team}")
            
            # Display prediction in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Match Prediction")
                st.markdown(f"*Predicted Score:* {prediction['predicted_score']}")
                
                st.markdown("*Alternative Scores:*")
                for score in prediction['alternative_scores']:
                    st.write(f"{score[0]} - {score[1]}")
            
            with col2:
                st.subheader("Outcome Probabilities")
                for outcome, prob in prediction['result_probabilities'].items():
                    st.markdown(f"{outcome}:** {prob}%")
                
                # Create a simple chart for probabilities
                probs_df = pd.DataFrame({
                    'Outcome': list(prediction['result_probabilities'].keys()),
                    'Probability': list(prediction['result_probabilities'].values())
                })
                st.bar_chart(probs_df.set_index('Outcome'))
            
            with col3:
                st.subheader("Team Ratings")
                st.markdown(f"{home_team}:** {prediction['home_rating']} (Home advantage included)")
                st.markdown(f"{away_team}:** {prediction['away_rating']}")
                
                # Rating comparison
                ratings_df = pd.DataFrame({
                    'Team': [home_team, away_team],
                    'Rating': [prediction['home_rating'] + 5, prediction['away_rating']]
                })
                st.bar_chart(ratings_df.set_index('Team'))
            
            # Display analysis
            st.subheader("AI Analysis")
            st.info(prediction['analysis'])
            
            # Disclaimer
            st.caption("Disclaimer: These predictions are for entertainment purposes only and are generated using a simplified model.")

# Add upcoming fixtures section
st.sidebar.divider()
st.sidebar.header("Upcoming Fixtures")

# Generate some random fixtures
def generate_fixtures(num_fixtures=5):
    """
    Generate random upcoming fixtures.
    
    Args:
        num_fixtures: Number of fixtures to generate
        
    Returns:
        List of fixture dictionaries
    """
    fixtures = []
    used_teams = set()
    
    # Start date for fixtures (next Saturday)
    today = datetime.now()
    next_saturday = today + timedelta(days=(5 - today.weekday() + 7) % 7)
    
    for i in range(num_fixtures):
        # Find teams not yet used
        available_teams = [team for team in teams if team not in used_teams]
        
        # If we don't have enough teams, reset the used teams
        if len(available_teams) < 2:
            used_teams = set()
            available_teams = teams
        
        # Select random teams
        home = random.choice(available_teams)
        used_teams.add(home)
        available_teams = [team for team in available_teams if team != home]
        away = random.choice(available_teams)
        used_teams.add(away)
        
        # Generate fixture date (Saturday + i weeks)
        fixture_date = next_saturday + timedelta(days=i*2)
        
        fixtures.append({
            "home": home,
            "away": away,
            "date": fixture_date.strftime("%d %b %Y"),
            "time": f"{random.choice([12, 15, 17, 20])}:00"
        })
    
    return fixtures

# Display fixtures
fixtures = generate_fixtures()
for fixture in fixtures:
    st.sidebar.markdown(f"{fixture['date']} at {fixture['time']}")
    st.sidebar.write(f"{fixture['home']} vs {fixture['away']}")
    if st.sidebar.button(f"Predict {fixture['home']} vs {fixture['away']}", key=f"{fixture['home']}_{fixture['away']}"):
        # Set the selectboxes to these teams
        st.session_state.home_team = fixture['home']
        st.session_state.away_team = fixture['away']
        st.rerun()

# Footer
st.divider()
st.caption("Football Score Predictions - Powered by AI Insights")
