import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\chouk\Downloads\ipl_matches.csv")

# Drop irrelevant columns (e.g., Umpires)
df = df.drop(['Umpire1', 'Umpire2'], axis=1)

# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables
label_encoder_team = LabelEncoder()
label_encoder_venue = LabelEncoder()
label_encoder_toss_winner = LabelEncoder()

df['Team1'] = label_encoder_team.fit_transform(df['Team1'])
df['Team2'] = label_encoder_team.fit_transform(df['Team2'])
df['Venue'] = label_encoder_venue.fit_transform(df['Venue'])
df['Toss_Winner'] = label_encoder_toss_winner.fit_transform(df['Toss_Winner'])
df['Toss_Decision'] = df['Toss_Decision'].apply(lambda x: 1 if x == 'field' else 0)
df['Winner'] = label_encoder_team.fit_transform(df['Winner'])

# Additional Feature Engineering
df['Team1_Recent_Form'] = df.groupby('Team1')['Winner'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
df['Team2_Recent_Form'] = df.groupby('Team2')['Winner'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)

# Define features and target
X = df.drop(['Winner', 'Date'], axis=1)
y = df['Winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)

# Streamlit app
st.set_page_config(page_title="IPL Match Winner Prediction", page_icon="🏏", layout="centered")


# Page layout
st.markdown('<div class="container">', unsafe_allow_html=True)

# Title without extra margin
st.markdown('<h1 class="title">IPL Match Winner Prediction</h1>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<h2>Select Match Details:</h2>', unsafe_allow_html=True)

# Team 1 selection
team1_name = st.selectbox("Team 1", label_encoder_team.classes_)

# Team 2 selection
team2_name = st.selectbox("Team 2", label_encoder_team.classes_)

# Venue selection
venue_name = st.selectbox("Venue", label_encoder_venue.classes_)

# Toss winner selection based on Team 1 and Team 2
if team1_name != team2_name:
    toss_winner_options = [team1_name, team2_name]
else:
    toss_winner_options = [team1_name]

toss_winner_name = st.selectbox("Toss Winner", toss_winner_options)

# Toss decision selection
toss_decision_text = st.selectbox("Toss Decision", ['field', 'bat'])

st.markdown('</div>', unsafe_allow_html=True)

# Prediction button with custom CSS class
if st.button("Predict Winner", key="predict_button"):
    # Convert selections to numeric
    team1 = label_encoder_team.transform([team1_name])[0]
    team2 = label_encoder_team.transform([team2_name])[0]
    venue = label_encoder_venue.transform([venue_name])[0]
    toss_winner = label_encoder_team.transform([toss_winner_name])[0]
    toss_decision = 1 if toss_decision_text == 'field' else 0

    # Feature engineering for user input
    team1_recent_form = df[df['Team1'] == team1]['Winner'].rolling(window=3, min_periods=1).sum().iloc[-1]
    team2_recent_form = df[df['Team2'] == team2]['Winner'].rolling(window=3, min_periods=1).sum().iloc[-1]

    # Prepare input data
    input_data = np.array([[team1, team2, venue, toss_winner, toss_decision, team1_recent_form, team2_recent_form]])

    # Make prediction
    prediction = gbc.predict(input_data)
    winner = label_encoder_team.inverse_transform(prediction)

    # Display prediction
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown(f'<h2>The predicted winner is: {winner[0]}</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close container
