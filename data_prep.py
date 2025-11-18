# data_prep.py
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = 'database.sqlite' 

def create_target(row):
    """Determines the match outcome (2: Home Win, 1: Draw, 0: Home Loss)."""
    if row['home_team_goal'] > row['away_team_goal']:
        return 2  # Home Win
    elif row['home_team_goal'] == row['away_team_goal']:
        return 1  # Draw
    else:
        return 0  # Home Loss

def get_recent_form(df, team_id_col, date_col, N=5):
    """
    Calculates the cumulative points a team earned in its N previous matches 
    based on the match date.
    """
    # 1. Create a column for points earned by the team in that match
    df['points'] = df['match_outcome'].apply(lambda x: 3 if x == 2 else (1 if x == 1 else 0)) 
    
    # 2. Adjust points earned for away teams to be calculated from the home-centric outcome
    df['team_points'] = df.apply(
        lambda row: row['points'] if row['home_team_api_id'] == row[team_id_col] else 
                    (3 if row['match_outcome'] == 0 else (1 if row['match_outcome'] == 1 else 0)),
        axis=1
    )
    
    # 3. Sort by team and date and apply rolling sum
    df_sorted = df.sort_values(by=date_col)
    
    # The 'shift(1)' is CRUCIAL: it ensures we only use data *before* the current match
    # Rolling sum calculates form points in the previous N matches
    df_sorted[f'{team_id_col}_form'] = df_sorted.groupby(team_id_col)['team_points'] \
        .rolling(window=N, closed='left').sum().reset_index(level=0, drop=True)
    
    # 4. Fill NaN values (the first N matches for each team) with 0
    df_sorted[f'{team_id_col}_form'] = df_sorted[f'{team_id_col}_form'].fillna(0)
    
    return df_sorted[['match_api_id', f'{team_id_col}_form']].rename(
        columns={f'{team_id_col}_form': f'{team_id_col}_form'}
    )


def load_data_and_create_features():
    """Loads all data, creates the target, and engineers all features."""
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load core match data
    query_match = """
    SELECT 
        id as match_api_id, home_team_api_id, away_team_api_id, 
        home_team_goal, away_team_goal, date
    FROM Match 
    WHERE home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL
    ORDER BY date
    """
    match_df = pd.read_sql(query_match, conn)
    
    # 2. Load and calculate static team ratings (Same as before)
    query_ratings = """
    SELECT 
        team_api_id, 
        AVG(buildUpPlaySpeed + buildUpPlayPassing + 
            chanceCreationPassing + chanceCreationCrossing + 
            defenceAggression + defenceTeamWidth) as team_strength_rating
    FROM Team_Attributes
    GROUP BY team_api_id
    """
    ratings_df = pd.read_sql(query_ratings, conn)
    
    # 3. Create Target Variable
    match_df['match_outcome'] = match_df.apply(create_target, axis=1)

    # 4. Feature Engineering - Static Rating Difference
    match_df['home_advantage'] = 1 
    
    match_df = match_df.merge(
        ratings_df, left_on='home_team_api_id', right_on='team_api_id'
    ).rename(columns={'team_strength_rating': 'home_rating_avg'}).drop(columns=['team_api_id'])

    match_df = match_df.merge(
        ratings_df, left_on='away_team_api_id', right_on='team_api_id'
    ).rename(columns={'team_strength_rating': 'away_rating_avg'}).drop(columns=['team_api_id'])
    
    match_df['rating_difference'] = match_df['home_rating_avg'] - match_df['away_rating_avg']

    # 5. Feature Engineering - Recent Form Difference (New Dynamic Feature)
    form_data = match_df.copy()
    
    # Calculate Home Team Form
    home_form = get_recent_form(form_data, 'home_team_api_id', 'date', N=5)
    match_df = match_df.merge(home_form, on='match_api_id').rename(
        columns={'home_team_api_id_form': 'home_form_points'}
    )
    
    # Calculate Away Team Form
    away_form = get_recent_form(form_data, 'away_team_api_id', 'date', N=5)
    match_df = match_df.merge(away_form, on='match_api_id').rename(
        columns={'away_team_api_id_form': 'away_form_points'}
    )
    
    match_df['form_difference'] = match_df['home_form_points'] - match_df['away_form_points']

    # Final feature set
    return match_df[['home_advantage', 'rating_difference', 'form_difference', 'match_outcome']].dropna()

if __name__ == '__main__':
    # Test the function (optional)
    df_test = load_data_and_create_features()
    print("Data preparation complete. Ready for model training.")
    print(df_test.describe())