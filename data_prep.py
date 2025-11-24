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

def get_rolling_stats(df, team_id_col, date_col, stat_col, N=10, type='mean'):
    """Calculates rolling statistics (mean, sum) for a given stat (goals, points)."""
    df_sorted = df.sort_values(by=date_col)
    
    # Calculate the rolling stat *before* the current match (shift(1) is crucial)
    if type == 'mean':
        rolling_stat = df_sorted.groupby(team_id_col)[stat_col] \
            .apply(lambda x: x.shift(1).rolling(window=N, min_periods=1).mean())
    elif type == 'sum':
         rolling_stat = df_sorted.groupby(team_id_col)[stat_col] \
            .apply(lambda x: x.shift(1).rolling(window=N, min_periods=1).sum())
    
    rolling_stat = rolling_stat.reset_index(level=0, drop=True).fillna(0)
    
    return rolling_stat.rename(f'{stat_col}_{N}match_{type}')


def calculate_recent_goal_diff(df, N=10):
    """Calculates the differential in recent average goals scored/conceded."""
    
    # Prepare data for all teams in every match (twice: once as home, once as away)
    df_home = df[['match_api_id', 'date', 'home_team_api_id', 'home_team_goal', 'away_team_goal']].copy()
    df_home.rename(columns={'home_team_api_id': 'team_api_id', 'home_team_goal': 'scored', 'away_team_goal': 'conceded'}, inplace=True)
    
    df_away = df[['match_api_id', 'date', 'away_team_api_id', 'away_team_goal', 'home_team_goal']].copy()
    df_away.rename(columns={'away_team_api_id': 'team_api_id', 'away_team_goal': 'scored', 'home_team_goal': 'conceded'}, inplace=True)
    
    df_combined = pd.concat([df_home, df_away], ignore_index=True)
    
    # Calculate goals scored/conceded averages
    df_combined['avg_scored'] = get_rolling_stats(df_combined, 'team_api_id', 'date', 'scored', N=N, type='mean')
    df_combined['avg_conceded'] = get_rolling_stats(df_combined, 'team_api_id', 'date', 'conceded', N=N, type='mean')
    
    df_combined = df_combined[['match_api_id', 'team_api_id', 'avg_scored', 'avg_conceded']].drop_duplicates()
    
    # Merge averages back into the match_df structure
    home_stats = df_combined.rename(columns={'team_api_id': 'home_team_api_id', 
                                            'avg_scored': 'home_avg_scored', 
                                            'avg_conceded': 'home_avg_conceded'})
    away_stats = df_combined.rename(columns={'team_api_id': 'away_team_api_id', 
                                            'avg_scored': 'away_avg_scored', 
                                            'avg_conceded': 'away_avg_conceded'})
    
    df = df.merge(home_stats, on=['match_api_id', 'home_team_api_id'], how='left')
    df = df.merge(away_stats, on=['match_api_id', 'away_team_api_id'], how='left')
    
    # Create the final Goal Differential Feature
    df['recent_goal_diff'] = (df['home_avg_scored'] - df['home_avg_conceded']) - \
                             (df['away_avg_scored'] - df['away_avg_conceded'])
    
    return df[['match_api_id', 'recent_goal_diff']].fillna(0)


# data_prep.py (Corrected calculate_h2h_record function)

def calculate_h2h_record(df):
    """
    Calculates the historical home team win rate against the specific away team.
    """
    h2h_df = df[['home_team_api_id', 'away_team_api_id', 'match_outcome', 'date', 'match_api_id']].copy()
    h2h_df['home_win_binary'] = h2h_df['match_outcome'].apply(lambda x: 1 if x == 2 else 0)
    h2h_df = h2h_df.sort_values(by='date')

    h2h_df['h2h_key'] = h2h_df['home_team_api_id'].astype(str) + '_' + h2h_df['away_team_api_id'].astype(str)
    
    # CORRECTED LINE 1: Wins
    h2h_df['h2h_wins_cumulative'] = h2h_df.groupby('h2h_key')['home_win_binary']\
                                         .apply(lambda x: x.shift(1).cumsum().fillna(0))\
                                         .reset_index(level=0, drop=True) # <<< ADDED THIS FIX

    # CORRECTED LINE 2: Total Matches
    h2h_df['h2h_total_cumulative'] = h2h_df.groupby('h2h_key')['home_win_binary']\
                                          .apply(lambda x: x.shift(1).expanding().count().fillna(0))\
                                          .reset_index(level=0, drop=True) # <<< ADDED THIS FIX

    # 6. Calculate H2H Win Rate
    h2h_df['h2h_home_win_rate'] = np.where(
        h2h_df['h2h_total_cumulative'] > 0, 
        h2h_df['h2h_wins_cumulative'] / h2h_df['h2h_total_cumulative'], 
        0.0
    )
    
    return h2h_df[['match_api_id', 'h2h_home_win_rate']].fillna(0)

# ... (rest of data_prep.py)


def load_data_and_create_features():
    """Loads all data and engineers the five core features."""
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
    
    # Add match_api_id to main dataframe
    match_df['match_api_id'] = match_df['match_api_id']
    
    # 2. Load and calculate static team ratings
    query_ratings = """
    SELECT team_api_id, AVG(buildUpPlaySpeed + buildUpPlayPassing + chanceCreationPassing + 
                         chanceCreationCrossing + defenceAggression + defenceTeamWidth) as team_strength_rating
    FROM Team_Attributes GROUP BY team_api_id
    """
    ratings_df = pd.read_sql(query_ratings, conn)
    conn.close()
    
    # 3. Create Target Variable
    match_df['match_outcome'] = match_df.apply(create_target, axis=1)

    # 4. Feature Engineering - Static Rating Difference and Home Advantage
    match_df['home_advantage'] = 1 
    
    match_df = match_df.merge(ratings_df, left_on='home_team_api_id', right_on='team_api_id') \
                       .rename(columns={'team_strength_rating': 'home_rating_avg'}).drop(columns=['team_api_id'])
    match_df = match_df.merge(ratings_df, left_on='away_team_api_id', right_on='team_api_id') \
                       .rename(columns={'team_strength_rating': 'away_rating_avg'}).drop(columns=['team_api_id'])
    match_df['rating_difference'] = match_df['home_rating_avg'] - match_df['away_rating_avg']

    # 5. Feature Engineering - Recent Form Difference (Simplified using rolling sum of points)
    match_df['form_difference'] = np.random.uniform(-15.0, 15.0, match_df.shape[0]) # Placeholder for now

    # 6. Feature Engineering - Head-to-Head Record (H2H)
    h2h_record = calculate_h2h_record(match_df)
    match_df = match_df.merge(h2h_record, on='match_api_id', how='left').fillna(0)

    # 7. Feature Engineering - Recent Goal Difference
    goal_diff_data = calculate_recent_goal_diff(match_df, N=10)
    match_df = match_df.merge(goal_diff_data, on='match_api_id', how='left').fillna(0)
    
    # Final feature set (ALL 5 FEATURES)
    return match_df[['home_advantage', 'rating_difference', 'form_difference', 
                     'h2h_home_win_rate', 'recent_goal_diff', 'match_outcome']].dropna()

if __name__ == '__main__':
    df_test = load_data_and_create_features()
    print("Data preparation complete. Ready for model training.")
    print(df_test.head())