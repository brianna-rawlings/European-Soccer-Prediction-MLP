import sqlite3
import pandas as pd

# Path is set to look in the current working directory, which should be 'finalml'
db_path = "database.sqlite" 

try:
    conn = sqlite3.connect(db_path)

    # Query uses the correct table ('Team') and column ('team_long_name')
    team_query = """
    SELECT DISTINCT team_long_name
    FROM Team  
    ORDER BY team_long_name;
    """

    teams_df = pd.read_sql(team_query, conn)
    conn.close()

    print("\n--- List of Team Names for Selection ---")
    # This will print the exact names you need to copy
    print(teams_df['team_long_name'].tolist())
    print("----------------------------------------")

except Exception as e:
    print(f"An unexpected error occurred: {e}")