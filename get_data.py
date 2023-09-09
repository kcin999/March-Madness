"""Scrapes Sports Reference and KenPom for NCAA statistical data"""
import os
import logging
import time
import sqlite3
import requests
import pandas as pd
import toml
import system

# Logging setup
logging.basicConfig(
    filename='warning.log',
    encoding='utf-8',
    level=logging.DEBUG
)

with open('config.toml', 'r') as f:
    config = toml.load(f)

def main():
    """Main Function"""
    system.createFolder(config['FOLDERS']['SEASONS_FOLDER'])
    user_answer = input("Would you like to download sports reference data? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(config['YEARS']['GAMELOG_STATS_YEAR'], config['YEARS']['MAX_YEAR'] + 1):
            # School Base Data
            print(f"Getting School Base Data {year}")
            season_data_df = get_season_data_from_sports_reference(year, 'school')
            season_data_df.to_csv(config['FILE']['SCHOOL_BASIC_SEASON_STATS_FILE'].format(year), index=False)
            time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

            # School Advanced Data
            print(f"Getting School Advanced Data {year}")
            df = get_advanced_season_data_from_sports_reference(year, 'school')
            df.to_csv(config['FILE']['SCHOOL_ADV_SEASON_STATS_FILE'].format(year), index=False)
            time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

            if year >= config['YEARS']['MIN_OPPONENT_STATS_YEAR']:
                # Opponent Base Data
                print(f"Getting Opponent Base Data {year}")
                df = get_season_data_from_sports_reference(year, 'opponent')
                df.to_csv(config['FILE']['OPPONENT_BASIC_SEASON_STATS_FILE'].format(year), index=False)
                time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

                # Opponent Advanced Data
                print(f"Getting Opponent Advanced Data {year}")
                df = get_advanced_season_data_from_sports_reference(year, 'opponent')
                df.to_csv(config['FILE']['OPPONENT_ADV_SEASON_STATS_FILE'].format(year), index=False)
                time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

            # Rating Data
            print(f"Getting Rating Data {year}")
            df = get_ratings_from_sports_reference(year)
            df.to_csv(config['FILE']['SPORTS_REFERENCE_RATINGS_FILE'].format(year), index=False)
            time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

            # Team Specific Data
            system.createFolder(config['FOLDERS']['SCHEDULE_FOLDER'].format(year))
            schedule_df = pd.DataFrame()
            gamelog_df = pd.DataFrame()
            for team in season_data_df['School'].to_list():
                print(f"Getting Schedule Data {team}, {year}")
                df = get_schedule_data_from_sports_reference(year, team)
                schedule_df = pd.concat([schedule_df, df])

                time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

                if year >= config['YEARS']['GAMELOG_STATS_YEAR']:
                    print(f"Getting Gamelog Data {team}, {year}")
                    df = get_game_logs_from_sports_reference(year, team)
                    gamelog_df = pd.concat([gamelog_df, df])
                    time.sleep(60 / config['OTHER']['SPORTS_REFERENCE_CALLS_PER_MINUTE'])

            schedule_df.to_csv(config['FILE']['YEAR_SCHEDULE_FILE'].format(year), index=False)

            if year >= config['YEARS']['GAMELOG_STATS_YEAR']:
                gamelog_df.to_csv(config['FILES']['YEAR_GAME_LOG_FILE'].format(year), index=False)


    user_answer = input("Would you like to get ranking data from KenPom? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(max(config['YEARS']['MIN_KENPOM_YEAR'], config['YEARS']['MIN_YEAR']), config['YEARS']['MAX_YEAR'] + 1):
            system.createFolder(config['FOLDERS']['SCHEDULE_FOLDER'].format(year))
            df = get_kenpom_data(year)
            df.to_csv(config['FILES']['KENPOM_FILE'].format(year), index=False)
            time.sleep(60/19)


    user_answer = input("Would you like to add from the CSVs into a sqlite database? (y/n) ")
    if user_answer.upper() in ["Y"]:
        # SQLITE connection
        con = sqlite3.connect(config['FILES']['DATABASE_FILE'])
        try:
            for year in range(config['YEARS']['MIN_YEAR'], config['YEARS']['MAX_YEAR'] + 1):
                add_data_to_db(year, con)

            # Team Mapping
            team_mapping_df = pd.read_csv('team_mapping.csv')
            team_mapping_df.to_sql('team_mapping', con, if_exists='append', index=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError("All modes must have run in order to build a correct sqlite file.") from exc

    print("Program has finished")


def get_team_list(year: int) -> pd.Series:
    """Gets the list of teams from the season.csv

    :param year: Year to check
    :type year: int

    :return: List of Teams
    :rtype: pd.Series
    """
    df = pd.read_csv(f'./Stats/Seasons/{year}_season_stats.csv')

    return df.School


def get_season_data_from_sports_reference(year: int, page: str) -> pd.DataFrame:
    """Gets the season data for a given year

    :param year: Year to get data for
    :type year: int

    :return: DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    if page.upper() == 'SCHOOL':
        url = config['URLS']['SCHOOL_STATS_URL']
        table_id = 'basic_school_stats'
    elif page.upper() == 'OPPONENT':
        url = config['URLS']['OPPONENT_SCHOOL_STATS_URL']
        table_id = 'basic_opp_stats'
    else:
        raise ValueError(f"Page Parameter, '{page}', is not known")
    # Gets Data
    response = requests.get(url.format(year), timeout=None)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': table_id
        }
    )[0]

    # Combines Columns into One Level
    df.columns = df.columns.map(' '.join).str.strip('|')

    # Renames Columns using regex
    df.columns = df.columns.str.replace(
        r'Unnamed: \d_level_\d ',
        '',
        regex=True
    )

    # Drop columns that have all blanks
    df = df.dropna(axis=1, how='all')

    # Removes rows that are in the middle of the table
    df = df[(df.Rk != 'Rk') & (pd.notna(df.Rk))]

    # If School column contains NCAA, then they made the tournament.
    # This creates a new column, if it contains it, and then it drops the original
    df['MadeTournament'] = df['School'].str.contains('NCAA')

    # Drops NCAA from row value
    df['School'] = df['School'].str.rstrip(' NCAA')

    # Drops Rank Column since it wasn't really helpful
    df = df.drop(columns=[
        'Rk'
    ])

    # Returns CSV
    return df


def get_advanced_season_data_from_sports_reference(year: int, page: str) -> pd.DataFrame:
    """Gets the advanced season data for a given year

    :param year: Year to get data for
    :type year: int

    :return: DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    if page.upper() == 'SCHOOL':
        url = config['URLS']['SCHOOL_ADVANCED_STATS_URL']
        table_id = 'adv_school_stats'
    elif page.upper() == 'OPPONENT':
        url = config['URLS']['OPPONENT_ADVANCED_STATS_URL']
        table_id = 'adv_opp_stats'
    else:
        raise ValueError(f"Page Parameter, '{page}', is not known")

    # Gets Data
    response = requests.get(url.format(year), timeout=None)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': table_id
        }
    )[0]

    # Combines Columns into One Level
    df.columns = df.columns.map(' '.join).str.strip('|')

    # Renames Columns using regex
    df.columns = df.columns.str.replace(
        r'Unnamed: \d_level_\d ',
        '',
        regex=True
    )

    # Drop columns that have all blanks
    df = df.dropna(axis=1, how='all')

    # Removes rows that are in the middle of the table
    df = df[(df.Rk != 'Rk') & (pd.notna(df.Rk))]

    # Drops NCAA from row value
    df['School'] = df['School'].str.rstrip(' NCAA')

    # Drops Rank Column since it wasn't really helpful
    df = df.drop(columns=[
        'Rk',
    ])

    # Returns CSV
    return df


def get_ratings_from_sports_reference(year: int):
    """Gets the ratings from sports reference

    :param year: Year to get the data for
    :type year: int
    :return: _description_
    :rtype: pd.Dataframe
    """
    # Gets Data
    response = requests.get(config['URLS']['SPORTS_REFERENCE_RATINGS_URL'].format(year), timeout=None)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': 'ratings'
        }
    )[0]

    # Combines Columns into One Level
    df.columns = df.columns.map(' '.join).str.strip('|')

    # Renames Columns using regex
    df.columns = df.columns.str.replace(
        r'Unnamed: (\d+)_level_\d ',
        '',
        regex=True
    )

    # Drop columns that have all blanks
    df = df.dropna(axis=1, how='all')

    # Removes rows that are in the middle of the table
    df = df[(df.Rk != 'Rk') & (pd.notna(df.Rk))]


    # Drops Rank Column since it wasn't really helpful
    df = df.drop(columns=[
        'Rk',
        'AP Rank'
    ], errors='ignore')

    # Returns CSV
    return df


def format_team_name(team: str) -> str:
    """Formats the team name to match what would be in sports-reference

    :param team: Team String
    :type team: str
    :return: Formatted Team Name for Sports-Reference URL
    :rtype: str
    """
    # Format team name
    team_name = team.replace(' ', '-').lower()
    team_name = team_name.replace('&', '')
    team_name = team_name.replace('(', '')
    team_name = team_name.replace(')', '')
    team_name = team_name.replace("'", '')
    team_name = team_name.replace('.', '')

    # Error for Teams appearing as different names
    if team_name in config['TEAM_MATCHING']:
        team_name = config['TEAM_MATCHING'][team_name]

    return team_name


def get_schedule_data_from_sports_reference(year: int, team: str) -> pd.DataFrame:
    """Gets the schedule information from sports reference

    :param year: Year to get the data for
    :type year: int
    :param team: Team to get the data for
    :type team: str

    :return:  DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    team_df = pd.DataFrame()
    if os.path.exists(config['FILES']['TEAM_SCHEDULE_FILE'].format(year, team)):
        return team_df

    team_name = format_team_name(team)
    # Gets Data
    response = requests.get(
        config['URLS']['SCHEDULE_URL'].format(team_name, year),
        timeout=None
    )

    if response.status_code == 404:
        logging.warning(
            "Could not find address (%s) for %s",
            config['URLS']['SCHEDULE_URL'].format(team_name, year),
            team
        )
        return team_df
    elif response.status_code != 200:
        logging.warning(
            "Status Code: %s return for URL: %s",
            response.status_code, config['URLS']['SCHEDULE_URL'].format(team_name, year)
        )

    # Reads Data into DataFrame
    try:
        team_df = pd.read_html(
            response.text,
            attrs={
                'id': 'schedule'
            }
        )[0]
    except ImportError:
        logging.warning(
            "Error for %s. URL: %s",
            team,
            config['URLS']['SCHEDULE_URL'].format(team_name, year)
        )
        return team_df

    team_df['Team 1'] = team


    # Renames Columns
    team_df = team_df.rename(columns={
        'Opponent': 'Team 2',
        'Tm': 'Team 1 Score',
        'Opp': 'Team 2 Score'
    })

    # Clean Team Names
    team_df['Team 1'] = team_df['Team 1'].str.replace(
        r' \(\d*\)',
        '',
        regex=True
    )
    team_df['Team 2'] = team_df['Team 2'].str.replace(
        r' \(\d*\)',
        '',
        regex=True
    )

    # Drops row that contains header information
    team_df = team_df[(team_df['Streak'] != 'Streak')]

    # Changes Streak to numerics
    team_df['Streak'] = team_df['Streak'].str.replace('W ', '')
    team_df['Streak'] = team_df['Streak'].str.replace('L ', '-')
    team_df['Streak'] = pd.to_numeric(team_df['Streak'])
    team_df['Streak'] = pd.to_numeric(team_df['Streak'])

    # Results
    # True -> Team 1 won the game
    # False -> Team 2 won the game
    team_df['Result'] = team_df['Team 1 Score'] > team_df['Team 2 Score']

    team_df.to_csv(config['FILES']['TEAM_SCHEDULE_FILE'].format(year, team), index=False)

    return team_df


def get_game_logs_from_sports_reference(year: int, team: str) -> pd.DataFrame:
    """Gets the game log information from sports reference

    :param year: Year to get the data for
    :type year: int
    :param team: Team to get the data for
    :type team: str

    :return:  DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    team_df = pd.DataFrame()

    if os.path.exists(config['FILES']['TEAM_GAME_LOG_FILE'].format(year, team)):
        return team_df

    team_name = format_team_name(team)

    # Gets Data
    response = requests.get(
        config['URLS']['SCHOOL_GAME_LOG_URL'].format(team_name, year),
        timeout=None
    )

    if response.status_code == 404:
        logging.warning(
            "Could not find address (%s) for %s",
            config['URLS']['SCHOOL_GAME_LOG_URL'].format(team_name, year),
            team
        )
        return team_df
    elif response.status_code != 200:
        logging.warning(
            "Status Code: %s return for URL: %s",
            response.status_code, config['URLS']['SCHOOL_GAME_LOG_URL'].format(team_name, year)
        )

    # Reads Data into DataFrame
    try:
        team_df = pd.read_html(
            response.text,
            attrs={
                'id': 'sgl-basic_NCAAM'
            }
        )[0]
    except ImportError:
        logging.warning(
            "Error for %s. URL: %s",
            team,
            config['URLS']['SCHOOL_GAME_LOG_URL'].format(team_name, year)
        )
        return team_df

    # Combines Columns into One Level
    team_df.columns = team_df.columns.map(' '.join).str.strip('|')

    # Renames Columns using regex
    team_df.columns = team_df.columns.str.replace(
        r'Unnamed: \d_level_\d ',
        '',
        regex=True
    )

    # Drop columns that have all blanks
    team_df = team_df.dropna(axis=1, how='all')

    # Removes rows that are in the middle of the table
    team_df = team_df[(team_df.G != 'G') & (pd.notna(team_df.G))]

    team_df['Team 1'] = team


    # Renames Columns
    team_df.columns = ['G','Date','Type', 'Team 2', 'W/L', 'Team 1 Score', 'Team 2 Score'] + team_df.columns.to_list()[7:]


    # Clean Team Names
    team_df['Team 1'] = team_df['Team 1'].str.replace(
        r' \(\d*\)',
        '',
        regex=True
    )
    team_df['Team 2'] = team_df['Team 2'].str.replace(
        r' \(\d*\)',
        '',
        regex=True
    )

    # Results
    # True -> Team 1 won the game
    # False -> Team 2 won the game
    team_df['Result'] = team_df['Team 1 Score'] > team_df['Team 2 Score']

    team_df.to_csv(config['FILES']['TEAM_GAME_LOG_FILE'].format(year, team), index=False)


    return team_df


def remove_duplicate_games(year: int):
    """Removes Duplicate Games from the schedules

    :param year: _description_
    :type year: int
    """
    df = pd.read_csv(config['FILE']['YEAR_SCHEDULE_FILE'].format(year))


    df = df.merge(
        df[['Team 2', 'Team 1', 'Date', 'Streak']],
        how='left',
        left_on=['Team 1', 'Team 2', 'Date'],
        right_on=['Team 2', 'Team 1', 'Date'],
        suffixes=[None, '_joined']
    )

    df = df.drop(columns=['Team 2_joined', 'Team 1_joined'])
    df = df.rename(columns={
        'Streak_joined': 'Team 2 Streak',
        'Streak': 'Team 1 Streak'
    })

    # Idea from here
    # https://stackoverflow.com/questions/51182228/python-delete-duplicates-in-a-dataframe-based-on-two-columns-combinations
    df = df[~df[['Date', 'Team 1', 'Team 2']].apply(frozenset, axis=1).duplicated()]
    df.to_csv(config['FILE']['YEAR_SCHEDULE_FILE'].format(year), index=False)


def get_kenpom_data(year: int) -> pd.DataFrame:
    """Scrapes KenPom data for advanced analytics

    :param year: Year to grab
    :type year: int
    """
    # Gets Data
    # Set Custom Headers to be able to work
    headers = {'User-Agent': 'PostmanRuntime/7.29.2'}
    response = requests.get(config['URLS']['KENPOM_URL'].format(year), timeout=None, headers = headers)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': 'ratings-table'
        }
    )[0]

    # Removes all headers but the top 2 and combines them
    df.columns = [col[0] + '_' + col[1] for col in df.columns]

    df.columns = df.columns.str.replace(r'Unnamed: \d+_level_0_', '', regex=True)

    df = df.dropna(how='any')

    df['Team'] = df['Team'].str.replace(r' \d+', '', regex=True)
    df['Team'] = df['Team'].str.replace('*', '', regex=True)

    # Removes small ranking columns
    df = df.loc[:,~df.columns.duplicated()].copy()

    # Removes rank columns
    df = df[(df.Rk != 'Rk') & (pd.notna(df.Rk))]


    # Convert datatypes
    df = df.astype({
        'AdjEM': 'float',
        'AdjO': 'float',
        'AdjD': 'float',
        'AdjT': 'float',
        'Luck': 'float',
        'Strength of Schedule_AdjEM': 'float',
        'Strength of Schedule_OppO': 'float',
        'Strength of Schedule_OppD': 'float',
        'NCSOS_AdjEM': 'float',
    })

    return df


def add_data_to_db(year: int, con: sqlite3.Connection):
    """Adds the data to the sql

    :param year: Year of the data to add
    :type year: int
    :param con: _description_
    :type con: sqlite3.Connection
    """
    school_basic_stats_df = pd.read_csv(config['FILE']['SCHOOL_BASIC_SEASON_STATS_FILE'].format(year))
    school_advanced_stats_df = pd.read_csv(config['FILE']['SCHOOL_ADV_SEASON_STATS_FILE'].format(year))

    # Remove Duplicate Information
    school_advanced_stats_df = school_advanced_stats_df.drop(columns=[
        'Overall G',
        'Overall W',
        'Overall L',
        'Overall W-L%',
        'Overall SRS',
        'Overall SOS',
        'Conf. W',
        'Conf. L',
        'Home W',
        'Home L',
        'Away W',
        'Away L',
        'Points Tm.',
        'Points Opp.'
    ])

    school_stats_df = pd.merge(school_basic_stats_df, school_advanced_stats_df, how='outer', on='School')
    school_stats_df['Year'] = year
    school_stats_df.to_sql('school_season_stats', con, if_exists='append', index=False)

    schedule_df = pd.read_csv(config['FILE']['YEAR_SCHEDULE_FILE'].format(year))
    schedule_df['Year'] = year
    schedule_df.to_sql('schedule_stats', con, if_exists='append', index=False)

    school_ratings_df = pd.read_csv(config['FILE']['SPORTS_REFERENCE_RATINGS_FILE'].format(year))
    school_ratings_df['Year'] = year
    school_ratings_df.to_sql('ratings_sr', con, if_exists='append')

    if year >= config['YEARS']['MIN_KENPOM_YEAR']:
        kenpom_df = pd.read_csv(config['FILES']['KENPOM_FILE'].format(year))
        kenpom_df['Year'] = year
        kenpom_df.to_sql('kenpom_stats', con, if_exists='append', index=False)


    if year >= config['YEARS']['MIN_OPPONENT_STATS_YEAR']:
        opponent_basic_stats_df = pd.read_csv(config['FILE']['OPPONENT_BASIC_SEASON_STATS_FILE'].format(year))
        opponent_advanced_stats_df = pd.read_csv(config['FILE']['OPPONENT_ADV_SEASON_STATS_FILE'].format(year))

        # Remove Duplicate Information
        opponent_advanced_stats_df = opponent_advanced_stats_df.drop(columns=[
            'Overall G',
            'Overall W',
            'Overall L',
            'Overall W-L%',
            'Overall SRS',
            'Overall SOS',
            'Conf. W',
            'Conf. L',
            'Home W',
            'Home L',
            'Away W',
            'Away L',
            'Points Tm.',
            'Points Opp.'
        ])

        opponent_stats_df = pd.merge(opponent_basic_stats_df, opponent_advanced_stats_df, how='outer', on='School')
        opponent_stats_df['Year'] = year
        opponent_stats_df.to_sql('opponent_season_stats', con, if_exists='append', index=False)


if __name__ == "__main__":
    main()
