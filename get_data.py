"""Scrapes Sports Reference and KenPom for NCAA statistical data"""
import os
import logging
import time
import sqlite3
import requests
import pandas as pd
import system

# Logging setup
logging.basicConfig(
    filename='warning.log',
    encoding='utf-8',
    level=logging.DEBUG
)

##### URLS #####
SCHOOL_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-school-stats.html"
SCHOOL_ADVANCED_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-advanced-school-stats.html"
OPPONENT_SCHOOL_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-opponent-stats.html"
OPPONENT_ADVANCED_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-advanced-opponent-stats.html"
SPORTS_REFERENCE_RATINGS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-ratings.html"
SCHEDULE_URL = "https://www.sports-reference.com/cbb/schools/{}/men/{}-schedule.html"
KENPOM_URL = "https://kenpom.com/index.php?y={}"

##### Folders #####
SCHEDULE_FOLDER = "./Stats/Schedules/{}"
SEASONS_FOLDER = "./Stats/Seasons"

##### Files #####
TEAM_SCHEDULE_FILE = './Stats/Schedules/{}/{}_schedule.csv'
YEAR_SCHEDULE_FILE = "./Stats/Schedules/{}_schedule.csv"
SCHOOL_BASIC_SEASON_STATS_FILE = './Stats/Seasons/{}_school_season_stats.csv'
SCHOOL_ADV_SEASON_STATS_FILE = './Stats/Seasons/{}_school_advanced_season_stats.csv'
OPPONENT_BASIC_SEASON_STATS_FILE = './Stats/Seasons/{}_opponent_season_stats.csv'
OPPONENT_ADV_SEASON_STATS_FILE = './Stats/Seasons/{}_opponent_advanced_season_stats.csv'
SPORTS_REFERENCE_RATINGS_FILE = './Stats/Seasons/{}_ratings.csv'
KENPOM_FILE = './Stats/Seasons/{}_kenpom.csv'
DATABASE_FILE = 'Stats/stats.sqlite'

## Years to get data for
MIN_YEAR = 2002
MAX_YEAR = 2023

# Opponent Data only goes back to 2010
MIN_OPPONENT_STATS_YEAR = 2010

# Kenpom only goes back to 2002
MIN_KENPOM_YEAR = 2002


SPORTS_REFERENCE_TEAM_NAMES = {
    'houston-christian': 'houston-baptist',
    'kansas-city': 'missouri-kansas-city',
    'little-rock': 'arkansas-little-rock',
    'louisiana': 'louisiana-lafayette',
    'siu-edwardsville': 'southern-illinois-edwardsville',
    'nc-state': 'north-carolina-state',
    'omaha': 'nebraska-omaha',
    'purdue-fort-wayne': 'ipfw',
    'tcu': 'texas-christian',
    'texas-rio-grande-valley': 'texas-pan-american',
    'the-citadel': 'citadel',
    'uab': 'alabama-birmingham',
    'uc-davis': 'california-davis',
    'uc-irvine': 'california-irvine',
    'uc-riverside': 'california-riverside',
    'uc-santa-barbara': 'california-santa-barbara',
    'ucl': 'ucla',
    'unc-asheville': 'north-carolina-asheville',
    'unc-greensboro': 'north-carolina-greensboro',
    'unc-wilmington': 'north-carolina-wilmington',
    'ut-arlington': 'texas-arlington',
    'utep': 'texas-el-paso',
    'uts': 'texas-san-antonio',
    'vmi': 'virginia-military-institute',
    'william--mary': 'william-mary',
}


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
        url = SCHOOL_STATS_URL
        table_id = 'basic_school_stats'
    elif page.upper() == 'OPPONENT':
        url = OPPONENT_SCHOOL_STATS_URL
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
        url = SCHOOL_ADVANCED_STATS_URL
        table_id = 'adv_school_stats'
    elif page.upper() == 'OPPONENT':
        url = OPPONENT_ADVANCED_STATS_URL
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
    response = requests.get(SPORTS_REFERENCE_RATINGS_URL.format(year), timeout=None)

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


def get_schedule_data_from_sports_reference(year: int) -> pd.DataFrame:
    """Gets the schedule information from sports reference

    :param year: Year to get the data for
    :type year: int

    :return:  DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    team_list = get_team_list(year)

    df = pd.DataFrame()
    for i, team in enumerate(team_list):

        if os.path.exists(TEAM_SCHEDULE_FILE.format(year, team)):
            continue

        # Sports reference has a rate limiter on it.
        # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
        time.sleep(60/19)

        # Format team name
        team_name = team.replace(' ', '-').lower()
        team_name = team_name.replace('&', '')
        team_name = team_name.replace('(', '')
        team_name = team_name.replace(')', '')
        team_name = team_name.replace("'", '')
        team_name = team_name.replace('.', '')

        # Error for houston-christian appearing as houston-baptist on sports reference
        if team_name in SPORTS_REFERENCE_TEAM_NAMES:
            team_name = SPORTS_REFERENCE_TEAM_NAMES[team_name]

        # Prints where we are at in the sequence
        print(f"Grabbing Data for {team} in {year} ({i}/{len(team_list)})")

        # Gets Data
        response = requests.get(
            SCHEDULE_URL.format(team_name, year),
            timeout=None
        )

        if response.status_code == 404:
            logging.warning(
                "Could not find address (%s) for %s",
                SCHEDULE_URL.format(team_name, year),
                team
            )
            continue
        elif response.status_code != 200:
            logging.warning(
                "Status Code: %s return for URL: %s",
                response.status_code, SCHEDULE_URL.format(team_name, year)
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
                SCHEDULE_URL.format(team_name, year)
            )
            continue

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

        team_df.to_csv(TEAM_SCHEDULE_FILE.format(year, team), index=False)

        df = pd.concat([df, team_df])

    return df


def combine_schedules(year: int):
    """Combines the individual team schedules into one large CSV file.

    :param year: Year to combine schedules
    :type year: int
    """
    df = pd.DataFrame()
    for file in os.listdir(SCHEDULE_FOLDER.format(year)):
        team_df = pd.read_csv(SCHEDULE_FOLDER.format(year) + "/" + file)
        team_df['Date'] = pd.to_datetime(team_df['Date'])

        team_df = team_df[[
            'Date',
            'Team 1',
            'Team 2',
            'SRS',
            'Type',
            'Team 1 Score',
            'Team 2 Score',
            'Streak',
            'Result'
        ]]
        # Shifts Streaks to be able to have the win streak coming into the game
        team_df['Streak'] = team_df['Streak'].shift(1)
        team_df['Streak'] = team_df['Streak'].fillna(0)

        df = pd.concat([df, team_df], ignore_index=True)

    df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)


def remove_duplicate_games(year: int):
    """Removes Duplicate Games from the schedules

    :param year: _description_
    :type year: int
    """
    df = pd.read_csv(YEAR_SCHEDULE_FILE.format(year))


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
    df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)


def get_kenpom_data(year: int) -> pd.DataFrame:
    """Scrapes KenPom data for advanced analytics

    :param year: Year to grab
    :type year: int
    """
    # Gets Data
    # Set Custom Headers to be able to work
    headers = {'User-Agent': 'PostmanRuntime/7.29.2'}
    response = requests.get(KENPOM_URL.format(year), timeout=None, headers = headers)

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
    school_basic_stats_df = pd.read_csv(SCHOOL_BASIC_SEASON_STATS_FILE.format(year))
    school_advanced_stats_df = pd.read_csv(SCHOOL_ADV_SEASON_STATS_FILE.format(year))

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

    schedule_df = pd.read_csv(YEAR_SCHEDULE_FILE.format(year))
    schedule_df['Year'] = year
    schedule_df.to_sql('schedule_stats', con, if_exists='append', index=False)

    school_ratings_df = pd.read_csv(SPORTS_REFERENCE_RATINGS_FILE.format(year))
    school_ratings_df['Year'] = year
    school_ratings_df.to_sql('ratings_sr', con, if_exists='append')

    if year >= MIN_KENPOM_YEAR:
        kenpom_df = pd.read_csv(KENPOM_FILE.format(year))
        kenpom_df['Year'] = year
        kenpom_df.to_sql('kenpom_stats', con, if_exists='append', index=False)


    if year >= MIN_OPPONENT_STATS_YEAR:
        opponent_basic_stats_df = pd.read_csv(OPPONENT_BASIC_SEASON_STATS_FILE.format(year))
        opponent_advanced_stats_df = pd.read_csv(OPPONENT_ADV_SEASON_STATS_FILE.format(year))

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


def main():
    """Main Function"""
    system.createFolder(SEASONS_FOLDER)
    user_answer = input("Would you like to download school season statistics for sports reference? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            df = get_season_data_from_sports_reference(year, 'school')
            df.to_csv(SCHOOL_BASIC_SEASON_STATS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download opponent season statistics for sports reference? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(max(MIN_OPPONENT_STATS_YEAR, MIN_YEAR), MAX_YEAR + 1):
            df = get_season_data_from_sports_reference(year, 'opponent')
            df.to_csv(OPPONENT_BASIC_SEASON_STATS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download school advanced season statistics from sports reference? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            df = get_advanced_season_data_from_sports_reference(year, 'school')
            df.to_csv(SCHOOL_ADV_SEASON_STATS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download opponent advanced season statistics from sports reference? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(max(MIN_OPPONENT_STATS_YEAR, MIN_YEAR), MAX_YEAR + 1):
            df = get_advanced_season_data_from_sports_reference(year, 'opponent')
            df.to_csv(OPPONENT_ADV_SEASON_STATS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download rating information from sports reference? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(max(MIN_YEAR, MIN_YEAR), MAX_YEAR + 1):
            df = get_ratings_from_sports_reference(year)
            df.to_csv(SPORTS_REFERENCE_RATINGS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download schedule information for each team? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            system.createFolder(SCHEDULE_FOLDER.format(year))
            df = get_schedule_data_from_sports_reference(year)
            df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)

    user_answer = input("Would you like to get ranking data from KenPom? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(max(MIN_KENPOM_YEAR, MIN_YEAR), MAX_YEAR + 1):
            system.createFolder(SCHEDULE_FOLDER.format(year))
            df = get_kenpom_data(year)
            df.to_csv(KENPOM_FILE.format(year), index=False)
            time.sleep(60/19)

    user_answer = input("Would you like to add from the combine individual schedule results into a larger single season record? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            combine_schedules(year)
            remove_duplicate_games(year)

    user_answer = input("Would you like to add from the CSVs into a sqlite database? (y/n) ")
    if user_answer.upper() in ["Y"]:
        # SQLITE connection
        con = sqlite3.connect(DATABASE_FILE)
        try:
            for year in range(MIN_YEAR, MAX_YEAR + 1):
                add_data_to_db(year, con)
        except FileNotFoundError as exc:
            raise FileNotFoundError("All modes must have run in order to build a correct sqlite file.") from exc

    print("Program has finished")


if __name__ == "__main__":
    main()
