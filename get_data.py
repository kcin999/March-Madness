"""Scrapes Sports Reference for NCAA schedule data"""
import os
import logging
import time
import requests
import pandas as pd
import system
import sqlite3

# Logging setup
logging.basicConfig(
    filename='warning.log',
    encoding='utf-8',
    level=logging.DEBUG
)

SCHOOL_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-school-stats.html"
ADVANCED_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-advanced-school-stats.html"
SCHEDULE_URL = "https://www.sports-reference.com/cbb/schools/{}/men/{}-schedule.html"
SCHEDULE_FOLDER = "./Stats/Schedules/{}"
SEASON_STAT_FILE = '"./Stats/Seasons/{}_season_stats.csv"'
TEAM_SCHEDULE_FILE = './Stats/Schedules/{}/{}_schedule.csv'
YEAR_SCHEDULE_FILE = "./Stats/Schedules/{}_schedule.csv"
BASIC_SEASON_STATS_FILE = './Stats/Seasons/{}_season_stats.csv'
ADV_SEASON_STATS_FILE = './Stats/Seasons/{}_advanced_season_stats.csv'
DATABASE_FILE = 'Stats/stats.sqlite'
MIN_YEAR = 2000
MAX_YEAR = 2023

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


def get_season_data_from_sports_reference(year: int) -> pd.DataFrame:
    """Gets the season data for a given year

    :param year: Year to get data for
    :type year: int

    :return: DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    # Gets Data
    response = requests.get(SCHOOL_STATS_URL.format(year), timeout=None)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': 'basic_school_stats'
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

def get_advanced_season_data_from_sports_reference(year: int) -> pd.DataFrame:
    """Gets the advanced season data for a given year

    :param year: Year to get data for
    :type year: int

    :return: DataFrame of Stats formatted for use
    :rtype: pd.DataFrame
    """
    # Gets Data
    response = requests.get(ADVANCED_STATS_URL.format(year), timeout=None)

    # Makes sure that there is no error
    response.raise_for_status()

    # Reads Data into DataFrame
    df = pd.read_html(
        response.text,
        attrs={
            'id': 'adv_school_stats'
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


def get_score_data_from_sports_reference(year: int) -> pd.DataFrame:
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


    df = df.merge(df[['Team 2', 'Team 1', 'Date', 'Streak']], how='left', left_on=['Team 1', 'Team 2', 'Date'], right_on=['Team 2', 'Team 1', 'Date'], suffixes=[None, '_joined'])

    df = df.drop(columns=['Team 2_joined', 'Team 1_joined'])
    df = df.rename(columns={
        'Streak_joined': 'Team 2 Streak',
        'Streak': 'Team 1 Streak'
    })

    # Idea from here
    # https://stackoverflow.com/questions/51182228/python-delete-duplicates-in-a-dataframe-based-on-two-columns-combinations
    df = df[~df[['Date', 'Team 1', 'Team 2']].apply(frozenset, axis=1).duplicated()]
    df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)


def add_data_to_db(year: int, con: sqlite3.Connection):
    """Adds the data to the sql

    :param year: Year of the data to add
    :type year: int
    :param con: _description_
    :type con: sqlite3.Connection
    """
    basic_stats_df = pd.read_csv(BASIC_SEASON_STATS_FILE.format(year))
    advanced_stats_df = pd.read_csv(ADV_SEASON_STATS_FILE.format(year))
    schedule_df = pd.read_csv(YEAR_SCHEDULE_FILE.format(year))

    # Remove Duplicate Information
    advanced_stats_df = advanced_stats_df.drop(columns=[
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

    stats_df = pd.merge(basic_stats_df, advanced_stats_df, how='outer', on='School')
    stats_df['Year'] = year
    schedule_df['Year'] = year

    stats_df.to_sql('season_stats', con, if_exists='append', index=False)
    schedule_df.to_sql('schedule_stats', con, if_exists='append', index=False)


def main():
    """Main Function"""
    user_answer = input("Would you like to download season statistics? (y/n) ")

    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            df = get_season_data_from_sports_reference(year)
            df.to_csv(SEASON_STAT_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)


    user_answer = input("Would you like to download advanced season statistics? (y/n) ")

    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            df = get_advanced_season_data_from_sports_reference(year)
            df.to_csv(ADV_SEASON_STATS_FILE.format(year), index=False)
            # Sports reference has a rate limiter on it.
            # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
            time.sleep(60/19)

    user_answer = input("Would you like to download schedule information for each team? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            system.createFolder(SCHEDULE_FOLDER.format(year))
            df = get_score_data_from_sports_reference(year)
            df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)

    user_answer = input("Would you like to add from the combine individual schedule results into a larger single season record? (y/n) ")
    if user_answer.upper() in ["Y"]:
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            combine_schedules(year)
            remove_duplicate_games(year)

    user_answer = input("Would you like to add from the CSVs into a sqlite database? (y/n) ")
    if user_answer.upper() in ["Y"]:
        # SQLITE connection
        con = sqlite3.connect(DATABASE_FILE)
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            add_data_to_db(year, con)


    print("Program has finished")


if __name__ == "__main__":
    main()
