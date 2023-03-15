"""Scrapes Sports Reference for NCAA schedule data"""
import os
import logging
import time
import requests
import pandas as pd
import system

# Logging setup
logging.basicConfig(
    filename='warning.log',
    encoding='utf-8',
    level=logging.DEBUG
)

SCHOOL_STATS_URL = "https://www.sports-reference.com/cbb/seasons/men/{}-school-stats.html"
SCHEDULE_URL = "https://www.sports-reference.com/cbb/schools/{}/men/{}-schedule.html"
TEAM_FILE_PATH = './Stats/Schedules/{}/{}_schedule.csv'
SCHEDULE_FOLDER = "./Stats/Schedules/{}"
YEAR_SCHEDULE_FILE = "./Stats/Schedules/{}_schedule.csv"
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

        if os.path.exists(TEAM_FILE_PATH.format(year, team)):
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

        team_df.to_csv(TEAM_FILE_PATH.format(year, team), index=False)

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
            'Team 1 Score',
            'Team 2 Score',
            'Streak',
            'Result'
        ]]
        df = pd.concat([df, team_df])

    df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)


def remove_duplicate_games(year: int):
    """Removes Duplicate Games from the schedules

    :param year: _description_
    :type year: int
    """
    df = pd.read_csv(YEAR_SCHEDULE_FILE.format(year))

    # Idea from here
    # https://stackoverflow.com/questions/51182228/python-delete-duplicates-in-a-dataframe-based-on-two-columns-combinations
    df = df[~df[['Date', 'Team 1', 'Team 2']].apply(frozenset, axis=1).duplicated()]
    df.to_csv(YEAR_SCHEDULE_FILE.format(year), index=False)


def main():
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        df = get_season_data_from_sports_reference(year)
        df.to_csv(f"./Stats/Seasons/{year}_season_stats.csv", index=False)
        # Sports reference has a rate limiter on it.
        # The rate no more than 20 calls per minute. So I am putting a sleep on it, to call only 19 times in a minute
        time.sleep(60/19)

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        system.createFolder(f'./Stats/Schedules/{year}')
        df = get_score_data_from_sports_reference(year)
        df.to_csv(f"./Stats/Schedules/{year}_schedule.csv", index=False)

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        combine_schedules(year)
        remove_duplicate_games(year)


if __name__ == "__main__":
    main()
