"""Machine Learning Module"""
import sqlite3
import datetime
import sys
import logging
import csv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score, classification_report
import pandas as pd
from skops.io import dump, load, get_untrusted_types
import system

# Setup Logging
models_logger = logging.getLogger('Model_Logger')
models_logger.setLevel(logging.INFO)
fh = logging.FileHandler('modelinfo.log')
fh.setFormatter(
        logging.Formatter(
            '%(asctime)s; %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'
        )
    )
fh.setLevel(logging.INFO)
models_logger.addHandler(fh)

SCHOOL_SEASON_STATS = [
    'Overall G',
    'Overall SRS',
    'Overall SOS',
    'Points Tm.',
    'Points Opp.',
    'Totals MP',
    'Totals FG',
    'Totals FGA',
    'Totals FG%',
    'Totals 3P',
    'Totals 3PA',
    'Totals 3P%',
    'Totals FT',
    'Totals FTA',
    'Totals FT%',
    'Totals ORB',
    'Totals TRB',
    'Totals AST',
    'Totals STL',
    'Totals BLK',
    'Totals TOV',
    'Totals PF',
    'School Advanced Pace',
    'School Advanced ORtg',
    'School Advanced FTr',
    'School Advanced 3PAr',
    'School Advanced TS%',
    'School Advanced TRB%',
    'School Advanced AST%',
    'School Advanced STL%',
    'School Advanced BLK%',
    'School Advanced eFG%',
    'School Advanced TOV%',
    'School Advanced ORB%',
    'School Advanced FT/FGA',
]

OPPONENT_SEASON_STATS = [
    "Overall G",
    "Overall SRS",
    "Overall SOS",
    "Points Tm.",
    "Points Opp.",
    "Opponent MP",
    "Opponent FG",
    "Opponent FGA",
    "Opponent FG%",
    "Opponent 3P",
    "Opponent 3PA",
    "Opponent 3P%",
    "Opponent FT",
    "Opponent FTA",
    "Opponent FT%",
    "Opponent ORB",
    "Opponent TRB",
    "Opponent AST",
    "Opponent STL",
    "Opponent BLK",
    "Opponent TOV",
    "Opponent PF",
    "MadeTournament",
    "Opponent Advanced Pace",
    "Opponent Advanced ORtg",
    "Opponent Advanced FTr",
    "Opponent Advanced 3PAr",
    "Opponent Advanced TS%",
    "Opponent Advanced TRB%",
    "Opponent Advanced AST%",
    "Opponent Advanced STL%",
    "Opponent Advanced BLK%",
    "Opponent Advanced eFG%",
    "Opponent Advanced TOV%",
    "Opponent Advanced ORB%",
    "Opponent Advanced FT/FGA",
]

KENPOM_STATS = [
    'AdjEM',
    'AdjO',
    'AdjD',
    'AdjT',
    'Luck',
    'Strength of Schedule_AdjEM',
    'Strength of Schedule_OppO',
    'Strength of Schedule_OppD',
    'NCSOS_AdjEM'
]

RATINGS_SR_STATS = [
    'Pts',
    'Opp',
    'MOV',
    'SOS',
    'SRS OSRS',
    'SRS DSRS',
    'SRS SRS',
    'Adjusted ORtg',
    'Adjusted DRtg',
    'Adjusted NRtg'
]

AVERAGE_COLUMNS = [
    'Points Tm.', 
    'Points Opp.', 
    'Totals MP', 
    'Totals FG', 
    'Totals FGA', 
    'Totals 3P', 
    'Totals 3PA', 
    'Totals FT',
    'Totals FTA',
    'Totals ORB',
    'Totals TRB',
    'Totals AST',
    'Totals AST',
    'Totals STL',
    'Totals BLK',
    'Totals TOV',
    'Totals PF',
    'Oppontent MP', 
    'Oppontent FG', 
    'Oppontent FGA', 
    'Oppontent 3P', 
    'Oppontent 3PA', 
    'Oppontent FT',
    'Oppontent FTA',
    'Oppontent ORB',
    'Oppontent TRB',
    'Oppontent AST',
    'Oppontent AST',
    'Oppontent STL',
    'Oppontent BLK',
    'Oppontent TOV',
    'Oppontent PF',
]

DATABASE_FILE = './Stats/stats.sqlite'

SQLITE_CONNECTION = sqlite3.connect(DATABASE_FILE)


def get_data(query: str, use_averages=False) -> pd.DataFrame:
    """Gets the data out of the SQLite database

    :param query: SQL Query
    :type query: str

    :param use_averages: Whether to use game average statistics or not, defaults to False
    :type use_averages: bool, optional

    :return: Dataframe based on the query returned. 
    :rtype: pd.DataFrame
    """
    # Query to Execute
    df = pd.read_sql_query(
        query,
        SQLITE_CONNECTION
    )

    # Converts some columns to averages by game. Columns must be in the AVERAGE_COLUMNS list
    if use_averages:
        # Sets lists for columns to drop that are the total columns
        drop_columns_team_1 = []
        drop_columns_team_2 = []

        for column_name in df.columns.tolist():
            if 'Overall G' in column_name :
                drop_columns_team_1.append('TEAM_1_Overall G')
                drop_columns_team_2.append('TEAM_2_Overall G')
            # Team 1 check
            column = column_name.replace('TEAM_1_', '')
            if column in AVERAGE_COLUMNS:
                df['TEAM_1_' + column.replace('Totals', 'Average')] = df['TEAM_1_' + column] / df['TEAM_1_Overall G']
                drop_columns_team_1.append('TEAM_1_' + column)

            # Team 2 Check
            column = column_name.replace('TEAM_2_', '')
            if column in AVERAGE_COLUMNS:
                df['TEAM_2_' + column.replace('Totals', 'Average')] = df['TEAM_2_' + column] / df['TEAM_2_Overall G']
                drop_columns_team_2.append('TEAM_2_' + column)

        # Drop undeeded columns
        df = df.drop(columns=[x for x in drop_columns_team_1])
        df = df.drop(columns=[x for x in drop_columns_team_2])

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data within the dataframe in order for it be more useable

    :param df: Dataframe to clean
    :type df: pd.DataFrame

    :return: Cleaned Dataframe
    :rtype: pd.DataFrame
    """
    # If Minutes Played being used in average, backfill with 40
    if 'TEAM_1_Average MP' in df.columns.tolist():
        df['TEAM_1_Average MP'].fillna(40, inplace=True)

    if 'TEAM_2_Average MP' in df.columns.tolist():
        df['TEAM_2_Average MP'].fillna(40, inplace=True)


    # If minutes played being used in total, backfill with 40 times number of games
    if 'TEAM_1_Totals MP' in df.columns.tolist() and 'TEAM_1_Overall G' in df.columns.tolist():
        df['TEAM_1_Totals MP'].fillna(40 * df['TEAM_1_Overall G'], inplace=True)

    if 'TEAM_2_Totals MP' in df.columns.tolist() and 'TEAM_2_Overall G' in df.columns.tolist():
        df['TEAM_2_Totals MP'].fillna(40 * df['TEAM_2_Overall G'], inplace=True)

    # Fill Blank win streaks with 0
    if 'Team 1 Streak' in df.columns.tolist():
        df['Team 1 Streak'].fillna(0)

    if 'Team 2 Streak' in df.columns.tolist():
        df['Team 2 Streak'].fillna(0)


    # All other columns, I am backfilling with the median
    df = df.fillna(df.median())

    return df


def train_model(df: pd.DataFrame):
    """Trains the Model for the best results

    :param df: Dataframe of both training and validation data
    :type df: pd.DataFrame

    :return: Sklearn model and a dictionary of the results for logging
    :rtype: sklearn.Model, dict
    """
    # Result Dictionary so I can remember / output what models I have tried and ran, along with stats about it
    results = {
        'pipeline': None,
        'paramdict': None,
        'bestparams': None,
        'bestestimator': None,
        'Training Validation':{
            'accuracy': None,
            'precision': [None, None],
            'recall': [None, None],
            'f1score': [None, None],
            'True Positives': None,
            'False Positives': None,
            'True Negatives': None,
            'False Negatives':None
        }
    }

    # Set up values for training
    y = df['Result']
    x = df.drop(columns=['Result'])

    pipe = Pipeline([
        ('ss', StandardScaler()),
        # ('pca', PCA()),
        # ('classifier', VotingClassifier(
        #     estimators=[
        #         ('logistic', LogisticRegression(max_iter=1000)),
        #         ('tree', DecisionTreeClassifier()), 
        #         ('ann', MLPClassifier(max_iter=1000))
        #     ]
        # ))
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    param_dict = {
        # 'classifier__solver': ['lbfgs', 'sgd', 'adam'],
        # 'classifier__alpha': [0.0001, 0.001, 0.01, 0.1]
    }

    grid_search = GridSearchCV(pipe, param_grid=[param_dict], n_jobs=2)

    grid_search.fit(x_train, y_train)

    predictions = grid_search.predict(x_test)

    cm = confusion_matrix(y_test, predictions)
    results['pipeline'] = pipe
    results['paramdict'] = param_dict
    results['bestparams'] = grid_search.best_params_
    results['bestestimator'] = grid_search.best_estimator_
    results['Training Validation']['precision'] = precision_score(y_test, predictions, average=None)
    results['Training Validation']['recall'] = recall_score(y_test, predictions, average=None)
    results['Training Validation']['accuracy'] = accuracy_score(y_test, predictions)
    results['Training Validation']['f1score'] = f1_score(y_test, predictions)
    results['Training Validation']['True Positives'] = cm[1][1]
    results['Training Validation']['False Positives'] = cm[0][1]
    results['Training Validation']['True Negatives'] = cm[0][0]
    results['Training Validation']['False Negatives'] = cm[1][0]


    return grid_search, results


def compare_model_against_ncaa(query: str, model, use_averages: bool):
    """Takes the model and makes predictions against the march madness data that I have.

    :param query: Query used to get the NCAA data. Must have columns that match the orginal traning
    :type query: str

    :param model: Model to validate
    :type model: Sklearn Model

    :param use_averages: Whether to use statistical averages in model
    :type use_averages: bool

    :return: Returns the results for the model for the log file
    :rtype: dict
    """
    results = {}
    df = get_data(query, use_averages)
    df = clean_data(df)

    y = df['Result']
    x = df.drop(columns=['Result'])

    predictions = model.predict(x)
    cm = confusion_matrix(y, predictions)

    results['precision'] = precision_score(y, predictions, average=None)
    results['recall'] = recall_score(y, predictions, average=None)
    results['accuracy'] = accuracy_score(y, predictions)
    results['f1score'] = f1_score(y, predictions)
    results['True Positives'] = cm[1][1]
    results['False Positives'] = cm[0][1]
    results['True Negatives'] = cm[0][0]
    results['False Negatives'] = cm[1][0]

    return results


def save_model(model, file_name: str = "my-model.skops"):
    """Saves the model into the folder directory for later use

    :param model: sklearn model to ssave
    :type model: Any

    :param file_name: Name of the file to send the model to, defaults to "my-model.skops"
    :type file_name: str, optional
    """
    dump(model, file_name)


def load_model(file_name: str = "my-model.skops", override_trust=False):
    """Loads the model from the saved skops file

    :param file_name: filename in which the model is currently loaded, defaults to "my-model.skops"
    :type file_name: str, optional

    :param override_trust: Whether to override the trust connection, defaults to False
    :type override_trust: bool, optional

    :return: Sklearn model which can be used
    :rtype: sklearn.Model
    """
    unknown_types = get_untrusted_types(file=file_name)
    if unknown_types and not override_trust:
        print("The following fields were deemed untrusted.")
        print(unknown_types)
        user_input = input("Would you like to continue? (y/n): ")
        if user_input.upper().strip() != 'Y':
            print("Goodbye")
            sys.exit()

    return load(file_name, trusted=True)


def output_results(results:dict, model_file_name: str, use_averages: bool, query: str):
    """Outputs the results of the program to the logfile

    :param results: Result Dictionary
    :type results: dict
    :param model_file_name: File name of the saved model
    :type model_file_name: str
    :param use_averages: Whether Season Averages were used
    :type use_averages: bool
    :param query: _description_
    :type query: str
    """

    # Output results to logfile
    results['use_averages'] = use_averages
    pipeline_line_break = '\n               '
    output_string_to_log = (
        f"Model Saved to: {model_file_name}\n"
        f"\tQuery: {query}\n"
        f"\tUse Averages: {use_averages}\n"
        "\tModel Information:\n"
            f"\t\tPipeline: {str(results['pipeline']).replace(pipeline_line_break, '')}\n"
            f"\t\tParam Dictionary: {results['paramdict']}\n"
            f"\t\tbestparams: {results['bestparams']}\n"
            f"\t\tbestestimator: {str(results['bestestimator']).replace(pipeline_line_break, '')}\n"
        "\tTraining:\n"
            f"\t\tprecision: {results['Training Validation']['precision']}\n"
            f"\t\trecall: {results['Training Validation']['recall']}\n"
            f"\t\taccuracy: {results['Training Validation']['accuracy']}\n"
            f"\t\tf1score: {results['Training Validation']['f1score']}\n"
            f"\t\tTrue Positives: {results['Training Validation']['True Positives']}\n"
            f"\t\tFalse Positives: {results['Training Validation']['False Positives']}\n"
            f"\t\tTrue Negatives: {results['Training Validation']['True Negatives']}\n"
            f"\t\tFalse Negatives: {results['Training Validation']['False Negatives']}\n"
        "\tNCAA:\n"
            f"\t\tprecision: {results['NCAA']['precision']}\n"
            f"\t\trecall: {results['NCAA']['recall']}\n"
            f"\t\taccuracy: {results['NCAA']['accuracy']}\n"
            f"\t\tf1score: {results['NCAA']['f1score']}\n"
            f"\t\tTrue Positives: {results['NCAA']['True Positives']}\n"
            f"\t\tFalse Positives: {results['NCAA']['False Positives']}\n"
            f"\t\tTrue Negatives: {results['NCAA']['True Negatives']}\n"
            f"\t\tFalse Negatives: {results['NCAA']['False Negatives']}"
            "\n\n\n"
    )

    models_logger.info(output_string_to_log)

    # Outputs Results to CSV as well
    with open('modelinfo.csv', 'a', newline='') as file:
        writer_object = csv.writer(file)

        writer_object.writerow([
            datetime.datetime.now().isoformat(),
            model_file_name,
            query,
            use_averages,
            str(results['pipeline']).replace(pipeline_line_break, ''),
            results['paramdict'],
            results['bestparams'],
            str(results['bestestimator']).replace(pipeline_line_break, ''),
            results['Training Validation']['precision'],
            results['Training Validation']['recall'],
            results['Training Validation']['accuracy'],
            results['Training Validation']['f1score'],
            results['Training Validation']['True Positives'],
            results['Training Validation']['False Positives'],
            results['Training Validation']['True Negatives'],
            results['Training Validation']['False Negatives'],
            results['NCAA']['precision'],
            results['NCAA']['recall'],
            results['NCAA']['accuracy'],
            results['NCAA']['f1score'],
            results['NCAA']['True Positives'],
            results['NCAA']['False Positives'],
            results['NCAA']['True Negatives'],
            results['NCAA']['False Negatives']
        ])

        file.close()


def main():
    """Main Function"""
    # Set up and parameters

    # school_stats_query = (
    #     "SELECT "
    #     "schst.Result, "
    #     "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
    #     f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in SCHOOL_SEASON_STATS])}, "
    #     f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in SCHOOL_SEASON_STATS])} "
    #     "FROM schedule_stats schst "
    #     "INNER JOIN school_season_stats seast_team1 ON schst.`Team 1` = seast_team1.School AND schst.Year = seast_team1.Year "
    #     "INNER JOIN school_season_stats seast_team2 ON schst.`Team 2` = seast_team2.School AND schst.Year = seast_team2.Year "
    #     "WHERE Type = '{}'"
    # )
    # school_and_opponent_stats = (
    #     "SELECT "
    #     "schst.Result, "
    #     "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
    #     f"{','.join(['school_stats1.`' + x + '` AS `SCHOOL_TEAM_1_' + x + '`' for x in SCHOOL_SEASON_STATS])}, "
    #     f"{','.join(['school_stats2.`' + x + '` AS `SCHOOL_TEAM_2_' + x + '`'for x in SCHOOL_SEASON_STATS])}, "
    #     f"{','.join(['opponent_stats1.`' + x + '` AS `OPPONENT_TEAM_1_' + x + '`' for x in OPPONENT_SEASON_STATS])}, "
    #     f"{','.join(['opponent_stats2.`' + x + '` AS `OPPONENT_TEAM_2_' + x + '`'for x in OPPONENT_SEASON_STATS])} "
    #     "FROM schedule_stats schst "
    #     "INNER JOIN school_season_stats school_stats1 ON schst.`Team 1` = school_stats1.School AND schst.Year = school_stats1.Year "
    #     "INNER JOIN school_season_stats school_stats2 ON schst.`Team 2` = school_stats2.School AND schst.Year = school_stats2.Year "
    #     "INNER JOIN opponent_season_stats opponent_stats1 ON schst.`Team 1` = opponent_stats1.School AND schst.Year = opponent_stats1.Year "
    #     "INNER JOIN opponent_season_stats opponent_stats2 ON schst.`Team 2` = opponent_stats2.School AND schst.Year = opponent_stats2.Year "
    #     "WHERE Type = '{}' AND schst.Year >= 2010 "
    # )
    kenpom_stats_query = (
        "SELECT "
        "schst.Result, "
        # "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
        f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in KENPOM_STATS])}, "
        f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in KENPOM_STATS])} "
        "FROM schedule_stats schst "
        "INNER JOIN team_mapping team_map1 ON schst.`Team 1` = team_map1.`Sports Reference` "
        "INNER JOIN team_mapping team_map2 ON schst.`Team 2` = team_map2.`Sports Reference` "
        "INNER JOIN kenpom_stats seast_team1 ON seast_team1.Team = team_map1.Kenpom AND schst.Year = seast_team1.Year "
        "INNER JOIN kenpom_stats seast_team2 ON seast_team2.Team = team_map2.Kenpom AND schst.Year = seast_team2.Year "
        "WHERE Type = '{}'"
    )
    # sr_ratings_query = (
    #     "SELECT "
    #     "schst.Result, "
    #     # "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
    #     f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in RATINGS_SR_STATS])}, "
    #     f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in RATINGS_SR_STATS])} "
    #     "FROM schedule_stats schst "
    #     "INNER JOIN ratings_sr seast_team1 ON schst.`Team 1` = seast_team1.School AND schst.Year = seast_team1.Year "
    #     "INNER JOIN ratings_sr seast_team2 ON schst.`Team 2` = seast_team2.School AND schst.Year = seast_team2.Year "
    #     "WHERE Type = '{}'"
    # )
    system.createFolder('./models/')
    model_file_name = 'models/' + datetime.datetime.now().strftime('%Y%m%d %H%M%S') + '_model.skops'
    use_averages = True

    # Get Data
    df = get_data(kenpom_stats_query.format('REG'), use_averages)
    df = clean_data(df)

    # Train Model
    model, results = train_model(df)

    # Save Model
    save_model(model, model_file_name)

    # Predict NCAA games for another step of validation
    results['NCAA'] = compare_model_against_ncaa(kenpom_stats_query.format('NCAA'), model, use_averages)

    output_results(results, model_file_name, use_averages, kenpom_stats_query)


if __name__ == "__main__":
    main()
