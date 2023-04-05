import sqlite3
import datetime
import sys
import logging
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
            datefmt='%H:%M:%S'
        )
    )
fh.setLevel(logging.INFO)
models_logger.addHandler(fh)

STAT_COLUMNS_TO_USE = [
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

KENPOM_RANKINGS = [
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
        ('classifier', LogisticRegression(max_iter=1000))
        # ('classifier', MLPClassifier(max_iter=1000))
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    param_dict = {
        # 'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        # 'classifier__max_depth': [4,8,12,20],
        # 'pca__n_components': [1, 5,10,15,18]
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


    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    print(precision_score(y_test, predictions, average=None))
    print(recall_score(y_test, predictions, average=None))

    print(classification_report(y_test, predictions))

    return grid_search, results


def compare_model_against_ncaa(query: str, model):
    """Takes the model and makes predictions against the march madness data that I have.

    :param query: Query used to get the NCAA data. Must have columns that match the orginal traning
    :type query: str

    :param model: Model to validate
    :type model: Sklearn Model

    :return: Returns the results for the model for the log file
    :rtype: dict
    """
    results = {}
    df = get_data(query)
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


def load_model(file_name: str = "my-model.skops"):
    """Loads the model from the saved skops file

    :param file_name: filename in which the model is currently loaded, defaults to "my-model.skops"
    :type file_name: str, optional

    :return: Sklearn model which can be used
    :rtype: sklearn.Model
    """
    unknown_types = get_untrusted_types(file=file_name)
    if unknown_types:
        print("The following fields were deemed untrusted.")
        print(unknown_types)
        user_input = input("Would you like to continue? (y/n): ")
        if user_input.upper().strip() != 'Y':
            print("Goodbye")
            sys.exit()

    return load(file_name, trusted=True)

def main():
    """Main Function"""
    # Set up and parameters
    # query = (
    #     "SELECT "
    #     "schst.Result, "
    #     "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
    #     f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in STAT_COLUMNS_TO_USE])}, "
    #     f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in STAT_COLUMNS_TO_USE])} "
    #     "FROM schedule_stats schst "
    #     "INNER JOIN school_season_stats seast_team1 ON schst.`Team 1` = seast_team1.School AND schst.Year = seast_team1.Year "
    #     "INNER JOIN school_season_stats seast_team2 ON schst.`Team 2` = seast_team2.School AND schst.Year = seast_team2.Year "
    #     "WHERE Type = '{}'"
    # )
    query = (
        "SELECT "
        "schst.Result, "
        "schst.`Team 1 Streak`,  schst.`Team 2 Streak`, "
        f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in KENPOM_RANKINGS])}, "
        f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in KENPOM_RANKINGS])} "
        "FROM schedule_stats schst "
        "INNER JOIN kenpom_stats seast_team1 ON schst.`Team 1` = seast_team1.Team AND schst.Year = seast_team1.Year "
        "INNER JOIN kenpom_stats seast_team2 ON schst.`Team 2` = seast_team2.Team AND schst.Year = seast_team2.Year "
        "WHERE Type = '{}'"
    )
    system.createFolder('.models/')
    model_file_name = 'models/' + datetime.datetime.now().strftime('%Y%m%d %H%M%S') + '_model.skops'
    use_averages = True

    # Get Data
    df = get_data(query.format('REG'))
    df = clean_data(df)

    # Train Model
    model, results = train_model(df)

    # Save Model
    save_model(model, model_file_name)

    # Predict NCAA games for another step of validation
    results['NCAA'] = compare_model_against_ncaa(query.format('NCAA'), model)

    # Output results to logfile
    results['use_averages'] = use_averages
    output_string_to_log = (
        f"Model Saved to: {model_file_name}\n"
        f"\tQuery: {query}\n"
        f"\tUse Averages: {use_averages}\n"
        f"\tbestparams: {results['bestparams']}\n"
        f"\tbestestimator: {results['bestestimator']}\n"
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


if __name__ == "__main__":
    main()
