import sqlite3
import sys
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report
import pandas as pd
from skops.io import dump, load, get_untrusted_types

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
    'Totals PF'
]

SQLITE_CONNECTION = sqlite3.connect('./Stats/stats.sqlite')

MODEL_NAME = 'final_model.sav'


def get_data(game_type='REG', season_averages: bool = False) -> pd.DataFrame:
    """Gets the data out of the SQLite database

    :param game_type: Game type to select, defaults to 'REG'

        * Options are: 
            * 'REG'
            * 'CTOURN'
            * 'NCAA'

    :type game_type: str, optional

    :param season_averages: Whether to use game average statistics or not, defaults to False
    :type season_averages: bool, optional

    :return: Dataframe of schedule data joined with the season statistics
    :rtype: pd.DataFrame
    """
    df = pd.read_sql_query(
        "SELECT "
        "schst.Result, "
        f"{','.join(['seast_team1.`' + x + '` AS `TEAM_1_' + x + '`' for x in STAT_COLUMNS_TO_USE])}, "
        f"{','.join(['seast_team2.`' + x + '` AS `TEAM_2_' + x + '`'for x in STAT_COLUMNS_TO_USE])} "
        "FROM schedule_stats schst "
        "INNER JOIN season_stats seast_team1 ON schst.`Team 1` = seast_team1.School AND schst.Year = seast_team1.Year "
        "INNER JOIN season_stats seast_team2 ON schst.`Team 2` = seast_team2.School AND schst.Year = seast_team2.Year "
        f"WHERE Type = '{game_type}'",
        SQLITE_CONNECTION
    )

    if season_averages:
        columns_to_not_calulate_average = ['Overall G', 'Overall SRS', 'Overall SOS', 'Totals FG%', 'Totals 3P%', 'Totals FT%']
        for column in STAT_COLUMNS_TO_USE:
            if column in columns_to_not_calulate_average:
                continue

            df['TEAM_1_' + column.replace('Totals', 'Average')] = df['TEAM_1_' + column] / df['TEAM_1_Overall G']
            df['TEAM_2_' + column.replace('Totals', 'Average')] = df['TEAM_2_' + column] / df['TEAM_2_Overall G']
    
        df = df.drop(columns=['TEAM_1_' + x for x in set(STAT_COLUMNS_TO_USE) - set(columns_to_not_calulate_average)])
        df = df.drop(columns=['TEAM_2_' + x for x in set(STAT_COLUMNS_TO_USE) - set(columns_to_not_calulate_average)])

    return df


def clean_data(df: pd.DataFrame, season_averages: bool = False) -> pd.DataFrame:
    """Cleans the data within the dataframe in order for it be more useable

    :param df: Dataframe to clean
    :type df: pd.DataFrame

    :param season_averages: Used for cleaning the data in a different fashion, defaults to False
    :type season_averages: bool, optional

    :return: Cleaned Dataframe
    :rtype: pd.DataFrame
    """
    if not season_averages:
        # Fill blank Minutes played with 40 times the number of games
        df['TEAM_1_Totals MP'].fillna(40 * df['TEAM_1_Overall G'], inplace=True)
        df['TEAM_2_Totals MP'].fillna(40 * df['TEAM_2_Overall G'], inplace=True)
    else:
        df['TEAM_1_Average MP'].fillna(40)
        df['TEAM_2_Average MP'].fillna(40)


    # All other columns, I am backfilling with the median
    df = df.fillna(df.median())

    return df


def train_model(df: pd.DataFrame):
    """Trains the Model for the best results

    :param df: Dataframe of both training and validation data
    :type df: pd.DataFrame

    :return: Sklearn model
    :rtype: sklearn.Model
    """
    y = df['Result']
    X = df.drop(columns=['Result'])

    pipe = Pipeline([
        ('ss', StandardScaler()),
        # ('pca', PCA()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_dict = {
        # 'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        # 'classifier__max_depth': [4,8,12,20],
        # 'pca__n_components': [1,2,3,4,5]
    }

    grid_search = GridSearchCV(pipe, param_grid=[param_dict])

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    predictions = grid_search.predict(X_test)
    print(precision_score(y_test, predictions, average=None))
    print(recall_score(y_test, predictions, average=None))

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.show()

    return grid_search


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
    season_averages = True
    df = get_data(season_averages=season_averages)
    df = clean_data(df, season_averages)
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
