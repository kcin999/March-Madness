import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import sqlite3
import pandas as pd

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

def get_data() -> pd.DataFrame:
    """Gets the data out of the SQLite database

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
        "WHERE Type = 'REG'",
        SQLITE_CONNECTION
    )

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data within the dataframe in order for it be more useable

    :param df: Dataframe to clean
    :type df: pd.DataFrame

    :return: Cleaned Dataframe
    :rtype: pd.DataFrame
    """
    # Fill blank Minutes played with 40 times the number of games
    df['TEAM_1_Totals MP'].fillna(40 * df['TEAM_1_Overall G'],inplace=True)
    df['TEAM_2_Totals MP'].fillna(40 * df['TEAM_2_Overall G'],inplace=True)

    # All other columns, I am backfilling with the median
    df = df.fillna(df.median())

    return df

def run_model(df: pd.DataFrame):
    y = df['Result']
    X = df.drop(columns=['Result'])

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    pca_reduction = PCA()
    X_reduced = pca_reduction.fit_transform(X_standardized)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2)

    classifier = SVC()
    tree = classifier.fit(X_train, y_train)


    predictions = classifier.predict(X_test)
    print(precision_score(y_test, predictions, average=None))
    print(recall_score(y_test, predictions, average=None))

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.show()

def main():
    """Main Function"""
    df = get_data()
    df = clean_data(df)
    run_model(df)

if __name__ == "__main__":
    main()