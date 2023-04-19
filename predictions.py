import tkinter as tk
from tkinter import ttk
import sqlite3
import pandas as pd
from machine_learning import load_model, get_data, clean_data

DATABASE_FILE = './Stats/stats.sqlite'
SQLITE_CONNECTION = sqlite3.connect(DATABASE_FILE)

def get_team_list() -> list:
    """Gets the Team List

    :return: List of teams
    :rtype: list
    """
    df = pd.read_sql_query(
        "SELECT DISTINCT Team FROM kenpom_stats",
        SQLITE_CONNECTION
    )
    return df.sort_values('Team')['Team'].values.tolist()

TEAM_LIST = get_team_list()

class GUI():
    """GUI for predictions"""
    TEAM_LIST = get_team_list()
    def __init__(self, model: str = None):
        """Init Function

        :param model: Machine Learning Model to load, defaults to None
        :type model: str, optional
        """
        self.model = load_model(model, override_trust = True)

        self.root = tk.Tk()

        self.root.title("The Perfect Bracket")
        self.root.geometry('350x350')

        self.team1 = tk.StringVar()
        self.team2 =tk.StringVar()
        self.prediction = tk.StringVar()
        self.team1_percentage = tk.StringVar()
        self.team2_percentage = tk.StringVar()

        self.create_selection_frame()
        self.create_results_frame()


    def create_selection_frame(self):
        """Creates the selection frame"""
        selection_frame = tk.Frame(self.root, width=200, height=400)

        team1_label = ttk.Label(selection_frame, text="Select Team 1")
        team1_selection = ttk.Combobox(selection_frame, values=TEAM_LIST, textvariable=self.team1)
        team1_label.grid(row=0, column=0)
        team1_selection.grid(row=1, column=0)

        team2_label = ttk.Label(selection_frame, text="Select Team 2")
        team2_selection = ttk.Combobox(selection_frame, values=TEAM_LIST, textvariable=self.team2)
        team2_label.grid(row=0, column=1)
        team2_selection.grid(row=1, column=1)

        btn = tk.Button(selection_frame, text="Make Predicition", command=self.make_prediction)
        btn.grid(row=2, column=0, columnspan=2)

        selection_frame.grid(row=0, column=0)


    def create_results_frame(self):
        """Creates the result frame"""
        results_frame = tk.Frame(self.root, width=200, height=400)

        prediciton_label = ttk.Label(results_frame, textvariable=self.prediction)
        prediciton_label.grid(row=0, column = 0)

        team1_percentage_label = ttk.Label(results_frame, textvariable=self.team1_percentage)
        team1_percentage_label.grid(row=1, column = 0)

        team2_percentage_label = ttk.Label(results_frame, textvariable=self.team2_percentage)
        team2_percentage_label.grid(row=2, column = 0)

        results_frame.grid(row=1, column=0)


    def run(self):
        """Runs the GUI"""
        self.root.mainloop()


    def make_prediction(self):
        """Makes the prediction"""
        df = self.get_team_data()
        predictions = self.model.predict(df)[0]

        try:
            predictions_probabilities = self.model.predict_proba(df)[0]
        except AttributeError:
            predictions_probabilities = False

        if predictions == 1:
            self.prediction.set(f"{self.team1.get()} is predicited to beat {self.team2.get()}")
        else:
            self.prediction.set(f"{self.team1.get()} is predicited to lose to {self.team2.get()}")

        if predictions_probabilities:
            self.team2_percentage.set(f"{self.team2.get()} has an assigned {predictions_probabilities[0]:.2%} to win")
            self.team1_percentage.set(f"{self.team1.get()} has an assigned {predictions_probabilities[1]:.2%} to win")
        else:
            self.team1_percentage.set("Percentages could not be predicted with the chosen model")
            self.team2_percentage.set("")


    def get_team_data(self):
        """Gets the team data based on what is selected"""
        year = 2023
        query = (
            "SELECT "
                "seast_team1.`AdjEM` AS `TEAM_1_AdjEM`, "
                "seast_team1.`AdjO` AS `TEAM_1_AdjO`, "
                "seast_team1.`AdjD` AS `TEAM_1_AdjD`, "
                "seast_team1.`AdjT` AS `TEAM_1_AdjT`, "
                "seast_team1.`Luck` AS `TEAM_1_Luck`, "
                "seast_team1.`Strength of Schedule_AdjEM` AS `TEAM_1_Strength of Schedule_AdjEM`, "
                "seast_team1.`Strength of Schedule_OppO` AS `TEAM_1_Strength of Schedule_OppO`, "
                "seast_team1.`Strength of Schedule_OppD` AS `TEAM_1_Strength of Schedule_OppD`, "
                "seast_team1.`NCSOS_AdjEM` AS `TEAM_1_NCSOS_AdjEM`, "
                "seast_team2.`AdjEM` AS `TEAM_2_AdjEM`, "
                "seast_team2.`AdjO` AS `TEAM_2_AdjO`, "
                "seast_team2.`AdjD` AS `TEAM_2_AdjD`, "
                "seast_team2.`AdjT` AS `TEAM_2_AdjT`, "
                "seast_team2.`Luck` AS `TEAM_2_Luck`, "
                "seast_team2.`Strength of Schedule_AdjEM` AS `TEAM_2_Strength of Schedule_AdjEM`, "
                "seast_team2.`Strength of Schedule_OppO` AS `TEAM_2_Strength of Schedule_OppO`, "
                "seast_team2.`Strength of Schedule_OppD` AS `TEAM_2_Strength of Schedule_OppD`, "
                "seast_team2.`NCSOS_AdjEM` AS `TEAM_2_NCSOS_AdjEM` "
            "FROM kenpom_stats seast_team1, kenpom_stats seast_team2 "
            f"WHERE seast_team1.Year = {year} AND seast_team1.Team = '{self.team1.get()}' AND seast_team2.Year = {year} AND seast_team2.Team = '{self.team2.get()}' "
        )


        df = get_data(query,True)
        df = clean_data(df)

        return df


if __name__ == "__main__":
    application = GUI()

    application.run()
