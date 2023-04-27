"""Given Bracket Matchups in the main, this will help predict all games"""
from machine_learning import load_model, clean_data, get_data

# Model name goes here
MODEL_FILE = ""
MODEL = load_model(MODEL_FILE, True)

YEAR = 2023

MIDWEST = [
    'Houston', 'Northern Kentucky',
    'Iowa', 'Auburn',
    'Miami FL', 'Drake',
    'Indiana', 'Kent St.',
    'Iowa St.', 'Pittsburgh',
    'Xavier', 'Kennesaw St.',
    'Texas A&M', 'Penn St.',
    'Texas', 'Colgate'
]

WEST = [
    'Kansas', 'Howard',
    'Arkansas', 'Illinois',
    "Saint Mary''s", "VCU",
    "Connecticut", "Iona",
    "TCU", "Arizona St.",
    "Gonzaga", "Grand Canyon",
    "Northwestern", "Boise St.",
    "UCLA", "UNC Asheville"
]

SOUTH = [
    "Alabama", "Texas A&M Corpus Chris",
    "Maryland", "West Virginia",
    "San Diego St.", "Charleston",
    "Virginia", "Furman",
    "Creighton", "N.C. State",
    "Baylor","UC Santa Barbara",
    "Missouri", "Utah St.",
    "Arizona", "Princeton"
]

EAST = [
    "Purdue", "Fairleigh Dickinson",
    "Memphis", "Florida Atlantic",
    "Duke", "Oral Roberts",
    "Tennessee", "Louisiana",
    "Kentucky", "Providence",
    "Kansas St.", "Montana St.",
    "Michigan St.", "USC", 
    "Marquette", "Vermont"
]

def make_prediction(team1: str, team2: str) -> str:
    """Makes one prediction between the two teams provided

    :param team1: Team 1 in the game
    :type team1: str
    :param team2: Team 2 in the game
    :type team2: str
    :raises ValueError: Thrown if team cannot be found in dataframe
    :return: Team that won the match
    :rtype: str
    """
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
            "seast_team1.`NCSOS_AdjEM` AS `TEAM_1_NCSOS_AdjEM`,  "
            "seast_team2.`AdjEM` AS `TEAM_2_AdjEM`, "
            "seast_team2.`AdjO` AS `TEAM_2_AdjO`, "
            "seast_team2.`AdjD` AS `TEAM_2_AdjD`, "
            "seast_team2.`AdjT` AS `TEAM_2_AdjT`, "
            "seast_team2.`Luck` AS `TEAM_2_Luck`, "
            "seast_team2.`Strength of Schedule_AdjEM` AS `TEAM_2_Strength of Schedule_AdjEM`, "
            "seast_team2.`Strength of Schedule_OppO` AS `TEAM_2_Strength of Schedule_OppO`, "
            "seast_team2.`Strength of Schedule_OppD` AS `TEAM_2_Strength of Schedule_OppD`, "
            "seast_team2.`NCSOS_AdjEM` AS `TEAM_2_NCSOS_AdjEM`  "
        "FROM kenpom_stats seast_team1, kenpom_stats seast_team2 "
        f"WHERE seast_team1.Year = {YEAR} AND seast_team1.Team = '{team1}' AND seast_team2.Year = {YEAR} AND seast_team2.Team = '{team2}' "
    )

    df = get_data(query,True)
    df = clean_data(df)

    if not df.empty:
        predictions = MODEL.predict(df)[0]

        if predictions == 1:
            print(f"{team1} is predicited to beat {team2}")
            return team1
        else:
            print(f"{team2} is predicited to beat to {team1}")
            return team2
    else:
        raise ValueError(f"ERROR: Could not find one of {team1} or {team2}")

def predict_region(region: str, round_of_64: list[str]) -> str:
    """Loops for predicting an entire region

    :param region: Which region is being predicted. Used for printing outputs.
    :type region: str
    :param round_of_64: Teams playing eachother in the round of 64. Teams playing eachother must be consectutive. 
    :type round_of_64: list[str]
    :return: winner of the region
    :rtype: str
    """

    round_of_32 = []
    sweet_sixteen = []
    elite_eight = []

    print(f"\n\n{region}: Round of 64")
    for i in range(0, len(round_of_64),2):
        round_of_32.append(make_prediction(round_of_64[i], round_of_64[i+1]))

    print(f"\n{region}: Round of 32")
    for i in range(0, len(round_of_32),2):
        sweet_sixteen.append(make_prediction(round_of_32[i], round_of_32[i+1]))

    print(f"\n{region}: Sweet Sixteen")
    for i in range(0, len(sweet_sixteen),2):
        elite_eight.append(make_prediction(sweet_sixteen[i], sweet_sixteen[i+1]))

    print(f"\n{region}: Elite Eight")
    for i in range(0, len(elite_eight),2):
        return make_prediction(elite_eight[i], elite_eight[i+1])


def predict_finals(final_four: list[str]) -> str:
    """Predicts the finals.

    :param final_four: List of teams playing in the final 4
    :type final_four: list[str]
    :return: National Champion
    :rtype: str
    """
    finals = []

    print("Final Four")
    for i in range(0, len(final_four), 2):
        finals.append(make_prediction(final_four[i], final_four[i+1]))

    print("\nFinals")
    for i in range(0, len(finals), 2):
        return make_prediction(finals[i], finals[i+1])

def main():
    """Main Function"""
    final_four = [
        predict_region('Midwest', MIDWEST), predict_region('West', WEST),
        predict_region('South', SOUTH), predict_region('East', EAST),
    ]

    winner = predict_finals(final_four)

    print(f"{winner} is predicted to win it all!")

if __name__ == "__main__":
    main()
