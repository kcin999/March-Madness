from machine_learning import load_model, get_team_data, clean_data

season_averages = True
team_1 = ('',)
team_2 = ('',)


df = get_team_data(team_1, team_2, 2023,season_averages=season_averages)
df = clean_data(df, season_averages)

model = load_model()

predictions = model.predict(df)
print(predictions)

for i, _team_1_name in enumerate(team_1):
    if predictions[i] == 1:
        print(f"{team_1[i]} is predicited to beat {team_2[i]}")
    else:
        print(f"{team_1[i]} is predicited to lose to {team_2[i]}")
