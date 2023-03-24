from machine_learning import load_model, get_team_data, clean_data

season_averages = False
team_1 = ('TCU', 'Xavier', 'Illinois', 'Indiana')
team_2 = ('Gonzaga', 'Pittsburgh', 'Arkansas', 'Miami FL')


df = get_team_data(team_1, team_2, 2023, 2, 8, season_averages)
# df = clean_data(df, season_averages)

model = load_model('logistic_regression.skops')

predictions = model.predict(df)
predictions_probabilities = model.predict_proba(df)

for i, _team_1_name in enumerate(team_1):
    print('\n\n\n')
    if predictions[i] == 1:
        print(f"{team_1[i]} is predicited to beat {team_2[i]}")
    else:
        print(f"{team_1[i]} is predicited to lose to {team_2[i]}")

    print(f"{team_2[i]} has an assigned {predictions_probabilities[i][0]:.2%} to win")
    print(f"{team_1[i]} has an assigned {predictions_probabilities[i][1]:.2%} to win")
