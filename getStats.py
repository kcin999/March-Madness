from sportsipy.ncaab.teams import Teams
import pandas


def getData():      
    for year in range(2015,2022):
        firstTime=True
        teamDataFrame = pandas.DataFrame
        for team in Teams(year=year):
            try:
                if firstTime:
                    teamDataFrame = team.dataframe
                    firstTime = False
                else:
                    teamDataFrame = pandas.concat([teamDataFrame, team.dataframe],axis=0)
            except:
                print("Skipping " + team.abbreviation)

        teamDataFrame["Year"] = year
        teamDataFrame.to_csv(str(year) + "_SeasonStats.csv")

if __name__ == "__main__":
    getData()