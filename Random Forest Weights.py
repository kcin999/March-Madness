"""A Script to help associate Random Forest's Features with the Column Names

To Run:

    * Put the query used in line 11 as a string
    * Put the Feature Importance List as a string in line 15
    * Run the file.
"""
from prettytable import PrettyTable
from machine_learning import get_data

# Query Goes Here
QUERY = ""

# String of feature importances.
FEATURE_IMPORTANCE_STRING = ""

def main():
    """Main Function"""
    # Gets Column list in the order retrieved
    df = get_data(QUERY, True)
    df = df.drop(columns=["Result"])
    column_list = df.columns.tolist()

    # Converts Feature Importance String into a list of floats.
    feature_importance_string = FEATURE_IMPORTANCE_STRING.replace('[', '')
    feature_importance_string = feature_importance_string.replace(']', '')
    feature_importance_string_list = feature_importance_string.split(' ')
    column_weight_list = []
    for weight in feature_importance_string_list:
        if weight != '':
            column_weight_list.append(float(weight))

    # Verification Step to make sure there are the same number of columns as weights
    if len(column_weight_list) != len(column_list):
        raise ValueError("Mismatching number of Feature Importances and Columns in Query")

    # Creates the Table which will then be printed.
    table = PrettyTable()
    table.field_names = ["Feature", "Weight"]
    for weight in column_weight_list:
        table.add_row([column_list[column_weight_list.index(weight)], weight])

    # Table Parameters used to sort the table
    table.sortby='Weight'
    table.reversesort = True

    print(table)

if __name__ == "__main__":
    main()
