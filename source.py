import pandas as pd

def read_data(filePath):
    data = pd.read_csv(filePath)
    return data

def get_USA_data(data):
    return data[data['iso_code'] == 'USA']

def remove_columns(data):
    columns_to_drop = ['iso_code', 'continent']
    data = data.drop(columns_to_drop, axis=1)
    return data

def clean_case_data():
    filePath = "cowid-covid-data.csv"
    case_data = read_data(filePath)
    US_data = get_USA_data(case_data)
    filtered_case_data = remove_columns(US_data)
    return filtered_case_data

def clean_twitter_mobility_data():
    filePath = "mobility.csv"
    mobility_data = read_data(filePath)
    return mobility_data

if __name__ == "__main__":
    case_data = clean_case_data()
    mobility_data = clean_twitter_mobility_data()