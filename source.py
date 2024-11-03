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

if __name__ == "__main__":
    filePath = "cowid-covid-data.csv"
    data = read_data(filePath)
    US_data = get_USA_data(data)
    filtered_data = remove_columns(US_data)
    print(filtered_data)