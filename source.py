import pandas as pd

def read_data(filePath):
    data = pd.read_csv(filePath)
    return data

filePath = "cowid-covid-data.csv"
data = read_data(filePath)