from mobility_model import clean_data
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd

def clean_data_for_prophet(data):
    data = data[["date", "new_cases", "mobility_data", "people_fully_vaccinated", "stringency_index"]]
    data = data.rename(columns={"date": "ds", "new_cases": "y"})
    data = data.dropna()
    return data

def plot_meta_prophet(test_data, speculation):
    plt.figure(figsize=(12, 6))
    
    plt.plot(test_data["ds"], test_data["y"], label="Ground Truth Data", color = "green")
    plt.xlabel("Month and Year")
    
    plt.plot(test_data["ds"], speculation["yhat"], label="Meta Prophet Prediction", color="red")
    plt.ylabel("New Covid - 19 Cases")
    
    plt.title("Speculation vs Ground Truth with Meta Prophet Model")
    plt.legend()
    plt.grid()
    plt.savefig("../plots/Prophet/image.png", format = "png")
    plt.show()

def run_meta_prophet_model(data):
    data = clean_data_for_prophet(data)
    
    # Training data is 2 years 6 months and Test data is for 6 months
    training_data = data[(data["ds"] >= pd.to_datetime("2020-01-01")) & (data["ds"] <= pd.to_datetime("2022-06-30"))]
    test_data = data[(data["ds"] >= pd.to_datetime("2022-07-01"))]

    meta_prophet_model = Prophet(weekly_seasonality = True)

    # We can also take into account the mobility_data.
    # meta_prophet_model.add_regressor("mobility_data", standardize = True)

    meta_prophet_model.add_regressor("people_fully_vaccinated", standardize = True)
    meta_prophet_model.add_regressor("stringency_index", standardize = True)

    meta_prophet_model.fit(training_data)
    speculation = meta_prophet_model.predict(test_data)
    plot_meta_prophet(test_data, speculation)

if __name__ == "__main__":
    data = clean_data()
    run_meta_prophet_model(data)