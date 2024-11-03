from source import compile_data
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def clean_data():
    data = compile_data()
    data = data.sort_values(by='date')
    shrink_data = data.groupby(["year", "week"]).agg({
        'new_cases' : 'sum',
        'new_deaths' : 'sum',
        'new_vaccinations': 'sum',
        'year': 'first',
        'week': 'first',
        'avg_USA': 'first',
        'date': 'last'
    })
    shrink_data = shrink_data.rename(columns={'avg_USA': 'mobility_data'})
    shrink_data = shrink_data.dropna(subset=["mobility_data"])
    shrink_data.to_csv('../data/weekly_cleaned_data.csv', index=False)
    return shrink_data

def plot_data(data, parameter):
    print_parameter = None
    if parameter == "new_cases":
        print_parameter = "New Cases"
    elif parameter == "new_deaths":
        print_parameter = "New Deaths"
    
    for year in [2020, 2021, 2022]:
        current_year_data = data[data['year'] == year]
        
        _, axis_left = plt.subplots(figsize=(12, 6))

        axis_left.plot(current_year_data["week"], current_year_data["mobility_data"], label="Mobility Data", color = "green")
        axis_left.set_ylabel("Mobility Data", color = "green")
        axis_left.set_title("Weekly Mobility and " + str(print_parameter) + "data for " + str(year))

        axis_right = axis_left.twinx()
        axis_right.plot(current_year_data['week'], current_year_data[parameter], label=print_parameter, color = "red")
        axis_right.set_ylabel(print_parameter, color = "red")

        axis_left.grid(True)
        axis_left.set_xlabel("Week")

        plt.savefig("../plots/mobility/" + str(year) + "-" + str(parameter) + "-mobility-plot.png", format = "png")
        plt.show()

def graph_plots(data):
    plot_data(data, "new_cases")
    plot_data(data, "new_deaths")

def set_index_on_date(data):
    data["date"] = pd.to_datetime(data["date"])
    data.set_index('date', inplace=True)
    data = data.asfreq('W')
    return data

def plot_seasonal_ARIMA(test_data, speculation):
    plt.figure(figsize=(12, 6))
    
    plt.plot(test_data.index, test_data["new_cases"], label="Ground Truth Data", color = "green")
    plt.xlabel("Week and Year")
    
    plt.plot(test_data.index, speculation, label="Seasonal ARIMA Prediction", color="red")
    plt.ylabel("New Covid - 19 Cases")
    
    plt.title("Speculation vs Ground Truth with Seasonal ARIMA Model")
    plt.legend()
    plt.savefig("../plots/SARIMA/image.png", format = "png")
    plt.show()

def run_Seasonal_ARIMA_model(data):
    data = set_index_on_date(data)

    training_data_len = int(0.7 * len(data))
    data["mobility_data"].interpolate(method="linear", inplace=True)

    training_data = data[:training_data_len].astype(float)
    test_data = data[training_data_len:].astype(float)

    seasonal_arima_model = SARIMAX(training_data["new_cases"], exog=training_data["mobility_data"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    fit_model = seasonal_arima_model.fit()

    speculation = fit_model.forecast(steps = len(test_data), exog=test_data["mobility_data"])
    plot_seasonal_ARIMA(test_data, speculation)

if __name__ == "__main__":
    data = clean_data()
    run_Seasonal_ARIMA_model(data)
    #graph_plots(data)