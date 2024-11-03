from source import compile_data
import matplotlib.pyplot as plt
import pandas as pd

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
        'date' : 'last'
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

def run_ARIMA_model(data):
    print(data.head())

if __name__ == "__main__":
    data = clean_data()
    run_ARIMA_model(data)
    #graph_plots(data)