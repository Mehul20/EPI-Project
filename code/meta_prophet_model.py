from mobility_model import clean_data
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import pandas as pd

def clean_data_for_prophet(data):
    data = data[["date", "new_cases", "mobility_data", "people_fully_vaccinated", "stringency_index", "human_development_index", "diabetes_prevalence", "gdp_per_capita", "new_vaccinations"]]
    data = data.rename(columns={"date": "ds", "new_cases": "y"})
    data = data.dropna()
    return data

def plot_meta_prophet(test_data, speculation, item):
    plt.figure(figsize=(12, 6))
    
    plt.plot(test_data["ds"], test_data["y"], label="Ground Truth Data", color = "green")
    plt.xlabel("Month and Year")
    
    plt.plot(test_data["ds"], speculation["yhat"], label="Meta Prophet Prediction", color="red")
    plt.ylabel("New Covid - 19 Cases")
    
    plt.title("Speculation vs Ground Truth with Meta Prophet Model")
    
    sub_title = ""
    for curr_item in item:
        sub_title += curr_item + ", "
    sub_title = sub_title[:len(sub_title) - 2]
    plt.suptitle("Exogenous Variables used are: " + sub_title, fontsize = 12)

    plt.legend()
    plt.grid()
    plt.savefig("../plots/Prophet/" + str(sub_title) + "-image.png", format = "png")
    plt.show()

def regressor_addition(meta_prophet_model, regressors):
    for regressor in regressors:
        meta_prophet_model = meta_prophet_model.add_regressor(regressor, standardize = True)
    return meta_prophet_model

def run_meta_prophet_model(data):
    data = clean_data_for_prophet(data)
    
    # Training data is 2 years 6 months and Test data is for 6 months
    training_data = data[(data["ds"] >= pd.to_datetime("2020-01-01")) & (data["ds"] <= pd.to_datetime("2022-06-30"))]
    test_data = data[(data["ds"] >= pd.to_datetime("2022-07-01"))]

    all_exog = ["mobility_data", "people_fully_vaccinated", "stringency_index", "human_development_index", "diabetes_prevalence", "gdp_per_capita"]
    case1 = [all_exog[1], all_exog[2]]
    case2 = [all_exog[2], all_exog[-1]]
    case3 = [all_exog[3], all_exog[0]]
    for item in [all_exog, case1, case2, case3]:
        meta_prophet_model = Prophet(weekly_seasonality = True)
        meta_prophet_model = regressor_addition(meta_prophet_model, item)
        meta_prophet_model.fit(training_data)
        speculation = meta_prophet_model.predict(test_data)
        plot_meta_prophet(test_data, speculation, item)

def analysis_meta_prophet(data):
    data = clean_data_for_prophet(data)
    all_vals = []
    
    # Training data is 2 years 6 months and Test data is for 6 months
    training_data = data[(data["ds"] >= pd.to_datetime("2020-01-01")) & (data["ds"] <= pd.to_datetime("2022-06-30"))]
    test_data = data[(data["ds"] >= pd.to_datetime("2022-07-01"))]

    exog_vars = ["new_vaccinations","mobility_data","stringency_index","human_development_index","diabetes_prevalence","gdp_per_capita"]
    for item in exog_vars:
        item_arr = [item]
        meta_prophet_model = Prophet(weekly_seasonality = True)
        meta_prophet_model = regressor_addition(meta_prophet_model, item_arr)
        meta_prophet_model.fit(training_data)
        speculation = meta_prophet_model.predict(test_data)
        rmse = root_mean_squared_error(test_data["y"], speculation["yhat"])
        all_vals.append([item_arr, rmse])
    return all_vals

if __name__ == "__main__":
    data = clean_data()
    run_meta_prophet_model(data)
    all_vals = analysis_meta_prophet(data)
    print(all_vals)