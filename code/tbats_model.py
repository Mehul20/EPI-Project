from mobility_model import clean_data
from sktime.forecasting.tbats import TBATS
import pandas as pd
import matplotlib.pyplot as plt

def clean_tbats_data():
    data = clean_data()
    tbats_data = data[["date", "new_cases"]]
    tbats_data = tbats_data.set_index("date")
    tbats_data = tbats_data.asfreq("W")
    tbats_data["new_cases"] = tbats_data["new_cases"].interpolate(method = "time").fillna(0)
    return tbats_data

def tbats_graphing(test_data, speculation):
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data["new_cases"], label="Ground Truth Data", color = "green")
    plt.xlabel("Weekly Data")
    
    plt.plot(test_data.index, speculation["new_cases"], label="TBATS model prediction", color="red")
    plt.ylabel("New Covid - 19 Cases")
    
    plt.title("Speculation vs Ground Truth with TBATS model")
    plt.legend()
    plt.grid()
    plt.savefig("../plots/tbats/image.png", format = "png")
    plt.show()

def tbats_processing():   
    data = clean_tbats_data()

    training_data = data[(data.index >= pd.to_datetime("2020-01-01")) & (data.index <= pd.to_datetime("2022-09-30"))]
    test_data = data[(data.index >= pd.to_datetime("2022-10-01"))]

    TBATS_model = TBATS(sp = 7)
    fit_TBATS = TBATS_model.fit(training_data)
   
    predictions_for = []
    for i in range(len(test_data)):
        predictions_for.append(i)

    prediction = fit_TBATS.predict(fh = predictions_for)

    tbats_graphing(test_data, prediction)

if __name__ == "__main__":
    tbats_processing()