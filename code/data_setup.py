import matplotlib.pyplot as plt
import pandas as pd
from source import compile_data
from si_model import extract_columns
import os
import pickle

def train_test_split(data, train_proportion):
    split_index = int(train_proportion * len(data))
    training_end_date = data.loc[split_index, 'date']
    # Getting nearest month end
    training_end_date = pd.Timestamp(training_end_date).to_period('M').end_time
    training_data = data[data['date'] <= training_end_date]
    test_data = data[data['date'] > training_end_date]
    return training_data, test_data

def plot_train_data(f, path, plot):
    f.plot()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    if plot:
        plt.show()

def plot_seasonal_train_data(f, path, plot):
    f.seasonal_decompose().plot()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    if plot:
        plt.show()

def plot_ground_truth(data, plot_save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['new_cases'], marker='o', label='New Cases')
    plt.title('New Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = plot_save_path + f'ground_truth.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.show()

def save_forecaster(f, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(f, file)

def load_forecaster(path):
    with open(path, 'rb') as file:
        forecaster = pickle.load(file)
    return forecaster

def get_data():
    plot_save_path = '../plots/LSTM/'
    forecaster_file_path = '../data/forecaster.pkl'
    file_paths = (plot_save_path, forecaster_file_path)
    data = compile_data()
    SI_data = extract_columns(data)
    train_data, test_data = train_test_split(SI_data, 0.75)
    days_to_pred = len(test_data)
    train_test_data = (train_data, test_data)
    return file_paths, train_test_data, days_to_pred

if __name__ == "__main__":
    paths, train_test_data, days_to_pred = get_data()
    plot_save_path, forecaster_file_path = paths
    train_data, test_data = train_test_data
    print(train_data.head())