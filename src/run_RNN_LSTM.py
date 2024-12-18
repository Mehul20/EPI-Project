import matplotlib.pyplot as plt
import pandas as pd
import os
from source import compile_data
from si_model import extract_columns
from scalecast.Forecaster import Forecaster
from LSTM_RNN import LSTM_1_layer, LSTM_2_layer, RNN_1_layer, RNN_2_layer

def train_test_split(data, train_proportion):
    split_index = int(train_proportion * len(data))
    training_end_date = pd.Timestamp(data.loc[split_index, 'date']).to_period('M').end_time
    training_data = data[data['date'] <= training_end_date]
    test_data = data[data['date'] > training_end_date]
    return training_data, test_data

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

def get_data():
    plot_save_path = '../plots/RNN/'
    data = compile_data()
    SI_data = extract_columns(data)
    train_data, test_data = train_test_split(SI_data, 0.75)
    train_test_data = (train_data, test_data)
    return plot_save_path, data, train_test_data

def Xvar_and_estimator(f):
    f.auto_Xvar_select(
        try_trend = False,
        irr_cycles = [120,132,144],
        cross_validate = True,
        cvkwargs = {'k':3},
        dynamic_tuning = 240,
    )
    f.set_estimator('rnn')
    return f

def setup_Forecaster():
    path, data, train_test_data = get_data()
    train_data, test_data = train_test_data
    f_model = Forecaster(y=data['new_cases'], current_dates=data['date'], test_length=len(test_data), future_dates=len(test_data), cis=True)
    f = Xvar_and_estimator(f_model)
    return f, path

def run_LSTM():
    f, path = setup_Forecaster()
    LSTM_1_layer(f)
    LSTM_2_layer(f)
    f.plot_test_set(ci=False, models=['LSTM_1_layer','LSTM_2_layer'], order_by='TestSetRMSE')
    plt.title('LSTM Models Test-Set Results',size=16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + f'LSTM.png')
    plt.show()

def run_RNN():
    f, path = setup_Forecaster()
    RNN_1_layer(f)
    RNN_2_layer(f)
    f.plot_test_set(ci=False, models=['RNN_1_layer','RNN_2_layer'], order_by='TestSetRMSE')
    plt.title('SimpleRNN Models Test-Set Results',size=16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + f'RNN.png')
    plt.show()

def plot_all():
    f, path = setup_Forecaster()
    RNN_1_layer(f)
    RNN_2_layer(f)
    LSTM_1_layer(f)
    LSTM_2_layer(f)
    f.plot_test_set(models=['RNN_1_layer','RNN_2_layer','LSTM_1_layer','LSTM_2_layer'], order_by='TestSetRMSE', include_train=False)
    plt.title('All Models Performance - Test Set Observations', size=16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path + f'all_models.png')
    plt.show()
    return f

def export_results(f):
    pd.set_option('display.float_format',  '{:.4f}'.format)
    f.export('model_summaries', models=['RNN_1_layer','RNN_2_layer','LSTM_1_layer','LSTM_2_layer'],
             determine_best_by = 'TestSetRMSE',
             to_excel=True)[['ModelNickname','TestSetRMSE','TestSetR2','InSampleRMSE','InSampleR2','best_model']]

if __name__ == "__main__":
    run_RNN()
    run_LSTM()
    f = plot_all()
    export_results(f)