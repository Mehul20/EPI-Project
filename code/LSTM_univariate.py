from scalecast.Forecaster import Forecaster
from scalecast.Pipeline import Transformer, Reverter, Pipeline
from LSTM_setup import save_forecaster, load_forecaster, get_data
from LSTM_setup import plot_train_data, plot_seasonal_train_data, plot_ground_truth
import os
import matplotlib.pyplot as plt

def create_forecaster(data, num_days_pred, plot_save_path, plot):
    dates = data['date']
    new_cases = data['new_cases']
    f = Forecaster(y=new_cases, current_dates=dates, future_dates=num_days_pred)
    plot_train_data(f, plot_save_path + f'univariate_F.png', plot)
    plot_seasonal_train_data(f, plot_save_path + f'seasonal_uni_F.png', plot)
    return f

def plot_uni_F_pred(f, path, plot):
    f.plot()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    if plot:
        plt.show()

def forecaster(f):
    f.set_estimator('rnn')
    f.manual_forecast(lags=18, layers_struct=[('LSTM', {'units': 64, 'activation': 'tanh'})], epochs=5000, call_me='lstm')
    
def get_univariate_forecast(f, plot_save_path, plot):
    transformer = Transformer(transformers = [('DetrendTransform',{'poly_order':2}), 'DeseasonTransform'])
    reverter = Reverter(reverters = ['DeseasonRevert', 'DetrendRevert'], base_transformer = transformer)
    pipeline = Pipeline(steps = [('Transform',transformer), ('Forecast', forecaster), ('Revert',reverter)])
    f = pipeline.fit_predict(f)
    plot_uni_F_pred(f, plot_save_path + f'univariate_F_pred.png', plot)
    return f

def get_fitted_vals(f):
    return f.export_fitted_vals('lstm')

def run_uni_LSTM():
    paths, train_test_data, days_to_pred = get_data()
    plot_save_path, forecaster_file_path = paths
    train_data, test_data = train_test_data
    f_train = create_forecaster(train_data, days_to_pred, plot_save_path, False)
    f_pred = get_univariate_forecast(f_train, plot_save_path, True)
    save_forecaster(f_pred, forecaster_file_path)
    f = load_forecaster(forecaster_file_path)
    plot_ground_truth(test_data, plot_save_path)
    fitted_vals = get_fitted_vals(f)

if __name__ == "__main__":
    run_uni_LSTM()