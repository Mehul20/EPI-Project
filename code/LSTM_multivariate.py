import os
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from LSTM_setup import get_data
from scalecast.util import find_optimal_transformation, gen_rnn_grid
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

def create_forecaster(data, num_days_pred):
    dates = data['date']
    f = Forecaster(y=data['new_cases'], current_dates=dates, future_dates=num_days_pred)
    f_d = Forecaster(y=data['new_deaths'], current_dates=dates, future_dates=num_days_pred)
    f_v = Forecaster(y=data['people_fully_vaccinated'], current_dates=dates, future_dates=num_days_pred)
    # f_m = Forecaster(y=data['avg_USA'], current_dates=dates, future_dates=num_days_pred)
    f_m = None
    # f.add_series(data['new_deaths'], called='new_deaths')
    # f.add_series(data['people_fully_vaccinated'], called='vaccinated_people')
    # f.add_series(data['avg_USA'], called='mobility')
    # plot_train_data(f, plot_save_path, plot)
    # plot_seasonal_train_data(f, plot_save_path, plot)
    return f, f_d, f_v, f_m

def get_transformer_reverter(f):
    transformer, reverter = find_optimal_transformation(f, set_aside_test_set=False, return_train_only=False, verbose=True,\
                                                        detrend_kwargs=[{'loess':True},{'poly_order':1},{'ln_trend':True}],\
                                                        m=365, test_length=10)
    return transformer, reverter

def get_rnn_grid():
    rnn_grid = gen_rnn_grid(layer_tries=10, min_layer_size=3, max_layer_size=5, units_pool=[100], epochs=[100], dropout_pool=[0, 0.05],\
                            validation_split =0.2, callbacks=EarlyStopping(monitor='val_loss', patience=3), random_seed=20)
    return rnn_grid

def forecaster(f, f_d, f_v, f_m):
    f.set_estimator('naive')
    f.manual_forecast()
    # univariate lstm model
    f.add_ar_terms(13)
    f.set_estimator('rnn')
    f.ingest_grid(rnn_grid)
    f.tune()
    f.auto_forecast(call_me='lstm_univariate')
    # multivariate lstm model
    f.add_series(f_d.y,called='deaths')
    f.add_series(f_v.y,called='vaccinations')
    # f.add_series(f_m.y,called='mobility')
    f.add_lagged_terms('deaths',lags=13,drop=True)
    f.add_lagged_terms('vaccinations',lags=13,drop=True)
    f.ingest_grid(rnn_grid)
    f.tune()
    f.auto_forecast(call_me='lstm_multivariate')

if __name__ == "__main__":
    paths, train_test_data, days_to_pred = get_data()
    plot_save_path, forecaster_file_path = paths
    train_data, test_data = train_test_data
    f, f_d, f_v, f_m = create_forecaster(train_data, days_to_pred)
    transformer, reverter = get_transformer_reverter(f)
    f_d = transformer.fit_transform(f_d)
    f_v = transformer.fit_transform(f_v)
    # f_m = transformer.fit_transform(f_m)
    f = transformer.fit_transform(f)
    rnn_grid = get_rnn_grid()
    forecaster(f, f_d, f_v, f_m)
    print("\n\n\n\n\n")
    f = reverter.fit_transform(f)
    f.plot_test_set(order_by='TestSetRMSE')
    # plt.savefig('LSTM MV test results.png')
    plt.show()
    pd.options.display.float_format = '{:,.4f}'.format
    summ = f.export('model_summaries',determine_best_by='TestSetRMSE')
    print(summ[['ModelNickname','TestSetRMSE','TestSetR2']])
    print(summ[['ModelNickname','HyperParams']].style.set_properties(height = 5))
    f.plot(order_by='TestSetRMSE')
    plt.show()