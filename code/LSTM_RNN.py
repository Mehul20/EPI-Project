def RNN_1_layer(f):
    f.manual_forecast(
        layers_struct=[('SimpleRNN',{'units':100,'dropout':0.2})],
        epochs=50,
        validation_split=0.2,
        plot_loss=False,
        call_me="RNN_1_layer",
    )

def RNN_2_layer(f):
    f.manual_forecast(
        layers_struct=[('SimpleRNN',{'units':100,'dropout':0})] * 2 + [('Dense',{'units':10})] * 2,
        epochs=50,
        random_seed=42,
        plot_loss=False,
        validation_split=0.2,
        call_me='RNN_2_layer',
    )

def LSTM_1_layer(f):
    f.manual_forecast(
        layers_struct=[('LSTM',{'units':100,'dropout':0})],
        epochs=50,
        plot_loss=False,
        validation_split=0.2,
        call_me='LSTM_1_layer',
    )

def LSTM_2_layer(f):
    f.manual_forecast(
        layers_struct=[('LSTM',{'units':100,'dropout':0})] * 2 + [('Dense',{'units':10})] * 2,
        epochs=50,
        random_seed=42,
        plot_loss=False,
        validation_split=0.2,
        call_me='LSTM_2_layer',
    )