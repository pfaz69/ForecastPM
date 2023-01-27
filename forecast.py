import numpy as np
from numpy import concatenate
from numpy.random import seed
from tensorflow import random as rnd
from math import sqrt
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from easyesn import PredictionESN
#from easyesn import backend as B
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from tensorflow_addons.layers import ESN

import keras.losses
from statsmodels.tsa.ar_model import AutoReg

# convert series to supervised learning
def _series_to_supervised(data, n_in=1, n_out=1):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input.
		n_out: Number of observations as output.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    num_graphs = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
	
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        list_col = [df[i] for i in range(1, data.shape[1]) ]
        cols.append(DataFrame([df[0].shift(i), *list_col]).T)
        # Put names for debugging
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(num_graphs)]
	
    # Output sequence (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        # Put names for debugging
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(num_graphs)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(num_graphs)]
	
    # Aggregate
    agg = concat(cols, axis=1)
    agg.columns = names

	# Drop rows with NaN values
    agg.dropna(inplace=True)

    # Drop CAMS from labels
    for i in range(n_out):
        col_drop_start = n_in*num_graphs + 1 + i
        col_drop_end = (n_in + 1)*num_graphs + i
        agg.drop(agg.columns[list(range(col_drop_start, col_drop_end))], axis=1, inplace=True)

    return agg


def prep_data(dtf_obs, dtf_CAMS):
    # Arrange Data to Focus ON, excluding CAM1-3-7
    time = 1 # 0-> t=0 1->t=24h, 2->t=48h
    #list_cams = [dtf_CAMS[i][time] for i in range(9)if i < 2] 
    list_cams = [dtf_CAMS[i][time] for i in range(len(dtf_CAMS))] #<- all cams
    #list_cams = [dtf_CAMS[i][time] for i in range(9) if i == 3 or i == 4 or i == 5 or i == 7] #if i == 0] #
    #list_cams = [dtf_CAMS[i][time] for i in range(9) if i == 4] #if i == 0] #
    #list_cams = []

    dataset = concat(	
                        [
                            dtf_obs, 
                            *list_cams
                        ], 
                        axis=1
                    )



    ds_values = dataset.values

    # Debug
    arr_nans = np.argwhere(np.isnan(np.array(ds_values)))
    assert(arr_nans.size == 0)

    return ds_values

def run_forecasting_algs(
                            ids_input,
                            design, 
                            ds_values, 
                            test_set_size, 
                            n_input_steps,
                            n_output_steps
                        ):
    
    # To get reproducible results with Keras/ESN
    if design == "esn_easy":
        seed(0)
    else:
        rnd.set_seed(0)
    
    
    
    
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(ds_values)

    # Prepare data for training and testing
    reframed = _series_to_supervised(scaled, n_input_steps, n_output_steps)
    refr_values = reframed.values

    # Split Dataset in Train and Test
    train, test = refr_values[0:-test_set_size], refr_values[-test_set_size:]
    
    # Split into input (all columns except the last ones) and outputs (latest columns)
    train_X, train_y = train[:, :-n_output_steps], train[:, -n_output_steps:]
    test_X, test_y = test[:, :-n_output_steps], test[:, -n_output_steps:] #test[:, 0]#

    if design != "esn_easy" and design != "nvar":
        # Reshaping is needed only for keras
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  
    
    # Python 3.7 doesn't have "match" yet so use if-else block instead 
    if design == 'lstm_1':
        neurons_0 = 1
        neurons_1 = 1
        neurons_2 = 1

        batch_size = 20
        epochs = 400#78
        model = Sequential()
        model.add(LSTM(neurons_0, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(LSTM(neurons_1, return_sequences=True))
        model.add(LSTM(neurons_2))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='huber', optimizer='adam')
    elif design == 'lstm_2':
        # design 2
        neurons_1 = 1
        batch_size = 20
        epochs = 400
        model = Sequential()
        model.add(LSTM(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'lstm_3':
        # design 3
        neurons_1 = 1
        batch_size = 20
        epochs = 200
        model = Sequential()
        model.add(LSTM(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    if design == 'gru_1':
        neurons_0 = 1
        neurons_1 = 1
        neurons_2 = 1

        batch_size = 20
        epochs = 400#78
        model = Sequential()
        model.add(GRU(neurons_0, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(GRU(neurons_1, return_sequences=True))
        model.add(GRU(neurons_2))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='huber', optimizer='adam')
    elif design == 'gru_2':
        # design 2
        neurons_1 = 1
        batch_size = 20
        epochs = 400
        model = Sequential()
        model.add(GRU(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'gru_3':
        # design 3
        neurons_1 = 1
        batch_size = 20
        epochs = 200
        model = Sequential()
        model.add(GRU(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    if design == 'rnn_1':
        neurons_0 = 1
        neurons_1 = 1
        neurons_2 = 1

        batch_size = 20
        epochs = 400#78
        model = Sequential()
        model.add(SimpleRNN(neurons_0, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(SimpleRNN(neurons_1, return_sequences=True))
        model.add(SimpleRNN(neurons_2))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='huber', optimizer='adam')
    elif design == 'rnn_2':
        # design 2
        neurons_1 = 1
        batch_size = 20
        epochs = 400
        model = Sequential()
        model.add(SimpleRNN(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'rnn_3':
        # design 3
        neurons_1 = 1
        batch_size = 20
        epochs = 200
        model = Sequential()
        model.add(SimpleRNN(neurons_1, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'esn_easy':
        esn = PredictionESN(n_input=train_X.shape[1], n_output=train_y.shape[1], n_reservoir=455, leakingRate=0.75, regressionParameters=[0.5e-0], solver="lsqr", feedback=False)
    elif design == 'esn_2':
        # design 2 replicated with an ESN
        neurons_1 = 1
        batch_size = 20
        epochs = 400
        model = Sequential()
        #model.add(ESN(units = 55, leaky=0.75, spectral_radius = 1.0, activation="tanh"))
        model.add(ESN(units = 12))
        
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'nvar':
        # Note: this is Work on progress - I got it working only without exog variable, which is useless
        model = AutoReg(
                            endog=train_X[:, 0],
                            lags=29,
                            exog=train_X[:, 1:]
                        )

    # fit network
    if design == "esn_easy":
        history = esn.fit(train_X, train_y, transientTime="Auto", verbose=1)
        ypred_scaled = esn.predict(test_X)
    elif design == "nvar":
        model_fit = model.fit()
        ypred_scaled = np.transpose([model_fit.predict(start=len(train_X[:, 0]), end=len(train_X[:, 0])+len(test_X[:, 0])-1, dynamic=False)])
        history = None
    else: 
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        ypred_scaled = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast ('fill_in' just  makes the input array of the right size so inverse_transform() will work)
    fill_in = np.zeros(test_X[:, n_output_steps:scaled.shape[1]].shape)
    #fill_in = np.zeros(test_X[:, n_output_steps-1:scaled.shape[1]].shape) # this fill here works if the observation is removed from the input
    y_pred = concatenate((ypred_scaled, fill_in), axis=1)
    y_pred = scaler.inverse_transform(y_pred)

    # Take the first value as solution
    y_pred = y_pred[:,0]
    # invert scaling for actual - observed data

    y_obs = concatenate((test_y, fill_in), axis=1)
    y_obs = scaler.inverse_transform(y_obs)

    # Take the first value as solution
    y_obs = y_obs[:,0]
    # calculate RMSE
    rmse = []
    for i_val in ids_input[1:]:
        rmse_ = sqrt(mean_squared_error(ds_values[-test_set_size:, i_val], ds_values[-test_set_size:, 0]))
        rmse_train = sqrt(mean_squared_error(ds_values[:-test_set_size, i_val], ds_values[:-test_set_size, 0]))
        rmse.append(rmse_)
        print('Test  RMSE CAM %d: %.3f' % (i_val, rmse_))
        print('Train RMSE CAM %d: %.3f' % (i_val, rmse_train))


    rmse_model = sqrt(mean_squared_error(y_obs, y_pred))
    #rmse_model_persistence = sqrt(mean_squared_error(y_obs[1:-1], np.roll(y_pred, -1)[1:-1]))

    rmse_persistence = sqrt(mean_squared_error(y_obs[1:-1], np.roll(y_obs, -1)[1:-1]))


    print('Test RMSE model: %.3f' % rmse_model)
    #print('Test RMSE model Persistence: %.3f' % rmse_model_persistence)
    print('Test RMSE Persistence: %.3f' % rmse_persistence)

    return history, rmse, rmse_persistence, rmse_model, y_pred, y_obs   

