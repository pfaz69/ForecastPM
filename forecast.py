import numpy as np
from numpy import concatenate
from numpy.random import seed
from tensorflow import random as rnd
from math import sqrt
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error, d2_absolute_error_score, d2_pinball_score, d2_tweedie_score
	
#from easyesn import PredictionESN
#from easyesn import backend as B
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN, RNN, Flatten
from tensorflow_addons.layers import ESN
from tensorflow_addons.rnn import ESNCell 
from keras.utils.vis_utils import plot_model

import keras.losses

from scipy import linalg 

from statsmodels.tsa.statespace.sarimax import SARIMAX

# convert series to supervised learning
def _series_to_supervised(data, n_in=1, n_out=1):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations and CAMS as a list or NumPy array.
		n_in: Number of lag observations and CAMS as input.
		n_out: Number of observations as output.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    num_graphs = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    df.dropna(inplace=True)

    cols, names = list(), list()


    '''
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        list_col = [df[j] for j in range(1, data.shape[1]) ]
        cols.append(DataFrame([df[0].shift(i), *list_col]).T)
        # Put names for debugging
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(num_graphs)]
    '''


    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
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

	# Drop rows with NaN values caused by the above shifting
    agg.dropna(inplace=True)


    # Drop CAMS from labels
    for i in range(n_out):
        col_drop_start = n_in*num_graphs + 1 + i
        col_drop_end = (n_in + 1)*num_graphs + i
        agg.drop(agg.columns[list(range(col_drop_start, col_drop_end))], axis=1, inplace=True)


    # Hack: Drop the first column to remove observation from input
    # NOTE: search this file with keyword "observation" to find the other corresponding hacks
    #for i in range(n_in, 0, -1):
    #    agg.drop(columns=agg.columns[(i - 1)*num_graphs], axis=1, inplace=True)    
    # --------------------------------------------------


    return agg


def prep_data(dtf_obs, dtf_CAMS, time_forecast):
    # Inputs:
    #           dtf_obs is a series of floats
    #           dtf_CAMS is a list of lists of series of floats
    #           time_forecast is the chosen time of forecast

    #time = 1 # 0-> t=0 1->t=24h, 2->t=48h
    #list_cams = [dtf_CAMS[i][time] for i in range(9)if i < 2] 
    #list_cams = [dtf_CAMS[i][time] for i in range(len(dtf_CAMS))] # <- all cams
    #list_cams = [dtf_CAMS[i][time] for i in range(9) if i == 3 or i == 4 or i == 5 or i == 7] #if i == 0] #
    #list_cams = [dtf_CAMS[i][time] for i in range(9) if i == 4] #if i == 0] #
    #list_cams = [] # <- no cams

    # Test
    #list_cams0 = [dtf_CAMS[i][time - 1] for i in range(len(dtf_CAMS))]
    #list_cams2 = [dtf_CAMS[i][time + 1] for i in range(len(dtf_CAMS))]

    # at time t = 0: available three possible inputs from CAMS
    
    '''
    list_cams00 = [dtf_CAMS[i][0] for i in range(len(dtf_CAMS))]
    list_cams24 = [dtf_CAMS[i][1].shift(1, fill_value=dtf_CAMS[i][1][0]) for i in range(len(dtf_CAMS))]
    list_cams48 = [dtf_CAMS[i][2].shift(2, fill_value=dtf_CAMS[i][2][0]) for i in range(len(dtf_CAMS))]
    '''
    '''
    list_cams00 = [dtf_CAMS[i][0] for i in range(len(dtf_CAMS))]
    list_cams24 = [dtf_CAMS[i][1] for i in range(len(dtf_CAMS))]
    list_cams48 = [dtf_CAMS[i][2] for i in range(len(dtf_CAMS))]
    '''
    #list_cams = [dtf_CAMS[i][time_forecast] for i in range(len(dtf_CAMS))]
    list_cams = [dtf_CAMS[i][1].shift(-time_forecast, fill_value=dtf_CAMS[i][1][0]) for i in range(len(dtf_CAMS))]
    #list_cams = [] # <- no cams
    # time 0
    #list_cams = [dtf_CAMS[i][0] for i in range(len(dtf_CAMS))]
    # time 1
    #list_cams = [dtf_CAMS[i][1].shift(-1, fill_value=dtf_CAMS[i][1][0]) for i in range(len(dtf_CAMS))]
    # time 2
    #list_cams = [dtf_CAMS[i][2].shift(-1, fill_value=dtf_CAMS[i][2][0]) for i in range(len(dtf_CAMS))]
    
    # at time t = 24: available two possible inputs from CAMS
    #list_cams24 = [dtf_CAMS[i][1].shift(-1, fill_value=dtf_CAMS[i][1][0]) for i in range(len(dtf_CAMS))]
    #list_cams48 = [dtf_CAMS[i][2].shift(-1, fill_value=dtf_CAMS[i][2][0]) for i in range(len(dtf_CAMS))]
    



    dataset = concat(	
                        [
                            dtf_obs, 
                            *list_cams,
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
                            n_output_steps,
                            time_forecast
                        ):

    # Set reproducibiliy on
    rnd.set_seed(0)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(ds_values)
    #scaled = ds_values[:,:2] # <- Hack to better check the head() - remove this to make the whole procedure work


    reframed = _series_to_supervised(scaled, n_input_steps, n_output_steps)
    #print(reframed.head()) # Check the head (it goes with the above hack)
    refr_values = reframed.values

    # Split Dataset in Train and Test
    train, test = refr_values[0:-test_set_size], refr_values[-test_set_size:]
    
    # Split into input (all columns except the last ones) and outputs (latest columns)
    train_X_, train_y_ = train[:, :-n_output_steps], train[:, -n_output_steps:]
    test_X_, test_y_ = test[:, :-n_output_steps], test[:, -n_output_steps:] #test[:, 0]#

    # Arrange data by complying each ANN need
    if design.startswith("lstm") or design.startswith("gru"):
        train_X = train_X_.reshape((train_X_.shape[0], 1, train_X_.shape[1]))
        test_X = test_X_.reshape((test_X_.shape[0], 1, test_X_.shape[1]))
        train_y = train_y_
        test_y  = test_y_
    elif design.startswith("rnn") or design.startswith("esn"):
        if (n_output_steps != 1):
            print('\033[91m' + "Error:" + '\033[0m' + "for SimpleRNN and ESN the number of output steps can only be 1")
            exit(0)
        if (n_input_steps == 1):
            print('\033[93m' + "Warning:" + '\033[0m' + " for SimpleRNN and ESN setting the input to one will prevent the recurrent machinery from working")
        train_X = np.reshape(train_X_, (train_X_.shape[0], n_input_steps, scaled.shape[1]))  
        test_X  = np.reshape(test_X_, (test_X_.shape[0], n_input_steps, scaled.shape[1]))  
        #train_X = np.reshape(train_X_, (train_X_.shape[0], n_input_steps, scaled.shape[1] - 1)) # if observation is removed from input this replaces the coresponding above line
        #test_X  = np.reshape(test_X_, (test_X_.shape[0], n_input_steps, scaled.shape[1] - 1)) # if observation is removed from input this replaces the coresponding above line  
        # Debug: check if reshaping has been done properly with the 'timestep' index
        assert(train_X[0][n_input_steps - 1][0] == train_X_[0][(n_input_steps - 1)*scaled.shape[1]]) # comment this out if observation is removed from input
        train_y = train_y_
        test_y  = test_y_
    else:
        train_X = train_X_
        test_X  = test_X_
        train_y = train_y_
        test_y  = test_y_      


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
        model.compile(loss='mae', optimizer='adam')
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
        model.compile(loss='mae', optimizer='adam')
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
        model.compile(loss='mae', optimizer='adam')
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
    #elif design == 'esn_easy':
    #    esn = PredictionESN(n_input=train_X.shape[1], n_output=train_y.shape[1], n_reservoir=455, leakingRate=0.75, regressionParameters=[0.5e-0], solver="lsqr", feedback=False)
    elif design == 'esn_1':
        pass # TBI
    elif design == 'esn_3':
        pass # TBI
    elif design == 'esn_2':
        rho = 0.85
        res_size = 40
        batch_size = 20
        epochs = 4000

        '''
        model = Sequential()
        #model.add(Dense(10))
        model.add(RNN(ESNCell(res_size, spectral_radius = rho, leaky=0.75, connectivity = 0.3), return_sequences=True))
        model.add(Flatten())
        model.add(Dense(train_y.shape[1]))#, activation='sigmoid'))
        model.compile(loss='mae', optimizer='adam')
        '''
        
        
        # Working Model
        rho = 0.85
        res_size = 40
        batch_size = 20
        epochs = 400
        model = Sequential()
        model.add(RNN(ESNCell(res_size, spectral_radius = rho, leaky=0.75, connectivity = 0.3), return_sequences=False))
        model.add(Dense(train_y.shape[1]))#, activation='sigmoid'))
        model.compile(loss='mae', optimizer='adam')
        
        # Try here:
        '''
        input_shape = Input(shape=train_X_.shape)
        rnn = RNN(ESNCell(res_size, spectral_radius = rho, leaky=0.75, connectivity = 0.8), return_sequences=False)(input_shape)
        merged = keras.layers.concatenate([input_shape, rnn], axis=1)
        model = Dense(train_y.shape[1])(merged)
        model.compile(loss='mae', optimizer='adam')
        '''
    elif design == 'wmp_2' or design == 'wmp4_2':
        batch_size = 20
        epochs = 200
        model = Sequential()
        model.add(Dense(5))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss='mae', optimizer='adam')
    elif design == 'sarimax':
        batch_size = 20
        epochs = 200
        # Set the order and seasonal_order parameters for the SARIMAX model

        # Hyperparameters with best AIC
        #order = (1, 0, 2)
        #seasonal_order = (1, 1, 2, 12)

        # Hyperparameters with best MSE
        order = (1, 1, 2)
        seasonal_order = (2, 0, 2, 12)

        # Hyperparameters by hand
        #order = (1, 1, 1)
        #seasonal_order = (1, 1, 1, 12)
        
        # Initialize the SARIMAX model
        model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order)
        
        # Fit the model to the training data
        model_fit = model.fit()
        
        # Generate predictions for the test data
        ypred_scaled = np.expand_dims(model_fit.predict(start=len(train_y), end=len(train_y)+len(test_y)-1), axis=1)
        history = None
    # fit network
    #if design == "esn_easy":
    #    history = esn.fit(train_X, train_y, transientTime="Auto", verbose=1)
    #    ypred_scaled = esn.predict(test_X)
    model.run_eagerly = False # Paolo if False will run optimized for performance
    
    if design != 'sarimax':
    
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        # Debug
        '''
        print("printing weights after fit ----------------------------------")
        print(model.summary())
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        print("num layers = " + str(len(model.layers)))
        for layer in model.layers: print(layer.get_config(), layer.get_weights())
        '''
        ypred_scaled = model.predict(test_X)
   


    # invert scaling for forecast ('fill_in' just  makes the input array of the right size so inverse_transform() will work)
    #fill_in = np.zeros(test_X_[:, n_output_steps:scaled.shape[1]].shape)
    fill_in = np.zeros([ypred_scaled.shape[0], scaled.shape[1] - n_output_steps])
    y_pred = concatenate((ypred_scaled, fill_in), axis=1)
    y_pred = scaler.inverse_transform(y_pred)

    # Take the first value as solution
    y_pred = y_pred[:,0]
    # invert scaling for actual - observed data

    y_obs = concatenate((test_y, fill_in), axis=1)
    y_obs = scaler.inverse_transform(y_obs)

    # Take the first value as solution
    y_obs = y_obs[:,0]
    # calculate SCORE
    score_cams = []
    #metric_val = r2_score
    metric_val = mean_squared_error #mean_absolute_error# 
    ds_values = np.concatenate((np.array([ds_values[:, 0]]).T , np.roll(ds_values[:, 1:], time_forecast, axis = 0)), axis = 1)
    for i_val in ids_input[1:]:
        score_ = metric_val(ds_values[-test_set_size:, 0], ds_values[-test_set_size:, i_val])
        score_train = metric_val(ds_values[:-test_set_size, 0], ds_values[:-test_set_size, i_val])
        score_cams.append(score_)
        print('Test  SCORE CAM %d: %.3f' % (i_val, score_))
        print('Train SCORE CAM %d: %.3f' % (i_val, score_train))


    score_model = metric_val(y_obs, y_pred)

    vec_score_size = 12
    step_score = len(y_obs)//vec_score_size
    score_model_vec = np.array([metric_val(y_obs[i*step_score:(i+1)*step_score], y_pred[i*step_score:(i+1)*step_score]) for i in range(vec_score_size)])

    score_persistence = metric_val(y_obs[0:-1], np.roll(y_obs, -1)[0:-1])


    print('Test SCORE model: %.3f' % score_model)
    print('Test SCORE vec:', score_model_vec)
    print('Test SCORE Persistence: %.3f' % score_persistence)

    #return history, [np.sqrt(score_cam) for score_cam in score_cams], sqrt(score_persistence), sqrt(score_model), y_pred, y_obs, score_model_vec#tmp pao     
    return history, score_cams, score_persistence, score_model, y_pred, y_obs, score_model_vec

