import numpy as np
import pandas as pd
from pandas import read_csv

'''
# convert series to supervised learning
def _series_to_supervised(data, n_in=1, n_out=1):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		list_col = [df[i] for i in range(1, data.shape[1]) ]
		cols.append(DataFrame([df[0].shift(i), *list_col]).T)
		#cols.append(df.shift(i)) # whole shift: commented out
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names

	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg
'''


def read_data(folder, dataset_name):

    use_averaged_data = True

    if use_averaged_data:
        # read_cvs() for _ENS
        dtf = read_csv(
                    folder + '/' + dataset_name + '_pm10_ENS.csv',
                    sep='\t', 
                    #sep='\s+' #tmp pao
                )
    else:
        # read_cvs() for _ENS_4pnts
        dtf = read_csv(
                        folder + '/' + dataset_name + '_pm10_ENS_4pnts.txt',
                        sep='\s+',
                        skiprows=lambda x: x % 4 != 0
                    )
    


    #dtf['datetime'] = pd.to_datetime(dtf[['D', 'M', 'Y', 'H']].astype(int).astype(str).apply(' '.join, 1), format='%d %m %Y %H')  # read_cvs() for _ENS
    '''
    obs_arr = dtf['Obs'][::9].reset_index(drop=True).astype(float).to_numpy()
    mask = np.zeros(len(obs_arr))
    obs_struct_arr = list(map(list, zip(obs_arr, mask)))
    dtf_obs = pd.Series(obs_struct_arr) # read_cvs() for _ENS
    '''


    dtf_obs = dtf['Obs'][::9].reset_index(drop=True).astype(float) # read_cvs() for _ENS

    dtf_CAMS = [[] for _ in range(9)]
    for i in range(9):
        for pred in ['Pred0', 'Pred24', 'Pred48']:
            dtf_CAMS[i].append(dtf[pred][i::9].reset_index(drop=True).astype(float)) # read_cvs() for _ENS

    return dtf_obs, dtf_CAMS


def correct_eventual_gaps(data_in, thsh_value_to_fix = None):

    # This function needs a 2D list as an input: if it is not provided...
    # ...data needs to be properly encapsulated

    if type(data_in) is not list: 
        data_proc = [[data_in]]
    elif type(data_in[0]) is not list:
        data_proc = [data_in]
    else:
        data_proc = data_in

   
    for model_data in data_proc:
        for ic, list_data in enumerate(model_data):
            if thsh_value_to_fix != None:
                for i, data in enumerate(list_data):
                    if (data >= thsh_value_to_fix):
                        list_data[i] = np.nan
            model_data[ic] = list_data.interpolate(limit_direction='both', method='linear') # Fix eventual NaNs

    # Eventually decapsulate
    if type(data_in) is not list:
        data_proc = data_proc[0][0]
    elif type(data_in[0]) is not list:
        data_proc = data_proc[0]

    return data_proc