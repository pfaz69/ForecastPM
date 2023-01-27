import numpy as np
import pandas as pd
from pandas import read_csv

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



def read_data(dataset_name):

    # read_cvs() for _ENS_4pnts
    
    dtf = read_csv(
                    'Data/' + dataset_name + '_pm10_ENS_4pnts.txt',
                    sep='\s+',
                    header=None, 
                    skiprows=lambda x: x % 4 != 0
                )
    

    # read_cvs() for _ENS
    '''
    dtf = read_csv(
                    'Data/' + dataset_name + '_pm10_ENS.txt',
                    sep='\s+',
                    header=None
                )
    '''

    #dtf['datetime'] = pd.to_datetime(dtf[[3, 2, 1, 4]].astype(int).astype(str).apply(' '.join, 1), format='%d %m %Y %H')  # read_cvs() for _ENS
    dtf['datetime'] = pd.to_datetime(dtf[[4, 3, 2, 5]].astype(int).astype(str).apply(' '.join, 1), format='%d %m %Y %H') # read_cvs() for _ENS_4pnts

    dtf.head()
    #dtf_obs = dtf[5][::9].reset_index(drop=True).astype(float) # read_cvs() for _ENS
    dtf_obs = dtf[6][::9].reset_index(drop=True).astype(float) # read_cvs() for _ENS_4pnts
    dtf_CAMS = [[] for _ in range(9)]
    for i in range(9):
        for j in range(3):
            #dtf_CAMS[i].append(dtf[6+j][i::9].reset_index(drop=True).astype(float)) # read_cvs() for _ENS
            dtf_CAMS[i].append(dtf[7+j][i::9].reset_index(drop=True).astype(float)) # read_cvs() for _ENS_4pnts
    return dtf_obs, dtf_CAMS


def correct_eventual_gaps(data_in, broken_value_flag = -1000):

    # This function needs a 2D list as an input: if it is not provided...
    # ...data needs to be properly encapsulated

    if type(data_in) is not list: 
        data_proc = [[data_in]]
    elif type(data_in[0]) is not list:
        data_proc = [data_in]
    else:
        data_proc = data_in

   
    for model_data in data_proc:
        for ic, cam_data in enumerate(model_data):
            for i, data in enumerate(cam_data):
                if (data == broken_value_flag):
                    cam_data[i] = np.nan
            model_data[ic] = cam_data.interpolate()

    # Eventually decapsulate
    if type(data_in) is not list:
        data_proc = data_proc[0][0]
    elif type(data_in[0]) is not list:
        data_proc = data_proc[0]

    return data_proc