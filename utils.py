import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.fft import fft

def compute_correlations(preds, obs):
    # Compute auto-correlation of observation and model and their cross-correlation

    # Compute Correlations 
    obs_corr = np.correlate(obs, obs, "same")
    x_corr = np.correlate(obs, preds, "same")
    mdl_corr = np.correlate(preds, preds, "same")

    # Normalize
    obs_corr = (obs_corr - obs_corr.min()) / (obs_corr.max() - obs_corr.min())
    x_corr = (x_corr - x_corr.min()) / (x_corr.max() - x_corr.min())
    mdl_corr = (mdl_corr - mdl_corr.min()) / (mdl_corr.max() - mdl_corr.min())
    return obs_corr, x_corr, mdl_corr


def adfuller_test(series, thsh=0.05, name=''):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r_adf = adfuller(series, autolag='AIC')
    print('ADF Statistic: %f' % r_adf[0])
    print('p-value: %f' % r_adf[1])
    print('Critical Values:')
    for key, value in r_adf[4].items():
        print('\t%s: %.3f' % (key, value))
    if r_adf[1] <= thsh:
        print("Observation series is stationary")

'''
def grangers_causation_matrix(data, variables, test='ssr_chi2test'):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    
    maxlag=12
    for c in df.columns:
        for r in df.index:
            #test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            test_result = grangercausalitytests(numpy.transpose(numpy.array([numpy.roll(data[:, 0] + numpy.random.randn(len(data[:, 0])), 1), data[:, 0]])), maxlag=4)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
'''

# Paolo: the following has ha problem with input data. But is it useful?
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data_values, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    data = pd.DataFrame(data_values)
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def perform_self_causality_test(input_data):
    grangercausalitytests(np.transpose(np.array([np.roll(input_data[:, 0] + 0.1*np.random.randn(len(input_data[:, 0])), 1), input_data[:, 0]])), maxlag=4)

def check_freqs(v_in):
    v_out = fft(v_in.values)
    v_out[0] = v_out[1]
    return np.abs(v_out)