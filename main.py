import numpy as np
import argparse
from load_data import read_data, correct_eventual_gaps
from visualize import plot_graphs_llist, plot_graphs_stacked, plot_bars, plot_multi_graphs
from forecast import prep_data, run_forecasting_algs
from utils import compute_correlations, adfuller_test, perform_self_causality_test, check_freqs
from locations_algorithms import locations, algorithms

parser = argparse.ArgumentParser(description='PM10 Forecast.')

    

parser.add_argument(
                        'dataset',
                        choices = locations,
                        help='dataset file'
                    )

parser.add_argument(
                        'design', 
                        choices = algorithms,
                        help='forecasting method'
                    )


parser.add_argument('--test_set_size', type=int, default=100, help='test set size')   
parser.add_argument('--num_input_steps', type=int, default=1, help='number of input steps')   
parser.add_argument('--num_output_steps', type=int, default=1, help='number of output steps')   
parser.add_argument('--time_0', type=int, default=0, help='timeseries starting step')   
parser.add_argument('--OR', type=int, default=0, help='timeseries starting step')   



args = parser.parse_args()
print("Dataset: " + args.dataset)
print("Design: " + args.design)

# Dict for outliers removal (OR)
thsh_outliers =	{
                    "FI_5_101983": 9.669090231591326,
                    "FI_5_15557": 15.682631246852862,
                    "FI_5_15609": 20.34192298826304,
                    "NO_5_62993": 20.582944723862678,
                    "IS_5_52109": 22.907589131842936,
                    "IS_5_52149": 26.45740284287375
                }



# Output Location
output_stub = args.dataset + '_' + args.design

# Load Data
#obs, cams = read_data('Data/Before-Latest', args.dataset) # tmp pao
obs, cams = read_data('Data/Latest', args.dataset)
# Debug: trim
#obs = obs[:637]
#cams = [[cam_XY[:637] for cam_XY in cam_X] for cam_X in cams]


# Visualize Raw Data
plot_graphs_llist('input_graphs', 'raw_obs', [[obs]])
plot_graphs_llist('input_graphs', 'raw_CAM', cams)

# Fix eventual gaps
#This works with headerless input files
#obs_c = correct_eventual_gaps(obs, thsh_value_to_fix = -1000)
#cams_c = correct_eventual_gaps(cams, thsh_value_to_fix = -999)
if args.OR == 1:
    thsh = thsh_outliers[args.dataset]
else:
    thsh = 10000
obs_c = correct_eventual_gaps(obs, thsh_value_to_fix = thsh) 
cams_c = correct_eventual_gaps(cams)


# Visualize Corrected Data
plot_graphs_llist('input_graphs', 'corrected_obs', [[obs_c]])
plot_graphs_llist('input_graphs', 'corrected_CAM', cams_c)


# Check autocorrelation (w.r.t. white noise autocorrelaton)
'''
from statsmodels.graphics.tsaplots import plot_acf
white_noise = np.random.normal(0, 1, size=obs_c.shape)
plt1 = plot_acf(obs_c, title="Autocorrelation Signal")
plt2 = plot_acf(white_noise, title="Autocorrelation white noise")
plt1.show()
plt2.show()

acorr_signal = np.correlate(obs_c, obs_c, 'full')[len(obs_c)-1:]
acorr_ref = np.correlate(white_noise, white_noise, 'full')[len(white_noise)-1:]
plot_multi_graphs(  
                    'output_graphs/output', 'autocorr_analysis', output_stub,
                    [acorr_signal/np.max(acorr_signal), acorr_ref/np.max(acorr_ref)], 
                    ['signal', 'WN'], 
                    do_log = False
                )

'''

# Shift data to avoid big gaps 
start_val = args.time_0

#start_val = 469 # 500 # 
#start_val = 0 # 500 # tmp pao: this goes with the hack to check shifting data

#end_val = 941# This value (combined with start_val = 469) makes very good FI_5_101983 graph
end_val = 10000
end_val = np.min([len(obs_c), end_val])
#args.test_set_size = 365
'''
end_val = np.min([len(obs_c), end_val])
args.test_set_size = end_val - 941 + 100
#args.test_set_size = 425

'''

#obs = obs.shift(-start_val)[:(obs.shape[0] - start_val)] # Debug: check obs[0] for nan's
obs_c = obs_c.shift(-start_val)[:(end_val - start_val)]
cams_c = [[cam.shift(-start_val)[:(end_val - start_val)] for cam in cam_list] for cam_list in cams_c]


# Visualize shifted and cropped Data
plot_graphs_llist('input_graphs', 'shift_corrected_obs', [[obs_c]])
plot_graphs_llist('input_graphs', 'shift_corrected_CAM', cams_c)


# Check autocorrelation (w.r.t. white noise autocorrelation)
'''
from statsmodels.graphics.tsaplots import plot_acf
white_noise = np.random.normal(0, 1, size=obs_c.shape)
plt1 = plot_acf(obs_c, title="Autocorrelation Signal")
plt2 = plot_acf(white_noise, title="Autocorrelation white noise")
plt1.show()
plt2.show()


acorr_signal = np.correlate(obs_c - np.mean(obs_c), obs_c - np.mean(obs_c), 'same')[0:30]#[len(obs_c)-1:]
acorr_ref = np.correlate(white_noise, white_noise, 'same')[0:30]#[len(white_noise)-1:]
plot_multi_graphs(  
                    'output_graphs/output', 'autocorr_analysis', output_stub,
                    [acorr_ref/np.max(acorr_ref), acorr_signal/np.max(acorr_signal)], 
                     [ 'WN', 'signal'], 
                    do_log = False
                )
'''


# Visualize Data FT (obs)
obs_c_freqs = check_freqs(obs_c)
plot_multi_graphs(  
                    'output_graphs/output', 'obs_spectrum', output_stub,
                    [obs_c_freqs], 
                    ['power spectrum'], 
                    do_log = False
                )

# Debug: Visualize Data FT (cams)
'''
num_predictions = 3
for i, cam in enumerate(cams_c):
    for j in range(num_predictions):
        cam_freqs = check_freqs(cam[j])
        plot_multi_graphs(  
                            'output_graphs/output', 'cam_' + str(i+1) +'_spectrum_' + str(24*j) + 'h', output_stub,
                            [cam_freqs], 
                            ['power spectrum cam ' + str(i+1) + ', prediction at ' + str(24*j) + 'h'], 
                            do_log = False
                        )
'''

# Check observation series for stationarity
adfuller_test(obs_c)

'''
# Debug: Check CAMS's prediction accuracy
color_flag = ['0', '\033[92m', '\033[91m']
for i, cam in enumerate(cams_c):
    min = np.finfo('d').max
    max = 0
    for j in range(num_predictions):
        val = np.sum(np.abs(cam[j] - obs_c))
        if val > max : max = val
        if val < min : min = val
    for j in range(num_predictions):
        val = np.sum(np.abs(cam[j] - obs_c))
        if val == min:
            color = '\033[92m'
        elif val == max:
            color = '\033[91m'
        else:
            color = '\033[96m'
        print(color + 'Cam ' + str(i+1) +' forecast SAE wrt obs at t = ' + str(j*24) + 'h -> ' + str(val) + '\033[0m')
'''

# forecast time: whether forecast at 0 or 24h (time_forecast=0 or 1)
# are used
time_forecast = 1


# Prep data
input_data = prep_data(obs_c, cams_c, time_forecast)

# Check causality cams -> observation
#var_names = ["obs", "cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8", "cam9"]
#grangers_causation_matrix(input_data, var_names)
#grangers_causation_matrix(input_data, variables = var_names)

# Check self causality observation -> observation
perform_self_causality_test(input_data)

# Visualize all input
plot_graphs_stacked('output_graphs/output', 'cams_vs_obs', output_stub, input_data)

# Set Prediction Parameters
test_set_size = args.test_set_size

# Predict
ids_input = list(range(input_data.shape[1]))

#Note y_obs has been returned here only for debug (y_obs should be equal to obs_c[-test_set_size:])
history, score_cams, score_pers, score_model, y_pred, y_obs, score_model_vec = run_forecasting_algs(
                                                                    ids_input,
                                                                    args.design, 
                                                                    input_data, 
                                                                    test_set_size,
                                                                    args.num_input_steps,
                                                                    args.num_output_steps,
                                                                    time_forecast
                                                                )


# Plot Score
plot_bars('output_graphs/output', 'SCORE', output_stub, ids_input[1:], score_cams, score_pers, score_model)

# Plot Loss
if history == None:
    print("No history with " + args.design)
elif args.design == "esn_easy":
    print('ESN Final Loss %.3f' % history)
else:
    ids = ['loss','val_loss']
    labels=['train', 'test']
    plot_multi_graphs(  
                        'output_graphs/output', 'loss', output_stub,
                        [history.history[ids[0]], history.history[ids[1]]], 
                        labels,
                        do_log = False
                     )

# Plot model vs observation
labels=['model', 'observation']
plot_multi_graphs(  
                    'output_graphs/output', 'model_vs_observed', output_stub,
                    [y_pred, obs_c.values[-test_set_size:]], 
                    labels,
                    do_log = True,
                    alpha = 0.7
                 )

# Plot model vs cams vs observation
input_data_test_area = input_data[-test_set_size:]
data = np.append(input_data_test_area, np.transpose([y_pred]), 1)
id_model = 10
plot_graphs_stacked('output_graphs/output', 'model_cams_vs_obs', output_stub, data, id_model)

# Compute Correlations
obs_corr, x_corr, mdl_corr = compute_correlations(y_pred, obs_c.values[-test_set_size:])

# Plot Correlations
labels=['observation', 'cross', 'model']
plot_multi_graphs(  
                    'output_graphs/output', 'correlations', output_stub,
                    [obs_corr[45:56], x_corr[45:56], mdl_corr[45:56]], 
                    labels,
                    do_log = False
                 )

# Plot SCORE vec
labels=['SCOREs']
plot_multi_graphs(  
                    'output_graphs/output', 'SCOREs', output_stub,
                    [score_model_vec], 
                    labels,
                 )


