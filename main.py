import numpy as np
import argparse
from load_data import read_data, correct_eventual_gaps
from visualize import plot_graphs_llist, plot_graphs_stacked, plot_bars, plot_multi_graphs
from forecast import prep_data, run_forecasting_algs
from utils import compute_correlations, adfuller_test, perform_self_causality_test, check_freqs


parser = argparse.ArgumentParser(description='PMx Forecast.')

parser.add_argument(
                        'dataset', 
                        choices =   [
                                        "Pallas",
                                        "Reykjavik",
                                        "Tromso",
                                        "FI_5_15557",
                                        "FI_5_15609",
                                        "IS_5_52109",
                                        "IS_5_52149",
                                        "NO_5_28816",
                                        "FI_5_101983"
                                    ],
                        help='dataset file'
                    )

parser.add_argument(
                        'design', 
                        choices =   [
                                        "lstm_1",
                                        "lstm_2",
                                        "lstm_3",
                                        "gru_1",
                                        "gru_2",
                                        "gru_3",
                                        "rnn_1",
                                        "rnn_2",
                                        "rnn_3",
                                        "esn_2",
                                        "esn_easy",
                                        "nvar"
                                    ],
                        help='forecasting method'
                    )


parser.add_argument('--test_set_size', type=int, default=100, help='test set size')   
parser.add_argument('--num_input_steps', type=int, default=1, help='number of input steps')   
parser.add_argument('--num_output_steps', type=int, default=1, help='number of output steps')   


args = parser.parse_args()
print("Dataset: " + args.dataset)
print("Design: " + args.design)

# Output Location
output_stub = args.dataset + '_' + args.design

# Load Data
obs, cams = read_data(args.dataset)

# Visualize Raw Data
plot_graphs_llist('input_graphs', 'raw_obs', [[obs]])
plot_graphs_llist('input_graphs', 'raw_CAM', cams)

# Fix eventual gaps
obs_c = correct_eventual_gaps(obs, broken_value_flag = -1000)
cams_c = correct_eventual_gaps(cams, broken_value_flag = -999)

# Visualize Corrected Data
plot_graphs_llist('input_graphs', 'corrected_obs', [[obs_c]])
plot_graphs_llist('input_graphs', 'corrected_CAM', cams_c)

# Visualize Data FT
obs_c_freqs = check_freqs(obs_c)
plot_multi_graphs(  
                    'output_graphs/output', 'obs_spectrum', output_stub,
                    [obs_c_freqs], 
                    ['power spectrum']
                )


# Check observation series for stationarity
adfuller_test(obs_c)

# Restrict range to avoid unfixeble corrupted observations
start_val = 500
n_obs_points = len(obs_c)
obs_c = obs_c[start_val:n_obs_points]
for i in range(9):
	for j in range(3):
		cams_c[i][j] = cams_c[i][j][start_val:n_obs_points]


# Prep
input_data = prep_data(obs_c, cams_c)

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
history, rmse_cams, rmse_pers, rmse_model, y_pred, y_obs = run_forecasting_algs(
                                                                    ids_input,
                                                                    args.design, 
                                                                    input_data, 
                                                                    test_set_size,
                                                                    args.num_input_steps,
                                                                    args.num_output_steps
                                                                )



# Plot RMSE
plot_bars('output_graphs/output', 'RMSE', output_stub, ids_input[1:], rmse_cams, rmse_pers, rmse_model)

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
                        labels
                     )

# Plot model vs observation
labels=['model', 'observation']
plot_multi_graphs(  
                    'output_graphs/output', 'model_vs_observed', output_stub,
                    [y_pred, obs_c.values[-test_set_size:]], 
                    labels
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
                    labels
                 )


