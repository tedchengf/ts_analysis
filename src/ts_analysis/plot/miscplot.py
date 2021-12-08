# miscplot.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_density(var_names, var_arrays, normalize = True, hist = False, show = False, save_name = None):
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert type(var_arrays) is np.ndarray and len(var_arrays.shape) == 2, "var_arrays must be an instance of numpy.ndarray with exactly 2 dimensions"
	for var_name, var_array in zip(var_names, var_arrays):
		if normalize == True:
			arr_sum = np.sum(var_array)
			var_array = np.divide(var_array, arr_sum)
		# sns.distplot(var_array, label = var_name, hist = hist, norm_hist=True)
		sns.displot(var_array, label = var_name)
	plt.legend()
	if save_name is not None:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def visualize_distribution(data, save_name = None):
	fig, ax = plt.subplots()
	for curr_inst in data:
		# sns.histplot(curr_inst, ax=ax, alpha = 0.1, binwidth = 0.02)
		sns.kdeplot(curr_inst, ax=ax)
	if save_name is not None:
		fig.savefig(save_name, format = "png", dpi = 1000, transparent = True)