# tsplot.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Visualize the significant intervals
# Parameters:
# 	highlight_intervals: a nested 3D array. The first dimension should be
#		variables. The second dimension should be the number of significant
#		intervals the variable has. The third dimension records a tuple of the
#		starting and ending position of each interval. Suppose we have one 
#		variable with two significant interval. The highlight_intervals will be
#		 [[[start, end], [start, end]]].
#	slot_num: an int denoting the number of time points there are. Suppose the 
#		original data has 90 time points, then slot_num = 90
#	names: a list containing the names of the variables
#	start_end: a tuple of ints. the start and end labels on the x-axis
#	interval: and int denoting the intervals between the labels on the x-axis
#	sig_color: either a list of colors or a string denoting a single color. If
#		sig_color is a list, then its' length must match the length of names
#	aspect: a float denoting thethe ratio between the x and y axis. Since this 
#		function uses imshow,this parameter is directly passed into the imshow
#		function. Default = 1.5
#	show: a boolean that denote whether the plot will be shown. Default = False
#	save_name: a save name to save the plot. If save_name = None, the plot will
#		not be saved. Default = None
def visualize_sig(intervals_dict, slot_num, start_end = None, interval = 100, sig_color = None, aspect = 1.5, exclude_empty = True):
	from matplotlib import colors
	# Set discrete cmap
	if sig_color is None:
		sig_color = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(intervals_dict)]
	if type(sig_color) is list or type(sig_color) is np.ndarray:
		assert (len(sig_color) == len(intervals_dict))
		cmap = colors.ListedColormap(sig_color)
		bounds = np.append(np.arange(0.5, len(sig_color)-0.5, step = 1), len(sig_color))
		norm = colors.BoundaryNorm(bounds, cmap.N)
	else:
		cmap = colors.ListedColormap(["grey", sig_color])
		bounds = [0,0.5,1]
		norm = colors.BoundaryNorm(bounds, cmap.N)
	# assemble significance array
	sig_arrays = []
	sig_vars = []
	for var_ind, var in enumerate(intervals_dict):
		var_hl = intervals_dict[var]
		curr_sig = np.zeros(slot_num)
		if len(var_hl) > 0:
			for interval in var_hl:
				for i in range(interval[0], interval[1]):
					curr_sig[i] = 1*(var_ind + 1)
		else:
			if exclude_empty == True: continue
		sig_arrays.append(curr_sig)
		sig_vars.append(var)
	if len(sig_arrays) == 0: return False
	sig_arrays = np.array(sig_arrays)
	sig_arrays[np.where(sig_arrays < 1)] = np.nan
	plt.rc('font', size = 3) 
	fig = plt.figure(figsize=(6.4,2))
	ax = fig.gca()
	ax.imshow(sig_arrays, cmap=cmap, norm=norm, interpolation="none", aspect=aspect)
	ax.set_yticks(np.arange(len(sig_vars), dtype = int))
	ax.set_yticklabels(sig_vars)
	ax.axes.get_xaxis().set_visible(False)
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	plt.close()
	return fig

# Parameters:
# 	Data: a 2D numpy array with the dimension of (variables, time points)
#	names: a list containing the names of the variables
#	title: a string denotaing the title of the plot. Default = None
#	start_end: a tuple of ints. the start and end labels on the x-axis
#	interval: and int denoting the intervals between the labels on the x-axis
#	axis: a list with four elements. It defines the x and y axes. This
#		parameter will be direclty passed into the matplotlib axis() function.
#		Default = [None, None, None, None]
#	colors: a list of colors that denotes the colors of each variable. Default
#		= None
#	highlight_intervals: a 2D numpy array with the dimension of (interval, 2).
#		The last dimension contains a a tuple recording the starting and ending 
#		position of each interval. Default = None
#	highlight_colors: a list of colors denoting the color of each highlight
#		interval. If defined, its length must align with the first dimension of
#		the highlight_intervals. Default = None (and all intervals will be red)
#	figsize: an 2-element tuple denoting the the size of the figure. It will be
#		directly passed to the matplotlib figure() function.
#	show: a boolean that denote whether the plot will be shown. Default = False
#	save_name: a save name to save the plot. If save_name = None, the plot will
#		not be saved. Default = None
def plot_1D(Data, names, title = None, start_end = None, interval = 100, axis = [None, None, None, None], colors = None, linestyles = None, highlight_intervals = None, highlight_colors = None, figsize = (6.4, 4.8), show = False, save_name = None):
	assert type(Data) is np.ndarray and len(Data.shape) == 2, "Data must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type(names) is np.ndarray, "names must be an instance of numpy.ndarray"
	assert len(axis) == 4, "axis must have exactly 4 elements"
	matplotlib.rcParams.update({'font.size': 6})
	fig = plt.figure(figsize = figsize)
	ax = fig.gca()
	ax.margins(x=0)
	for var_index in range(len(names)):
		var_name = names[var_index]
		corr_results = Data[var_index]
		if colors is not None:
			curr_color = colors[var_index]
		else: curr_color = None
		if linestyles is not None:
			curr_style = linestyles[var_index]
		else: curr_style = None
		ax.plot(corr_results, label = var_name, color = curr_color, linestyle = curr_style)
		# else:
		# 	ax.plot(corr_results, label = var_name)
	if title is not None:
		ax.set_title(title)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		tick_num = len(np.arange(0, x_range, step = step, dtype = int))
		ax.set_xticks(np.arange(0, x_range, step = step, dtype = int))
		ax.set_xticklabels(label[:tick_num])
	if highlight_intervals is not None:
		assert type(highlight_intervals) is np.ndarray and len(highlight_intervals.shape) == 2, "highlight_intervals must be an instance of numpy.ndarray with exactly 2 dimensions"
		if highlight_colors is None:
			highlight_colors = np.empty(highlight_intervals.shape[0])
			for ind in range(len(highlight_colors)): highlight_colors[ind] = "red"
		for ind, x_interval in enumerate(highlight_intervals):
			plt.axvspan(x_interval[0], x_interval[1], color = highlight_colors[ind], alpha = 0.4)
	ax.axis(axis)
	ax.legend()
	if save_name is not None:
		fig.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		fig.show()
	plt.close()
	return fig

def plot_subjects(Data, title, frame, start_end = None, interval = 100, axis = [None, None, None, None], show = False, save = False, save_name = "./all_sub_corr.png"):
	matplotlib.rcParams.update({'font.size': 3})
	fig, ax = plt.subplots(nrows=frame[0], ncols=frame[1])
	for sub_index in range(Data.shape[0]):
		row_ind = sub_index // frame[1]
		col_ind = sub_index % frame[1]
		ax[row_ind, col_ind].plot(Data[sub_index, :])
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round((end-start) / x_range))
		ax[row_ind, col_ind].xaxis.set_ticklabels(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	if show == True:
		plt.show()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000)

def plot_corr_2D(Data, title, y_label = None, start_end = None, interval = 100, clim = None, show = False, save = False, save_name = "2D_corr.png"):
	assert type(Data) is np.ndarray and len(Data.shape) == 2, "Data must be an instance of numpy.ndarray with exactly 2 dimensions"
	current_cmap = plt.cm.get_cmap()
	current_cmap.set_bad(color='grey')
	plt.figure(figsize = (10,5))
	plt.imshow(Data, interpolation="none", aspect='auto')
	plt.title(title)
	plt.colorbar()
	if clim is not None:
		plt.clim(clim)
	if y_label is not None:
		assert type(y_label) is np.ndarray, "y_label must be an instance of numpy.ndarray"
		assert y_label.shape[0] == Data.shape[0], "The first dimension of Data and y_label does not match"
		plt.yticks(np.arange(0, len(y_label), step = 1, dtype = int), y_label)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

# means: a numpy 2D array with the dimension of (variable, time points)
# stds: a numpy 2D arrayw with the dimension of (variable, time points). For
#		each time point, the y region between mean + std abd mean - std will
#		be colored.
def plot_variability(var_names, means, stds, start_end = None, interval = 100, axis = [None, None, None, None], colors = None, show = False, save_name = "./var.png"):
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert type (means) is np.ndarray and len(means.shape) == 2, "means must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type (stds) is np.ndarray and len(stds.shape) == 2, "stds must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert len(axis) == 4, "axis must have exactly 4 elements"	
	for var_index, var_name in enumerate(var_names):
		var_mean = means[var_index]
		var_std = stds[var_index]
		if colors is not None:
			curr_color = colors[var_index]
			plt.plot(var_mean, label = var_name, color=curr_color)
			plt.fill_between(np.arange(0, len(var_mean), step = 1, dtype = int), var_mean - var_std, var_mean + var_std, alpha = 0.3, color=curr_color)
		else:
			plt.plot(var_mean, label = var_name)
			plt.fill_between(np.arange(0, len(var_mean), step = 1, dtype = int), var_mean - var_std, var_mean + var_std, alpha = 0.3)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = means.shape[1]
		step = int(round((end-start) / x_range))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	plt.axis(axis)
	plt.legend()
	if save_name is not None:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

