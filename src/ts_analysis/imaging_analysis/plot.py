# RSA_plot.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(matrix, title, var_names = None, clim = None, show = False, save = False):
	print (title + " dim: " + str(matrix.shape))
	matplotlib.rcParams.update({'font.size': 7})
	plt.figure()
	if var_names is not None:
		extent_max = 2*(matrix.shape[0])
	else:
		extent_max = matrix.shape[0]
	plt.imshow(matrix, interpolation='none', extent = [0, extent_max, 0, extent_max]);
	plt.title(title)
	if clim is not None:
		plt.clim(clim[0],clim[1])
	plt.colorbar()
	if var_names is not None:
		locs = np.arange(1, extent_max, step=2)
		plt.xticks(locs, var_names, rotation = 30)
		plt.yticks(np.flip(locs), var_names, rotation = 30)
	if save == True:
		plt.savefig(title + ".png", format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_corr(all_corr_results, var_names, start_end = None, interval = 100, axis = [None, None, None, None], show = False, save = False, save_name = "./corr_results.png"):
	assert type(all_corr_results) is np.ndarray and len(all_corr_results.shape) == 2, "all_corr_results must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert len(axis) == 4, "axis must have exactly 4 elements"
	matplotlib.rcParams.update({'font.size': 6})
	for var_index in range(len(var_names)):
		var_name = var_names[var_index]
		corr_results = all_corr_results[var_index]
		plt.plot(corr_results, label = var_name)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = all_corr_results.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	plt.axis(axis)
	plt.legend()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

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
def visualize_sig(highlight_intervals, slot_num, names, start_end = None, interval = 100, sig_color = "orange", aspect = 1.5, show = False, save_name = None):
	assert len(names) == len(highlight_intervals), "The first dimension of highlight_interval must match the length of variable names"
	assert type(sig_color) is str or type(sig_color) is list, "sig_color must be either a string or a list"
	from matplotlib import colors
	# Set discrete cmap
	if type(sig_color) is list:
		assert (len(sig_color) == len(names))
		cmap = colors.ListedColormap(sig_color)
		bounds = np.append(np.arange(1, len(sig_color), step = 0.5), len(sig_color))
		norm = colors.BoundaryNorm(bounds, cmap.N)
	else:
		cmap = colors.ListedColormap(["grey", sig_color])
		bounds = [0,0.5,1]
		norm = colors.BoundaryNorm(bounds, cmap.N)
	# assemble significance array
	sig_array = np.zeros((len(names), slot_num))
	for var_ind, var_name in enumerate(names):
		var_hi = highlight_intervals[var_ind]
		if len(var_hi) != 0:
			for interval in var_hi:
				for i in range(interval[0],interval[1]):
					sig_array[var_ind, i] = 1*(var_ind + 1)
	sig_array[np.where(sig_array < 1)] = np.nan
	plt.rc('font', size = 3) 
	fig = plt.figure(figsize=(6.4,2))
	ax = fig.gca()
	ax.imshow(sig_array, cmap=cmap, norm=norm, interpolation = "none", aspect = aspect)
	ax.set_yticks(np.arange(len(names), dtype = int))
	ax.set_yticklabels(names)
	ax.axes.get_xaxis().set_visible(False)
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	if show == True:
		fig.show()
	if save_name is not None:
		fig.savefig(save_name, format = "png", dpi = 1000, transparent = True)
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
def plot_1D(Data, names, title = None, start_end = None, interval = 100, axis = [None, None, None, None], colors = None, highlight_intervals = None, highlight_colors = None, figsize = (6.4, 4.8), show = False, save_name = None):
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
			ax.plot(corr_results, label = var_name, color = curr_color)
		else:
			ax.plot(corr_results, label = var_name)
	if title is not None:
		ax.set_title(title)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		ax.set_xticks(np.arange(0, x_range, step = step, dtype = int))
		ax.set_xticklabels(label)
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

def RDM_scatter_plot(fig, tri_x, tri_y, condition_mask, cond_pair_dict, s=0.5):
	colormap = generate_RDM_colormap(condition_mask, cond_pair_dict)
	ax = fig.gca()
	ax.scatter(tri_x, tri_y, c = colormap, s = s)
	return fig

def generate_RDM_colormap(condition_mask, cond_pair_dict):
	color_map = []
	for r in range(len(condition_mask)):
		for c in range(r+1, len(condition_mask)):
			curr_cond_pair = str(condition_mask[r]) + str(condition_mask[c])
			color_map.append(cond_pair_dict[curr_cond_pair])
	return np.array(color_map)

def plot_dendrogram_with_matrix(linkage_matrix_1, linkage_matrix_2, matrix, clim = None,  **kwargs):
	from scipy.cluster.hierarchy import dendrogram
	# Initialize figure
	fig = plt.figure(figsize=(10,10))
	# Plot first dendrogram
	# ax1 = fig.add_axes([0.15,0.1,0.15,0.6])
	ax1 = fig.add_axes([0.1,0.1,0.15,0.65])
	d1 = dendrogram(linkage_matrix_1, **kwargs, orientation='left')
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.axis("off")
	# Plot second dendrogram
	# ax1 = fig.add_axes([0.3,0.7,0.6,0.15])
	ax2 = fig.add_axes([0.25,0.75,0.65,0.15])
	d2 = dendrogram(linkage_matrix_2, **kwargs)
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.axis("off")
	# plot matrix
	# axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	axmatrix = fig.add_axes([0.25,0.1,0.65,0.65])
	idx1 = d1['leaves']
	idx2 = d2['leaves']
	matrix = matrix[idx1,:]
	matrix = matrix[:,idx2]
	im = axmatrix.matshow(matrix, aspect='auto', origin='lower', cmap = "plasma")
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])
	# plot colorbar
	if clim is not None:
		im.set_clim(clim[0],clim[1])
	axcolor = fig.add_axes([0.93,0.1,0.02,0.65])
	plt.colorbar(im, cax=axcolor)
	return fig

def plot_dendrogram_with_variable_matrix(linkage_matrix, matrix, var_names, clim = None,  **kwargs):
	from scipy.cluster.hierarchy import dendrogram
	# Initialize figure
	fig = plt.figure(figsize=(10,10))
	# Plot top dendrogram
	ax1 = fig.add_axes([0.1,0.75,0.8,0.15])
	d1 = dendrogram(linkage_matrix, **kwargs)
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.axis("off")
	# plot matrix
	axmatrix = fig.add_axes([0.1,0.1,0.8,0.65])
	idx1 = d1['leaves']
	matrix = matrix[:,idx1]
	im = axmatrix.matshow(matrix, aspect='auto', cmap="plasma")
	axmatrix.set_xticks([])
	axmatrix.set_yticks(np.arange(matrix.shape[0], dtype = int))
	axmatrix.set_yticklabels(var_names)
	# plot colorbar
	if clim is not None:
		im.set_clim(clim[0],clim[1])
	axcolor = fig.add_axes([0.93,0.1,0.02,0.65])
	plt.colorbar(im, cax=axcolor)
	return fig
