# matplot.py

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