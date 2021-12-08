# mds.py

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#------------------------		  PCoA Functions	   	-----------------------#

def PCoA(dist_mat, n_dim = 2):
	# Create centering matrix
	n = dist_mat.shape[0]
	centering_matrix = np.eye(n) - (1/n)*(np.ones((n,n)))
	# Square the matrix, and then apply double centering. This is the same as obtaining dist_mat * dist_mat'
	dist_mat = dist_mat**2
	transformed_dist = -0.5*(centering_matrix.dot(dist_mat)).dot(centering_matrix)
	# Extract Eigenvalues and Eigenvectors
	eigenvalues, eigenvectors = np.linalg.eig(transformed_dist)
	eigenvectors = eigenvectors.transpose()
	sorted_indices = np.flip(np.argsort(eigenvalues))
	sorted_eigenval = eigenvalues[sorted_indices]
	sorted_eigenvec = eigenvectors[sorted_indices]
	# Obtain principle coordinates and their corresponding eigenvalues
	PCs = []
	eig_vals = []
	for dim in range(n_dim):
		PC = np.sqrt(sorted_eigenval[dim]) * sorted_eigenvec[dim]
		PCs.append(PC)
		eig_vals.append(np.sqrt(sorted_eigenval[dim]))
	return np.array(PCs), np.array(eig_vals)

#-----------------------	   Dimension Analysis	   ------------------------#

def dimension_analysis(dimension_name, dimension_order, variable_matrix, var_names, save_directory = "./"):
	# Normalize the Matrix
	sorted_mat = normalize(variable_matrix, indices = dimension_order)
	r, p = pearsonr_with_p(sorted_mat, sorted_mat[0, :])
	print (dimension_name)
	for ind in range(len(r)):
		print (str(round(r[ind], 3)) + "	" + str(round(p[ind], 3)))

	fig = plt.figure(figsize=(20,10))
	mat_ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
	im = mat_ax.imshow(sorted_mat, aspect = "auto", cmap = "plasma")
	plt.yticks(np.arange(sorted_mat.shape[0], dtype = int), var_names)
	color_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
	plt.colorbar(im, cax = color_ax, orientation = "horizontal")
	fig.savefig(save_directory + dimension_name + " comparison.png", form = "png", dpi = 500, transparent = True)
	plt.close(fig = fig)

def normalize(matrix, indices = None):
	sorted_mat = []
	for row_ind in range(matrix.shape[0]):
		if indices is not None:
			row_sorted = matrix[row_ind, :][indices]
		else:
			row_sorted = matrix[row_ind, :]
		row_sorted = row_sorted - row_sorted.min()
		row_sorted = row_sorted / row_sorted.max()
		sorted_mat.append(row_sorted)
	return np.array(sorted_mat)

def pearsonr_with_p(matrix, target):
	r = []
	p = []
	for row_ind in range(matrix.shape[0]):
		curr_row = matrix[row_ind, :]
		curr_r, curr_p = stats.pearsonr(curr_row, target)
		r.append(curr_r)
		p.append(curr_p)
	return np.array(r), np.array(p)

def pearson_correlation(matrix, target):
	results = []
	for row_ind in range(matrix.shape[0]):
		curr_row = matrix[row_ind, :]
		results.append(np.corrcoef(curr_row, target)[0, 1])
	return results

def spearman_correlation(matrix, target):
	results = []
	for row_ind in range(matrix.shape[0]):
		curr_row = matrix[row_ind, :]
		results.append(stats.spearmanr(curr_row, target)[0])
	return results
