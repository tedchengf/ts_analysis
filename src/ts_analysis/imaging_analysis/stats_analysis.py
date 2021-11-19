import numpy as np
from scipy import linalg
from scipy import stats
import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

#------------------------		  PCA Functions	    	-----------------------#

# eeg should be in the dimension of (trial, channel, timepoints)
def PCA_transform(eeg, n_components, variance_threshold):
	# # Below is the sklearn implementation
	# from sklearn.decomposition import PCA
	# temp = np.swapaxes(eeg.copy(),1,2)
	# temp_shape = temp.shape
	# reshaped_eeg = np.reshape(temp, (-1, temp.shape[-1]))
	# model = PCA(n_components = n_components, svd_solver = "full")
	# results = model.fit_transform(reshaped_eeg)
	# # model.fit(reshaped_eeg)
	# # comp_mat = model.components_
	# # explained_var = model.explained_variance_ratio_
	# # results = np.dot(comp_mat, reshaped_eeg.T).T
	# PCA_eeg = np.reshape(results, (temp_shape[0], temp_shape[1], -1))
	# PCA_eeg = np.swapaxes(PCA_eeg, 1, 2)
	# explained_variance = sum(np.around(model.explained_variance_ratio_, decimals = 2)) * 100

	# # Below is the custom eigen decomposition implementation
	temp = np.swapaxes(eeg.copy(),1,2)
	temp_shape = temp.shape
	reshaped_eeg = np.reshape(temp, (-1, temp.shape[-1]))
	PCA_eeg, evals, explained_variance = PCA(reshaped_eeg, n_components, variance_threshold)
	PCA_eeg = np.reshape(PCA_eeg, (temp_shape[0], temp_shape[1], -1))
	PCA_eeg = np.swapaxes(PCA_eeg, 1, 2)
	n_components = len(evals)

	return PCA_eeg, n_components, explained_variance

# This is the support function that contains a custom implementation of eigen decomposition
def PCA(data, n_components = None, var_threshold = None):
	assert type(data) is np.ndarray and len(data.shape) == 2, "data must be an instance of numpy ndarray with exactly 2 dimensions"
	if n_components is not None: assert n_components <= data.shape[1], "n_components must be smaller than or equal to the number of columns"
	if var_threshold is not None: assert 0<= var_threshold and var_threshold <= 1, "invalide var_threshold range; 0 <= var_threshold <= 1"
	n_samples, n_features = data.shape
	# centering
	data -= np.mean(data, axis=0)

	# calculate the covariance matrix
	R = np.cov(data.copy(), rowvar=False)
	# calculate eigenvectors & eigenvalues of the covariance matrix
	# use 'eigh' rather than 'eig' since R is symmetric, 
	# the performance gain is substantial
	evals, evecs = linalg.eigh(R)
	# sort eigenvectors & eigenvalue in decreasing order
	idx = np.argsort(evals)[::-1]
	evecs = evecs[:,idx]
	evals = evals[idx]
	alt_result = np.dot(data, evecs)

	# singular vector decomposition
	U, S, Vt = linalg.svd(data, full_matrices=False)
	U, Vt = _svd_flip(U, Vt)
	# calculate explained variance
	explained_variance = (S ** 2) / (n_samples - 1)
	total_var = explained_variance.sum()
	explained_variance_ratio = explained_variance / total_var
	# calculate projection (after potential dimension reduction)
	n_comp = determine_n_comp(n_components, var_threshold, explained_variance_ratio)
	U = U[:, :n_comp]
	S = S[:n_comp]
	Vt = Vt[:n_comp, :n_comp]
	result = np.matmul(np.matmul(U, np.diag(S)), Vt)

	print (evecs[0, :], Vt.transpose()[0, :])
	print (evecs[1, :], Vt.transpose()[1, :])

	return result, explained_variance[:n_comp], explained_variance_ratio[:n_comp]

def determine_n_comp(n_components, threshold, variance_explained):
	if n_components is not None:
		return n_components
	elif threshold is not None:
		sum_var_explained = 0
		n_components = 0
		while sum_var_explained < threshold and n_components < len(variance_explained):
			sum_var_explained += variance_explained[n_components]
			n_components += 1
		return n_components
	else:
		return len(variance_explained)

def _svd_flip(u, v):
	max_abs_cols = np.argmax(np.abs(u), axis=0)
	signs = np.sign(u[max_abs_cols, range(u.shape[1])])
	u *= signs
	v *= signs[:, np.newaxis]
	return u, v

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

#-----------------------	   Dendrogram Functions	   ------------------------#

def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def generate_linkage_matrix(model):
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count
	linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
	return linkage_matrix

#-----------------------	   Cluster Evaluation	   ------------------------#

def build_tree(clustering_model):
	max_id = clustering_model.n_leaves_
	ii = itertools.count(max_id)
	tree_list = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in clustering_model.children_]
	level_info = dict({})
	tree_root, max_depth = build_node(tree_list[-1]["node_id"], max_id, -1, tree_list, level_info)
	level_dict = __build_layer_dict(level_info)
	return tree_root, max_depth, level_dict

def build_node(curr_id, max_id, depth, tree_list, results):
	# initialize the dictionary
	curr_node = dict({"curr_id": curr_id, "left_child": None, "right_child": None, "children": None, "end": False, "depth": depth+1})
	# create a new depth entry in the results dictionary
	if depth+1 not in results:
		results.update({depth+1:[]})

	# stopping condition
	if curr_id < max_id:
		# The node is a leaf node
		curr_node["children"] = [curr_id]
		curr_node["end"] = True
		max_depth = depth + 1
		results[depth+1].append(curr_node["children"])
		return curr_node, max_depth
	# normal condition
	node_dict = tree_list[curr_id - max_id]
	curr_node["left_child"], left_max = build_node(node_dict["left"], max_id, depth+1, tree_list, results)
	curr_node["right_child"], right_max = build_node(node_dict["right"], max_id, depth+1, tree_list, results)
	curr_node["children"] = curr_node["left_child"]["children"] + curr_node["right_child"]["children"]
	results[depth+1].append(curr_node["children"])
	# return condition
	if left_max > right_max:
		max_depth = left_max
	else:
		max_depth = right_max
	return curr_node, max_depth

def __build_layer_dict(level_info):
	# inherit the end nodes to later depth entries
	end_nodes = []
	for lev in level_info.keys():
		curr_lev = level_info[lev]
		# copy end nodes from the previous layer
		prev_end_nodes = end_nodes.copy()
		# append end nodes from the current layer
		for cluster in curr_lev:
			if len(cluster) == 1:
				end_nodes.append(cluster)
		level_info[lev] = curr_lev + prev_end_nodes
	# assemble a dictionary with masks
	level_dict = dict({})
	mask_length = len(level_info[0][0])
	for lev in level_info.keys():
		curr_lev = level_info[lev]
		curr_level_dict = dict({"clusters":curr_lev})
		curr_mask = __build_masks(curr_lev, mask_length)
		curr_level_dict.update({"mask": curr_mask})
		level_dict.update({lev:curr_level_dict})
	return level_dict

def __build_masks(groupings, mask_length):
	mask = np.ones(mask_length) * -1
	for label, cluster in enumerate(groupings):
		mask[cluster] = label
	assert -1 not in mask, "There exist at least one unlabeled data"
	return mask.astype(int)

def evaluate_clusters(var_arrays, level_dict, num_sample, eval_func, skip_start_end = True):
	evaluate_results = []
	for var_ind in range(var_arrays.shape[0]):
		variable = np.array([var_arrays[var_ind]])
		var_score = []
		for lev in level_dict.keys():
			if skip_start_end == True:
				if len(level_dict[lev]["clusters"]) < 2 or len(level_dict[lev]["clusters"]) == num_sample:
					continue
			lev_mask = level_dict[lev]["mask"]
			var_score.append(eval_func(variable.transpose(), lev_mask))
		evaluate_results.append(var_score)
	return np.array(evaluate_results)

def weighted_variance(variable, cluster_labels):
	labels = np.unique(cluster_labels)
	n_sample = len(cluster_labels)
	weighted_variance = 0
	for c_label in labels:
		curr_label_mask = cluster_labels[:] == c_label
		clusters = variable[curr_label_mask]
		curr_variance = np.var(clusters) * (len(clusters)/n_sample)
		weighted_variance = weighted_variance + curr_variance
	return weighted_variance/np.var(variable)

def integration(var_cord):
	assert type(var_cord) is np.ndarray and len(var_cord.shape) == 2, "var_cord must be an intance of np.ndarray with exactly 2 dimensions"
	area = []
	for var in var_cord:
		area.append(np.trapz(var))
	return np.array(area)

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
