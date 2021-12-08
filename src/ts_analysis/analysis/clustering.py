# clustering.py

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import itertools

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
