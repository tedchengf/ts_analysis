'''
Class CPerm_diff
Author: Feng Cheng, fcheng03@tufts.edu
Last Update: Feb 13, 2020

A class for finding significant clusters from a group of 1D data. The distribution of permuted cluster t-values is obtained by finding the t value of the largest cluster each time after permuting rows of A with rows of B and performing pairwise t-test between the cols of diff_AB and a zero matrix.

The steps for generting the distribution is as following:
	For each iteration:
		permute rows A and B
		diff_AB = A - B
		ground_diff = zero matrix
		perform pairwise t-test on cols of diff_AB and cols of ground_diff
		append the t/p value of the largest cluster to the distribution

Args
----
Name: string
	The Name of the variable. Only used by CPerm.print_attribute_info().
A: numpy 2D array, shape = (num_total_examples, length_per_example)
	Each row is an example of an 1D data. For example, in correlation processing, each row is the subjects' 1D correlation data, and each col is a time point of the correlation data.
B: numpy 2D array, shape = (num_total_examples, length_per_example)
	Each row is an example of an 1D data that represents the Null state. For example, in correlation processing, each row is the subjects' 1D permuted correlation data, and each col is a time point of the correlation data.
per_iter: int, must be larger than 0
	The number of permutations the class will perform on the cluster.
ttest_type: string, either "pw" or "ind". default = "pw"
	The type of t-test applied. "pw" is pairwise, "ind" is 
per_prop: float, between 0 and 1, default = 0.5
	The proportion of examples permuted during each iteration of permutation   (if you do not know exactly what this argument does, leave it to be 0.5).
threshold: float, between 0 and 1, default = 0.05
	The threshold for identifying the significant clusters. The threshold should be in p value.
tails: int, {-1, 0, 1}, default = 0
	The tail for obtaining the significances of clusters.
	-1 : negative side
	 1 : positive side
	 0 : both sides 
min_span: int, must be larger than 0, default = 2
	The minimal span of a cluster. For example, if min_span = 2, a cluster need to have at least 2 cols (or 2 time points) to be identified. Useful for removing significant clusters that are caused by noises.
show_progress:  bool, default = False
	If show_progress == True, the class will print a progress bar for permutation
show_attributes: bool, default = False
	If show_attributes == True, the class will print out information of the class attributes after all clusters are found. This can also be achieved by calling CPerm.print_attribute_info().
pos_label: python 1D list, len(pos_label) == A.shape[1], default = None
	The position label for the columns. If pos_label is provided, the class will also provide the start and the end of each cluster in terms of their position labels.

Returns
-------
instance: a CPerm instance
output_matrix: 2D numpy array
	A matrix containing information of all significant clusters. Each row of the matrix represent a significant cluster.
	If pos_label is provided, the cols are arranged in the following format:
		[start and end position (tuple), start and end position label (tuple), t value of the cluster (float), p value of the cluster (float)]
	If pos_label is None, the cols are arrnged in the following format:
		[start and end position (tuple), t value of the cluster (float), p value of the cluster (float)]
	Note: The start position of a cluster is the index of the first element of the cluster in the array, and the end position of a cluster is the index of the last element of the cluster in the array plus one.
	For example, in the following array,
		[0, 1, 1, 1, 0, 0]
	the start and end position will be (1, 4)
output_pandas: a pandas instance
	The output_matrix arranged in pandas form. Can be printed directly.

Example
-------
>>> from CPerm import *
>>> A = all_subjects
>>> B = all_subjects_permuted
>>> A.shape == B.shape
True
>>> var_CPerm, var_matrix, var_pandas_out = CPerm("OLD", A, B, 1000)
>>> print(var_pandas_out)
		POS   Pos Label     Tval    Pval
1  (47, 52)  (307, 350)  13.1322  0.1095
'''

import sys
import numpy as np
from scipy import stats
from tqdm import tqdm
from numba import njit, jit
import pandas

###############################################################################
#								   CPerm Class								  #
###############################################################################

class CPerm_diff:

	# The __new__ function for creating class
	# Note: I am not very familiar with the new function, and there are some
	# 		problems with this implementation (i.e. pickle will not work on
	# 		the CPerm instance). The new function is used to return vars in
	#		addition to the instance after the instance is created. The class 
	# 		can still function normally with __init__ only, but the user 
	#		needs to access the variables through class attributes
	def __new__(cls, Name, A, B, per_iter, ttest_type = "pw", per_prop=0.5, threshold=0.05, tails=0, min_span = 2, show_progress=False, show_attributes=False, pos_label=None):
		# Checking input formats
		assert A.shape[1] == B.shape[1], "The two input datasets must match in the second dimension"
		assert len(A.shape) == 2 and len(B.shape) == 2, "The input datasets must be 2D"
		assert (per_prop > 0) and (per_prop < 1), "Invalid range; 0 < per_prop < 1"
		assert (threshold > 0) and (per_prop < 1), "Invalid range; 0 < per_prop < 1"
		assert (tails == 0) or (tails == 1) or (tails == -1), "Invalid input; tails must be 0, 1 or -1"
		assert ttest_type in ("pw", "ind"), "Invalid ttest_type; must be either pw or ind"
		if ttest_type == "pw":
			assert A.shape == B.shape, "For ttest_type pw, A and B must match in all dimensions"
		if pos_label is not None:
			assert pos_label.shape[0] == A.shape[1], "The cols of pos_label must match the cols of the input datasets"
		# Creating an CPerm instance
		instance = super(CPerm_diff, cls).__new__(cls)
		instance.__init__(Name, A, B, ttest_type, int(per_iter), per_prop, threshold, tails, min_span, show_progress, show_attributes, pos_label)
		if instance.cluster_num == 0:
			return instance, [], []
		else:
			return instance, instance.output_matrix.copy(), instance.output_pandas.copy()

	def __init__(self, Name, A, B, ttest_type, per_iter, per_prop, threshold, tails, min_span, show_progress, show_attributes, pos_label):
		self.Var_name = Name 					# The name of the variable
		self.A = A.copy() 						# Original Data
		self.B = B.copy() 						# Permuted Data
		self._min_span = min_span 				# min_span for clusters
		self.pos_label = pos_label 				# position label
		self.Tvals = None 						# Tvals for columns
		self.Pvals = None 						# Pvals for columns
		self.pos_cluster_label = None 			# Positive sig cluster pos
		self.neg_cluster_label = None 			# Negative sig cluster pos
		self.cluster_pos = None 				# All cluster pos
		self.cluster_num = None 				# Total num of sig clusters
		self.cluster_label = None 				# pos_label
		self.cluster_tvals = None 				# Tvals of all sig clusters
		self.cluster_pvals = None 				# Pvals of all sig clusters
		self.cluster_tvals_distribution = None 	# Permutation Tval distribution
		self.output_matrix = None 				# output matrix for sig cluster
		self.output_pandas = None 				# output pandas for sig cluster

		# Find t and p values for each col
		self.__find_t_p_vals(ttest_type)
		# Find significant clusters
		self.__find_clusters(threshold, tails)
		if self.cluster_num != 0:
			# Permute A and B
			self.__cluster_permutation(ttest_type, per_prop, per_iter, threshold, show_progress)
			# obtain cluster_label (return None if pos_label is None)
			self.__find_cluster_label()
			# Formulate clusters into matrix and pandas format
			self.__form_output_matrix()
		if show_attributes == True:
			self.print_attribute_info()
		return 

	def print_attribute_info(self):
		print ("Variable = " + self.Var_name)
		print ("data A: " + str(self.A.shape))
		print ("data B: " + str(self.B.shape))
		print ("positive label: ", end = "")
		if self.pos_cluster_label is None:
			print ("not avaliable")
		else:
			print ("avaliable")
		print ("negative label: ", end = "")
		if self.neg_cluster_label is None:
			print ("not avaliable")
		else:
			print ("avaliable")
		print ("cluster information")
		if self.output_pandas is None:
			print ("No cluster found")
		else:
			print (self.output_pandas)
		return
	
	# Find significant clusters with a different threshold
	def find_sig_clusters(self, threshold):
		sig_mask = np.array(self.cluster_pvals <= threshold, dtype = bool)
		return self.cluster_pos[sig_mask].copy(), self.cluster_tvals[sig_mask].copy(), self.cluster_pvals[sig_mask].copy()

	def calculate_std_err(self):
		mean = np.mean(self.A, axis = 0)
		std = np.std(self.A, axis = 0)
		return mean, std

	def get_sig_intervals(self, threshold=0.05, pos_offset=0):
		if self.cluster_pvals is None: return {self.Var_name: []}
		sig_mask = np.array(self.cluster_pvals <= threshold, dtype = bool)
		if sum(sig_mask) == 0: return {self.Var_name: []}
		return {self.Var_name: np.add(self.cluster_pos[sig_mask].copy(), pos_offset)}

	def __repr__(self):
		return self.Var_name + "\n" + self.output_pandas.__repr__()

#								Private Functions							  #
	
	# Find the t and p values for each column
	def __find_t_p_vals(self, ttest_type):
		all_tval, all_pval = self.__t_test(self.A, self.B, ttest_type)
		self.Tvals = all_tval
		self.Pvals = all_pval
		return all_tval, all_pval

	# Find the significant clusters
	def __find_clusters(self, threshold, tails):
		pos_mask, neg_mask = self.__get_cluster_masks(threshold, self.Pvals, self.Tvals)
		if tails == 0:
			pos_clusters, pos_num, pos_pos = self.__label_clusters(pos_mask)
			neg_clusters, neg_num, neg_pos = self.__label_clusters(neg_mask)
			pos_cluster_tvals = self.__find_cluster_tval(self.Tvals, pos_pos)
			neg_cluster_tvals = self.__find_cluster_tval(self.Tvals, neg_pos)
			self.pos_cluster_label = np.array(pos_clusters)
			self.neg_cluster_label = np.array(neg_clusters)
			self.cluster_num = pos_num + neg_num
			self.cluster_pos = np.array(pos_pos + neg_pos)
			self.cluster_tvals = np.array(pos_cluster_tvals + neg_cluster_tvals)
		if tails == 1:
			pos_clusters, pos_num, pos_pos = self.__label_clusters(pos_mask)
			pos_cluster_tvals = self.__find_cluster_tval(self.Tvals, pos_pos)
			self.pos_cluster_label = np.array(pos_clusters)			
			self.cluster_num = pos_num
			self.cluster_pos = np.array(pos_pos)
			self.cluster_tvals = np.array(pos_cluster_tvals)
		if tails == -1:
			neg_clusters, neg_num, neg_pos = self.__label_clusters(neg_mask)
			neg_cluster_tvals = self.__find_cluster_tval(self.Tvals, neg_pos)
			self.neg_cluster_label = np.array(neg_clusters)			
			self.cluster_num = neg_num
			self.cluster_pos = np.array(neg_pos)
			self.cluster_tvals = np.array(neg_cluster_tvals)	
		return self.cluster_num, self.cluster_pos.copy(), self.cluster_tvals.copy()

	# Permute A and B to obtain cluster_tvals_distribution
	def __cluster_permutation(self, ttest_type, proportion, per_iter, threshold, show_progress):
		split_ind = self.A.shape[0]
		merged_data = np.concatenate((self.A, self.B), axis = 0)
		indices = np.arange(merged_data.shape[0])

		cluster_tval_distribution = []
		if show_progress == False:
			iter_range = range(per_iter)
		else:
			# tqdm wrapper is for drawing progress bar.
			# Function normally like a for loop
			iter_range = tqdm(range(per_iter), file=sys.stdout)
		for index in iter_range:
			np.random.shuffle(indices)
			temp_data = merged_data[indices, :]
			temp_A = temp_data[:split_ind, :]
			temp_B = temp_data[split_ind:, :]
			# obtaining t/p values of the newly permuted data
			curr_tvals, curr_pvals = self.__t_test(temp_A, temp_B, ttest_type)
			# finding masks for significant clusters
			pos_mask, neg_mask = self.__get_cluster_masks(threshold, curr_pvals, curr_tvals)
			# finding the clusters
			labels = np.zeros(self.A.shape[1], dtype = int)
			pos_labels, pos_num, pos_pos = label_clusters(pos_mask, self._min_span, labels)
			neg_labels, neg_num, neg_pos = label_clusters(neg_mask, self._min_span, labels)
			# No clusters
			if (pos_num == 0) and (neg_num == 0):
				cluster_tval_distribution.append(0)
			else:
				pos_cluster_tvals = self.__find_cluster_tval(curr_tvals, pos_pos)
				neg_cluster_tvals = self.__find_cluster_tval(curr_tvals, neg_pos)
				all_cluster_tvals = np.concatenate((pos_cluster_tvals, np.absolute(neg_cluster_tvals)))
				# apeend the new t value to the distribution
				cluster_tval_distribution.append(np.amax(all_cluster_tvals))
		# sorting and completing the negative side of the distribution
		# there are twice as many t values as there are iters of permutation
		pos_dist = np.sort(cluster_tval_distribution)
		neg_dist = np.flip(pos_dist) * -1
		self.cluster_tvals_distribution = np.concatenate((neg_dist, pos_dist))
		self.cluster_pvals = self.__find_cluster_pval(self.cluster_num, self.cluster_tvals, self.cluster_tvals_distribution)
		return self.cluster_tvals_distribution.copy(), self.cluster_pvals.copy()
	
	# Create a mask that labels significant cols 
	def __get_cluster_masks(self, threshold, pvals, tvals):
		pval_mask = np.array(pvals <= threshold, dtype = int)
		masked_tvals = np.multiply(pval_mask, tvals)
		pos_mask = np.array(masked_tvals > 0, dtype = int)
		neg_mask = np.array(masked_tvals < 0, dtype = int)
		return pos_mask, neg_mask

	# Calculate the p value of a cluster from cluster_tvals_distribution
	def __find_cluster_pval(self, cluster_num, cluster_tvals, cluster_tvals_distribution):
		pvals = []
		for cluster_index in range(cluster_num):
			tval = cluster_tvals[cluster_index]
			if tval >= 0:
				pval = np.sum(cluster_tvals_distribution >= tval)/len(cluster_tvals_distribution)
				pvals.append(pval)
			else:
				pval = np.sum(cluster_tvals_distribution <= tval)/len(cluster_tvals_distribution)
				pvals.append(pval)
		return np.array(pvals)

	# Find the t/p value of all cols
	def __t_test(self, A, B, ttest_type):
		all_tval = []
		all_pval = []
		for time_ind in range(A.shape[1]):
			if ttest_type == "ind":
				tval, pval = stats.ttest_ind(A[:,time_ind], B[:,time_ind])
			else:
				tval, pval = stats.ttest_rel(A[:,time_ind], B[:,time_ind])
			all_tval.append(tval)
			all_pval.append(pval)
		return np.array(all_tval), np.array(all_pval) 

	# Label the clusters
	def __label_clusters(self, mask):
		labels = np.zeros(len(mask), dtype = int)		
		num_cluster = 0
		pos_cluster = []
		index = 0
		while index < len(mask):
			if mask[index] == 1:
				cluster_start = index
				while mask[index] == 1:
					index = index + 1
					# cluster reaches the end of array
					if index == len(mask):
						break
				cluster_end = index
				if cluster_end - cluster_start >= self._min_span:
					labels[cluster_start:cluster_end] = num_cluster + 1
					pos_cluster.append((cluster_start, cluster_end))
					num_cluster = num_cluster + 1
			index = index + 1
		return labels, num_cluster, pos_cluster

	# Find the sum of tvals of a cluster
	def __find_cluster_tval(self, tvals, pos):
		cluster_tvals = []
		for pos_tuple in pos:
			curr_tval = np.sum(tvals[pos_tuple[0]:pos_tuple[1]])
			cluster_tvals.append(curr_tval)
		return cluster_tvals

	# Find the cluster label based on pos_label
	def __find_cluster_label(self):
		if self.pos_label is None:
			return
		cluster_label = []
		for pos in self.cluster_pos:
			if pos[1] == self.pos_label.shape[0]:
				cluster_label.append((self.pos_label[pos[0]], self.pos_label[pos[1] - 1]))
			else:
				cluster_label.append((self.pos_label[pos[0]], self.pos_label[pos[1]]))
		self.cluster_label = cluster_label

	# Formulate the output
	def __form_output_matrix(self):
		output_matrix = []
		for index in range(len(self.cluster_pos)):
			line_output = []
			line_output.append(index + 1)
			line_output.append((self.cluster_pos[index][0], self.cluster_pos[index][1]))
			if self.cluster_label is not None:
				line_output.append((self.cluster_label[index][0], self.cluster_label[index][1]))
			line_output.append(self.cluster_tvals[index])
			line_output.append(self.cluster_pvals[index])
			output_matrix.append(line_output)
		output_matrix = np.array(output_matrix, dtype = object)
		if self.cluster_label is not None:
			label = ["POS", "Pos Label", "Tval", "Pval"]
		else:
			label = ["POS", "Tval", "Pval"]
		self.output_matrix = output_matrix
		self.output_pandas = pandas.DataFrame(output_matrix[:,1:], columns = label, index = output_matrix[:,0])

@njit
def label_clusters(mask, min_span, labels):		
	num_cluster = 0
	pos_cluster = []
	index = 0
	while index < len(mask):
		if mask[index] == 1:
			cluster_start = index
			while mask[index] == 1:
				index = index + 1
				# cluster reaches the end of array
				if index == len(mask):
					break
			cluster_end = index
			if cluster_end - cluster_start >= min_span:
				labels[cluster_start:cluster_end] = num_cluster + 1
				pos_cluster.append((cluster_start, cluster_end))
				num_cluster = num_cluster + 1
		index = index + 1
	return labels, num_cluster, pos_cluster

###############################################################################
#							  Auxiliary Functions							  #
###############################################################################

# def summarize_sig(var_name, output_df, p_thres=0.05, res_dict = None, pos_offset=0):
# 	if res_dict is None: res_dict = dict({})
# 	sig_intervals = []
# 	if type(output_df) is not not pandas.core.frame.DataFrame:
# 		res_dict.update({var_names:sig_intervals})
# 	else:
# 		for cluster_ind in range(len(output_df["POS"])):
# 			if output_df["Pval"][cluster_ind + 1] <= p_thres:
# 				curr_pos = output_df["POS"][cluster_ind + 1]
# 				curr_pos = np.add(curr_pos, pos_offset)
# 				sig_intervals.append()

def assemble_highlight_intervals(output_dfs, var_names, var_colors=None, p_thres=0.05, pos_offset=0):
	if len(output_dfs) == 0:
		return None, None
	if var_colors is not None:
		assert len(var_colors) == len(output_dfs), "The number of var_colors specified must match the number of variables contained in output_dfs"
	highlight_intervals = []
	highlight_colors = []
	highlight_vars = []
	print(output_dfs[0])
	for var_ind, var_df in enumerate(output_dfs):
		if type(var_df) is not pandas.core.frame.DataFrame: continue
		for cluster_ind in range(len(var_df["POS"])):
			if var_df["Pval"][cluster_ind + 1] <= p_thres:
				curr_pos = var_df["POS"][cluster_ind + 1]
				curr_pos = np.add(curr_pos, pos_offset)
				highlight_intervals.append(curr_pos)
				if var_colors is not None:
					highlight_colors.append(var_colors[var_ind])
				else:
					highlight_colors.append("red")
				highlight_vars.append(var_names[var_ind])
	if len(highlight_intervals) == 0:
		return None, None, None
	else:
		return np.array(highlight_intervals), np.array(highlight_vars), np.array(highlight_colors)