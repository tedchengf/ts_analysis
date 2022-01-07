# multivariate.py

import numpy as np
from numba import njit, jit, prange
from scipy import stats
from tqdm import tqdm
import warnings
import matplotlib
import matplotlib.pyplot as plt

from ts_analysis.dataframes import rdm
from ts_analysis.utilities import aux
from ts_analysis.utilities import matop
from ts_analysis.dataframes.dframe import DFrame

###############################################################################
#								  tsRSA class 								  #
###############################################################################

class tsRSA:
	def __init__(self, target_tsRDMs, candidate_RDMs, name, identifier):
		self.name = None
		self.target_tsRDMs = None
		self.candidate_RDMs = None
		self.identifier = None
		self.trial_distribution = None

		self.__target_dict = None
		self.__candidate_dict = None
		self.__identifier_dict = None

		self.__initialize_instance(target_tsRDMs, candidate_RDMs, name, identifier)

	def RSA(self, target_names = None, candidate_names = None, interpolation = "subject", show_progress = False):
		if show_progress == True:
			print ("\nPerforming RSA")
		tar_tsRDMs, cand_RDMs = self.__assemble_data(target_names, candidate_names)
		results = []
		pbar = tqdm(total = len(cand_RDMs), disable = not show_progress)
		c_names = []
		for cand in cand_RDMs:
			c_names.append(cand.name)
			if interpolation == "subject":
				sum_tsRDM, count_RDM = self.__average_tsRDM(tar_tsRDMs)
				tri_mask = count_RDM.tri > 0
				averaged_tstri = np.divide(sum_tsRDM.ts_tri[:, tri_mask], count_RDM.tri[tri_mask])
				curr_result = np.empty(averaged_tstri.shape[0])
				quick_pearsonr_tstri_b(averaged_tstri, cand.tri[tri_mask], curr_result)
				results.append([curr_result])
			else:
				target_results = []
				for tsRDM in tar_tsRDMs:
					target_tri = tsRDM.ts_tri
					candidate_tri = cand.slice(tsRDM.identifier, ktype = "identity")
					curr_result = np.empty(tsRDM.ts_tri.shape[0])
					quick_pearsonr_tstri_b(target_tri, candidate_tri, curr_result)
					target_results.append(curr_result)
				results.append(np.array(target_results))
			pbar.update(1)
		if interpolation == "subject": t_names = ["averaged targets"]
		else:
			t_names = []
			for ts_RDM in tar_tsRDMs: t_names.append(ts_RDM.name)
		results = np.array(results)
		print(results.shape)
		# return DFrame(results, ["candidates", "targets", "time points"], [c_names, t_names, np.arange(results.shape[2])])
		return RSA_results(results, t_names, c_names)

	def estimate_upper_lower_bounds(self, TFunc, target_names = None,  show_progress = False):
		if show_progress == True:
			print ("\nEstimating Bounds")
		assert self.identifier is not None, "The identifier is undefined"
		# Step 1: Apply transformation to the target RDMs
		tar_tsRDMs = self.__assemble_data(target_names, None)[0]
		pbar = tqdm(total = 3, disable = not show_progress)
		transformed_tsRDMs = []
		for ts_RDM in tar_tsRDMs:
			transformed_RDM = ts_RDM.transform(TFunc, return_type = "instance")
			transformed_tsRDMs.append(transformed_RDM)
		pbar.update(1)
		# Step 2: Obtain Average target RDM
		sum_tsRDM, count_RDM = self.__average_tsRDM(transformed_tsRDMs)
		pbar.update(1)
		# Step 3: Estimate upper and lower bound
		upperbound = np.zeros((sum_tsRDM.ts_tri.shape[0]))
		lowerbound = np.zeros((sum_tsRDM.ts_tri.shape[0]))
		for ts_RDM in transformed_tsRDMs:
			temp_results = np.zeros((sum_tsRDM.ts_tri.shape[0]))
			curr_tstri = sum_tsRDM.slice(ts_RDM.identifier, ktype = "identity")
			curr_count = count_RDM.slice(ts_RDM.identifier, ktype = "identity")
			# estimate upperbound
			upperbound_tstri = np.divide(curr_tstri, curr_count)
			quick_pearsonr_tstri(ts_RDM.ts_tri, upperbound_tstri, temp_results)
			upperbound += temp_results
			# estimate lowerbound
			curr_tstri -= ts_RDM.ts_tri
			curr_count -= 1
			# A mask is needed because it is possible that the the subject contain triangular values that are not found in the lowerbound_tstri
			curr_count_mask = curr_count > 0
			lowerbound_tstri = np.divide(curr_tstri[:, curr_count_mask], curr_count[curr_count_mask])
			quick_pearsonr_tstri(ts_RDM.ts_tri[:, curr_count_mask], lowerbound_tstri, temp_results)
			lowerbound += temp_results
		pbar.update(1)
		return np.divide(upperbound, len(transformed_tsRDMs)), np.divide(lowerbound, len(transformed_tsRDMs))

	def check_distribution(self):
		count_RDM = rdm.RDM(np.empty((self.identifier.shape[0])), "RDM overlap", tri = np.zeros((matop.find_tri_dim(self.identifier.shape[0])), dtype = int), identifier = self.identifier)
		for ts_RDM in self.target_tsRDMs:
			curr_trial_ind = aux.dict_arr_query(ts_RDM.identifier, self.__identifier_dict)[0]
			curr_tri_ind = matop.extract_tri_ind(curr_trial_ind, len(self.identifier))
			count_RDM.tri[curr_tri_ind] += 1
		return self.trial_distribution.copy(), count_RDM

# 								Other Basic Functions						   #

	def slice(self, trial_ind, ktype = "index", target_names = None, candidate_names = None):
		assert ktype in ("index", "identity"), "The parameter ktype must be one from (index, identity)"
		if ktype == "index":
			extract_identifier = self.identifier[trial_ind]
		else:
			extract_identifier = trial_ind
		tar_tsRDMs, cand_RDMs = self.__assemble_data(target_names, candidate_names)
		new_target_tsRDMs = []
		for ts_RDM in tar_tsRDMs:
			new_target_tsRDMs.append(ts_RDM.slice(extract_identifier, ktype = "identity", return_type = "instance", silence_warning = True))
		new_candidate_RDMs = []
		for cRDM in cand_RDMs:
			new_candidate_RDMs.append(cRDM.slice(extract_identifier, ktype = "identity", return_type = "instance"))
		return tsRSA(new_target_tsRDMs, new_candidate_RDMs, self.name, extract_identifier)

	def copy(self, name = None):
		if name is None:
			name = self.name
		return tsRSA(self.target_tsRDMs.copy(), self.candidate_RDMs.copy(), name, self.identifier)

	def __repr__(self):
		type_str = "Type: tsRSA"
		name_str = "Data Name: " + self.name
		trial_str = "Trial: " + str(len(self.identifier))
		target_str = "Target tsRDMs: " + str(len(self.target_tsRDMs))
		candidate_str = "Candidate RDMs:"
		for k in self.__candidate_dict.keys():
			candidate_str += "\n    - " + k
		return type_str + "\n" + name_str + "\n" + trial_str + "\n" + target_str + "\n" + candidate_str

#------------------------------- Private Functions ---------------------------#

	def __initialize_instance(self, target_tsRDMs, candidate_RDMs, name, identifier):
		self.name = name
		assert len(identifier) == candidate_RDMs[0].data.shape[0]
		# Initialize trial identifiers
		self.identifier = np.array(identifier)
		assert len(identifier.shape) == 1, "The parameter identifier must be an instance of numpy.ndarray with exactly 1 dimensions"
		self.__identifier_dict = dict(zip(identifier, np.arange(len(identifier))))
		# Initialize tsRDMs and trial distribution
		self.target_tsRDMs = np.array(target_tsRDMs)
		self.trial_distribution = np.zeros((len(self.target_tsRDMs), len(self.identifier)), dtype = bool)
		self.__target_dict = {}
		for i, ts_RDM in enumerate(target_tsRDMs):
			assert isinstance(ts_RDM, rdm.tsRDM), "The parameter target_tsRDM must be a list of tsRDM instances"
			self.__target_dict.update({ts_RDM.name: i})
			curr_dist, missing_keys = aux.dict_arr_query(ts_RDM.identifier, self.__identifier_dict)
			assert len(missing_keys) == 0, "The target_tsRDMs contain identifiers uninitialized in the current instance: " + str(missing_keys)
			self.trial_distribution[i][curr_dist] = True
		self.candidate_RDMs = np.array(candidate_RDMs)
		self.__candidate_dict = {}
		for i, c_RDM in enumerate(candidate_RDMs):
			assert isinstance(c_RDM, rdm.RDM), "The parameter candidate_RDMs must be a list of RDM instances"
			assert candidate_RDMs[0].data.shape[0] == c_RDM.data.shape[0], "All RDM instances in the parameter candidate_RDMs should have the same trial dimension"
			self.__candidate_dict.update({c_RDM.name: i})

	def __assemble_data(self, target_names, candidate_names):
		if target_names is None:
			tar_tsRDMs = self.target_tsRDMs
		else:
			tar_ind, missing_keys = aux.dict_arr_query(target_names, self.__target_dict)
			if len(missing_keys) != 0:
				warnings.warn("The following target names are undefined in the current instance: " + str(missing_keys))
			tar_tsRDMs = self.target_tsRDMs[tar_ind]
		if candidate_names is None:
			cand_RDMs = self.candidate_RDMs
		else:
			can_ind, missing_keys = aux.dict_arr_query(candidate_names, self.__candidate_dict)
			if len(missing_keys) != 0:
				warnings.warn("The following candidate names are undefined in the current instance: " + str(missing_keys))
			cand_RDMs = self.candidate_RDMs[can_ind]
		return tar_tsRDMs, cand_RDMs

	def __average_tsRDM(self, ts_RDMs):
		count_RDM = rdm.RDM(np.zeros((self.identifier.shape[0]), dtype = int), "RDM overlap", tri = np.zeros((matop.find_tri_dim(self.identifier.shape[0])), dtype = int), identifier = self.identifier)
		sum_tsRDM = rdm.tsRDM(np.empty((self.identifier.shape[0], 1, 1)), "Sum RDM", ts_tri = np.zeros((ts_RDMs[0].ts_tri.shape[0],matop.find_tri_dim(self.identifier.shape[0]))), identifier = self.identifier)
		for ts_RDM in ts_RDMs:
			curr_trial_ind, missing_keys = aux.dict_arr_query(ts_RDM.identifier, self.__identifier_dict)
			curr_tri_ind = matop.extract_tri_ind(curr_trial_ind, len(self.identifier))
			count_RDM.data[curr_trial_ind] += 1
			count_RDM.tri[curr_tri_ind] += 1
			sum_tsRDM.ts_tri[:, curr_tri_ind] += ts_RDM.ts_tri
		return sum_tsRDM, count_RDM

	def __ts_corr(self, tar_tsRDMs, cand_RDMs, interpolation):
		results = []
		pbar = tqdm(total = len(cand_RDMs), disable = not show_progress)
		c_names = []
		for cand in cand_RDMs:
			c_names.append(cand.name)
			if interpolation == "subject":
				sum_tsRDM, count_RDM = self.__average_tsRDM(tar_tsRDMs)
				tri_mask = count_RDM.tri > 0
				averaged_tstri = np.divide(sum_tsRDM.ts_tri[:, tri_mask], count_RDM.tri[tri_mask])
				curr_result = np.empty(averaged_tstri.shape[0])
				quick_pearsonr_tstri_b(averaged_tstri, cand.tri[tri_mask], curr_result)
				results.append([curr_result])
			else:
				target_results = []
				for tsRDM in tar_tsRDMs:
					target_tri = tsRDM.ts_tri
					candidate_tri = cand.slice(tsRDM.identifier, ktype = "identity")
					curr_result = np.empty(tsRDM.ts_tri.shape[0])
					quick_pearsonr_tstri_b(target_tri, candidate_tri, curr_result)
					target_results.append(curr_result)
				results.append(np.array(target_results))
			pbar.update(1)
		if interpolation == "subject": t_names = ["averaged targets"]
		else:
			t_names = []
			for ts_RDM in tar_tsRDMs: t_names.append(ts_RDM.name)
		return RSA_results(results, t_names, c_names)

###############################################################################
#								 RSA_results class 							  #
###############################################################################

def collapse_RSA_results(all_RSA_results, name, collapse_dimension = "target", target_names = None, candidate_names = None):
	if collapse_dimension == "target":
		target_names = all_RSA_results[0].target_names.copy()
		base_results = all_RSA_results[0].slice(candidate_names = candidate_names)
		for ind in range(1, len(all_RSA_results)):
			if candidate_names is not None: assert all_RSA_results[0].candidate_names == all_RSA_results[ind].candidate_names, "Candidate names mismatch between instances of all_RSA_results"
			target_names = np.append(target_names,all_RSA_results[ind].target_names)
			curr_result = all_RSA_results[ind].slice(candidate_names = candidate_names)
			base_results = np.append(base_results, curr_result, axis = 1)
		if candidate_names is None: candidate_names = all_RSA_results[0].candidate_names
		return RSA_results(base_results, target_names, candidate_names)
	return None

class RSA_results:
	def __init__(self, results, target_names, candidate_names):
		self.name = None
		self.results = None
		self.target_names = None
		self.candidate_names = None
		self.upperbound = None
		self.lowerbound = None

		self.__target_dict = None
		self.__candidate_dict = None

		# Initialization
		results = np.array(results)
		self.results = results
		self.target_names = np.array(target_names)
		self.candidate_names = np.array(candidate_names)
		assert len(results.shape) == 3, "The parameter results must have three dimensions"
		assert results.shape[0] == len(candidate_names), "The parameter candidate_names must match the first dimension of the results"
		assert results.shape[1] == len(target_names), "The parameter target_names must match the second dimension of the results"
		self.__target_dict = dict(zip(target_names,np.arange(results.shape[1])))
		self.__candidate_dict = dict(zip(candidate_names,np.arange(results.shape[0])))

	# TODO: make the interpretation of None input more consistant
	def plot(self, title = None, candidate_names = None, target_names = None, bounds = None, fig = None, start_end = None, interval = 100, axis = [None, None, None, None], colors = None, font_size = 6):
		data_result = self.slice(candidate_names, target_names = target_names, return_type = "instance")
		data = np.average(data_result.results, axis = 1)
		matplotlib.rcParams.update({'font.size': font_size})
		if fig is None:
			fig = plt.figure(figsize = (6.4, 4.8))
		ax = fig.gca()
		ax.margins(x=0)
		ax.axis(axis)
		if title is not None:
			ax.set_title(title)
		if start_end is not None:
			start = int(start_end[0])
			end = int(start_end[1])
			label = np.linspace(start, end, (end-start)//interval+1, dtype=int)
			# x_range = data.shape[1]
			x_range = self.results.shape[2]
			step = int(round(x_range / (len(label) - 1)))
			tick_num = len(np.arange(0, x_range, step = step, dtype = int))
			ax.set_xticks(np.arange(0, x_range, step = step, dtype = int))
			ax.set_xticklabels(label[:tick_num])	
		if bounds is not None:
			assert bounds in ("all", "upper", "lower"), "If defined, the parameter bounds must be one from (all, upper, lower)"
			if bounds == "all" or bounds == "upper":
				assert self.upperbound is not None,"The upperbound is undefined"
				ax.plot(self.upperbound, label = "upperbound", color = "black", linestyle = "-")
			if bounds == "all" or bounds == "lower":	
				assert self.lowerbound is not None,"The lowerbound is undefined"
				ax.plot(self.lowerbound, label = "lowerbound", color = "black", linestyle = ":")
		if len(data_result.candidate_names) > 0:
			for c_ind, c_name in enumerate(data_result.candidate_names):
				if colors is not None:
					ax.plot(data[c_ind], label = c_name, color = colors[c_ind])
				else:
					ax.plot(data[c_ind], label = c_name)
		ax.legend()
		plt.close()
		return fig
		
	def slice(self, candidate_names = None, target_names = None, return_type = "arr"):
		assert return_type in ("arr", "instance"), "The parameter return_type must be one from (arr, instance)"
		if candidate_names is None:
			cand_ind = np.arange(self.results.shape[0])
		else:
			cand_ind, missing_keys = aux.dict_arr_query(candidate_names, self.__candidate_dict)
			if len(missing_keys) > 0:
				warnings.warn("The following candidate names are undefined: " + str(missing_keys))
		if target_names is None:
			tar_ind = np.arange(self.results.shape[1])
		else:
			tar_ind, missing_keys = aux.dict_arr_query(target_names, self.__target_dict)
			if len(missing_keys) > 0:
				warnings.warn("The following target names are undefined: " + str(missing_keys))
		new_results = self.results[np.ix_(cand_ind, tar_ind)].copy()
		if return_type == "arr":
			return self.results[np.ix_(cand_ind, tar_ind)].copy()
		else:
			return (RSA_results(new_results, self.target_names[tar_ind].copy(), self.candidate_names[cand_ind].copy()))

	def __repr__(self):
		type_str = "Type: results"
		if self.name is None: name_str = "Data Name: Undefined"
		else: name_str = "Data Name: " + self.name
		results_str = "Result Dimension: " + str(self.results.shape)
		candidate_str = "Candidate Names:"
		for k in self.candidate_names:
			candidate_str += "\n    - " + k
		return type_str + "\n" + name_str + "\n" + results_str + "\n" + candidate_str


###############################################################################
#							   Support functions		   					  #
###############################################################################

def corrcoef_z_transform(tri):
	tri = np.subtract(1, tri)
	results = np.empty(len(tri), dtype = tri.dtype)
	quick_z_transform(tri, results)
	return results

def invert_corrcoef(tri):
	return np.subtract(1, tri)

def z_transform(tri):
	results = np.empty(len(tri), dtype = tri.dtype)
	quick_z_transform(tri, results)
	return results

@njit(parallel = True)
def quick_z_transform(tri, results):
	for i in prange(len(tri)):
		results[i] = np.log((1+tri[i])/(1-tri[i]))/2

# @njit(parallel = True)
# def quick_pearsonr_tstri(ts_a, ts_b, result_ts):
# 	for t_ind in prange(ts_a.shape[0]):
# 		result_ts[t_ind] = np.corrcoef(ts_a[t_ind,:], ts_b[t_ind,:])[0,1]

def quick_pearsonr_tstri(ts_a, ts_b, result_ts):
	for t_ind in range(ts_a.shape[0]):
		result_ts[t_ind] = np.corrcoef(ts_a[t_ind,:], ts_b[t_ind,:])[0,1]

@njit(parallel = True)
def quick_pearsonr_tstri_b(ts_a, b, result_ts):
	for t_ind in prange(ts_a.shape[0]):
		result_ts[t_ind] = np.corrcoef(ts_a[t_ind,:], b)[0,1]
