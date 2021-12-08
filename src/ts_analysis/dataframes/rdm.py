# rdm.py

from ts_analysis.utilities import aux
from ts_analysis.utilities import matop


import numpy as np
from numba import njit, jit, prange
import warnings

###############################################################################
#								  	RDM class		   						  #
###############################################################################

class RDM:
	# data should be in the dimension of (trial,) or (triangular)
	# trial_identifier should be in the dimension of (trial,)
	def __init__(self, data, name, tri = None, trial_identifier = None):
		self.name = None 					# Name of the current instance
		self.data = None 					# 1D raw data
		self.tri = None 					# 1D Triangular data of RDM
		self.trial_identifier = None 		# 1D identifier array

		self.__trial_identifier_dict = None # Dictionary of identifiers

		# Initialization
		self.name = name
		data = np.array(data)
		assert len(data.shape) == 1, "The parameter data must be an instance of numpy.ndarray with exactly one dimension"
		self.data = data.copy()
		if tri is not None:
			tri = np.array(tri)
			assert len(tri.shape) == 1, "The parameter tri must be an instance of numpy.ndarray with exactly one dimension"
			assert tri.shape[0] == matop.find_tri_dim(data.shape[0]), "The dimension of parameter tri does not agree with that of data"
			self.tri = tri.copy()
		if trial_identifier is not None:
			trial_identifier = np.array(trial_identifier)
			assert len(trial_identifier.shape) == 1, "The parameter trial_identifier must be an instance of numpy.ndarray with exactly three dimensions"
			assert len(trial_identifier) == self.data.shape[0], "The trial dimension of the parameter trial_identifier does not match the first dimension of the parameter data"
			self.trial_identifier = np.array(trial_identifier).copy()
			self.__trial_identifier_dict = dict(zip(trial_identifier, np.arange(len(trial_identifier))))

	def fill(self, DFunc, update = True):
		dis_mat = DFunc(self.data.copy())
		print (dis_mat)
		dim = dis_mat.shape[0]
		tri = np.empty(((self.data.shape[0]*self.data.shape[0]-self.data.shape[0])//2), dtype = dis_mat.dtype)
		for row in range(dim):
			row_start = int((2*dim - row - 1)*(row/2))
			row_end = int(row_start + dim - row - 1)
			tri[row_start:row_end] = dis_mat[row,row+1:]
		if update == True:
			self.tri = tri
		return tri.copy()	

# 								Other Basic Functions						   #

	# Note: if return_instance = False, the tri array returned is a shallow copy
	def slice(self, trial_ind, extract_type = "index", return_type = "tri_arr", silence_warning = False):
		assert extract_type in ("index", "identifier"), "The parameter extract_type must be one from (index, identifier)"
		assert return_type in ("tri_arr", "index", "instance"), "The parameter return_type must be one from (tri_arr, index, instance)"
		if extract_type == "index":
			arr_ind = trial_ind
		else:
			assert self.trial_identifier is not None, "The trial_identifier is not defined"
			arr_ind, missing_keys = aux.dict_arr_query(trial_ind, self.__trial_identifier_dict)
			if len(missing_keys) != 0 and silence_warning == False:
				warnings.warn("The following identifiers are undefined: " + str(missing_keys))
		tri_ind = matop.extract_tri_ind(arr_ind, self.data.shape[0])
		if return_type == "tri_arr": return self.tri[tri_ind].copy()
		elif return_type == "index": return tri_ind
		else: 
			if self.trial_identifier is not None:
				subset_keys = self.trial_identifier[arr_ind]
			else:
				subset_keys = None
			copy_RDM = RDM(self.data[arr_ind], name = self.name, tri = self.tri[tri_ind], trial_identifier = subset_keys)
			return copy_RDM

	def __repr__(self):
		type_str = "Type: RDM"
		name_str = "Data Name: " + self.name
		data_str = "Data: " + str(self.data.shape)
		if self.tri is not None:
			tri_str = "Traingular Array: " + str(self.tri.shape)
		else:
			tri_str = "Triangular Array: Uninitialized"
		if self.trial_identifier is not None:
			identifier_str = "Trial Identifier: " + str(self.trial_identifier.shape)
		else:
			identifier_str = "Trial Identifier: Undefined"
		return type_str + "\n" + name_str + "\n" + data_str + "\n" + tri_str + "\n" + identifier_str

	def __getitem__(self, key):
		assert self.tri is not None, "tri uninitialized"
		arr_ind = np.array(key)
		tri_ind = matop.extract_tri_ind(arr_ind, self.data.shape[0])
		return self.tri[tri_ind]

	def __setitem__(self, key, value):
		assert self.tri is not None, "tri uninitialized"		
		arr_ind = np.array(key)
		tri_ind = matop.extract_tri_ind(arr_ind, self.data.shape[0])
		self.tri[tri_ind] = value

	def copy(self, name = None):
		if name is not None:
			name = self.name
		return RDM(self.data, name, tri = self.tri, trial_identifier = self.trial_identifier)

	def get_full_RDM(self, diag_val = 0):
		RDM_dim = self.data.shape[0]
		return matop.expand_lower_triangular(self.tri, self.data.shape[0], diag_val)

###############################################################################
#								  tsRDM class		   						  #
###############################################################################

class tsRDM:
	# ts_data should be in the dimension of (trial, channel, time points)
	# trial_identifier should be in the dimension of (trial,)
	def __init__(self, ts_data, name, ts_tri = None, trial_identifier = None):
		self.name = None 					# Name of the current instance
		self.ts_data = None 				# 3D time-series data
		self.ts_tri = None 					# 2D time-series triangular data
		self.trial_identifier = None 		# 1D identifier array

		self.__trial_identifier_dict = None # Dictionary of identifiers

		# Initialization
		self.name = name
		ts_data = np.array(ts_data)
		assert len(ts_data.shape) == 3, "The parameter ts_data must be an instance of numpy.ndarray with exactly 3 dimensions"
		self.ts_data = ts_data.copy()
		if ts_tri is not None:
			ts_tri = np.array(ts_tri)
			assert len(ts_tri.shape) == 2, "The parameter trial_identifier must be an instance of numpy.ndarray with exactly 1 dimensions"
			assert ts_tri.shape[1] == matop.find_tri_dim(ts_data.shape[0]), "The dimensions of parameter ts_tri do not agree with those of ts_data"
			self.ts_tri = ts_tri.copy()
		if trial_identifier is not None:
			trial_identifier = np.array(trial_identifier)
			assert len(trial_identifier.shape) == 1, "The parameter trial_identifier must be an instance of numpy.ndarray with exactly three dimensions"
			assert len(trial_identifier) == self.ts_data.shape[0], "The trial dimension of the parameter trial_identifier does not match the first dimension of the parameter data"
			self.trial_identifier = np.array(trial_identifier).copy()
			self.__trial_identifier_dict = dict(zip(trial_identifier, np.arange(len(trial_identifier))))

	def fill(self, time_window = None, step = None, padding = True, DFunc = None, update = True):
		# Initialize data for later processing (molding them into correct arrangements)
		data, ts_tri = self.__initialize_data(self.ts_data.copy(), time_window, step, padding)
		# Calculating RDM. The default DFunc is corrcoef, and the calculation is parallelized to reduce the runtime. 
		if DFunc is None:
			quick_pearsonr_tsRDM(data, ts_tri)
		else:
			corr_tsRDM(data, ts_tri, DFunc)
		# Update & Return
		if update == True:
			self.ts_tri = ts_tri.copy()
		return ts_tri

	def transform(self, TFunc, update = False, return_type = "tri_arr"):
		assert return_type in ("tri_arr", "instance"), "The parameter return_type must be one from (tri_arr, instance)"
		if update == False:
			transformed_ts_tri = np.empty((self.ts_tri.shape))
		else:
			transformed_ts_tri = self.ts_tri
		for tri_ind in range(self.ts_tri.shape[0]):
			curr_tri = self.ts_tri[tri_ind, :]
			transformed_ts_tri[tri_ind, :] = TFunc(curr_tri)
		if return_type == "instance":
			copy_tsRDM = tsRDM(self.ts_data, name = self.name, ts_tri = transformed_ts_tri, trial_identifier = self.trial_identifier)
			return copy_tsRDM
		else:
			return transformed_ts_tri.copy()

# 								Other Basic Functions						   #

	def __repr__(self):
		type_str = "Type: tsRDM"
		name_str = "Data Name: " + self.name
		ts_data_str = "Time Series Data: " + str(self.ts_data.shape)
		if self.ts_tri is not None:
			ts_tri_str = "Time Series Traingular Array: " + str(self.ts_tri.shape)
		else:
			ts_tri_str = "Time Series Triangular Array: Uninitialized"
		if self.trial_identifier is not None:
			identifier_str = "Trial Identifier: " + str(self.trial_identifier.shape)
		else:
			identifier_str = "Trial Identifier: Undefined"
		return type_str + "\n" + name_str + "\n" + ts_data_str + "\n" + ts_tri_str + "\n" + identifier_str

	def __getitem__(self, key):
		assert self.ts_tri is not None, "ts_tri uninitialized"
		arr_ind = np.array(key)
		tri_ind = matop.extract_tri_ind(arr_ind, self.ts_data.shape[0])
		return self.ts_tri[:, tri_ind]

	def __setitem__(self, key, value):
		assert self.ts_tri is not None, "ts_tri uninitialized"		
		arr_ind = np.array(key)
		tri_ind = matop.extract_tri_ind(arr_ind, self.ts_data.shape[0])
		self.ts_tri[:, tri_ind] = value

	def copy(self, name = None):
		if name is not None:
			name = self.name
		return tsRDM(self.ts_data, name, ts_tri = self.ts_tri, trial_identifier = self.trial_identifier)

	# Note: if return_instance = False, the tri array returned is a shallow copy
	def slice(self, trial_ind, extract_type = "index", return_type = "tri_arr", silence_warning = False):
		assert extract_type in ("index", "identifier"), "The parameter extract_type must be one from (index, identifier)"
		assert return_type in ("tri_arr", "index", "instance"), "The parameter return_type must be one from (tri_arr, index, instance)"
		assert self.ts_tri is not None, "ts_tri is uninitialized"
		if extract_type == "index":
			arr_ind = trial_ind
		else:
			assert self.trial_identifier is not None, "The trial_identifier is not defined"
			arr_ind, missing_keys = aux.dict_arr_query(trial_ind, self.__trial_identifier_dict)
			if len(missing_keys) != 0 and silence_warning == False:
				warnings.warn("The following identifiers are undefined: " + str(missing_keys))
		tri_ind = matop.extract_tri_ind(arr_ind, self.ts_data.shape[0])
		if return_type == "tri_arr": return self.ts_tri[:,tri_ind].copy()
		elif return_type == "index": return tri_ind
		else: 
			if self.trial_identifier is not None:
				subset_keys = self.trial_identifier[arr_ind]
			else:
				subset_keys = None
			copy_tsRDM = tsRDM(self.ts_data[arr_ind,:,:], name = self.name, ts_tri = self.ts_tri[:,tri_ind], trial_identifier = subset_keys)
			return copy_tsRDM

	def get_full_RDM(self, diag_val = 0):
		RDM_dim = self.ts_data.shape[0]
		ts_RDM = []
		for tri in self.ts_tri:
			ts_RDM.append(matop.expand_lower_triangular(tri, self.ts_data.shape[0], diag_val))
		return np.array(ts_RDM)

#------------------------------- Private Functions ---------------------------#

	def __initialize_data(self, data, time_window, step, padding):
		if time_window is None:
			ts_tri = np.empty((data.shape[2], matop.find_tri_dim(data.shape[0])))	
			return data, ts_tri
		else:
			assert type(time_window) is int and type(step) is int, "time_window and step must be integers"
			# creating padding
			if padding == True:
				assert (time_window - 1) % 2 == 0 and time_window >= 3, "if padding is set to True, time_window must be an odd integer greater than or equal to 3"
				padding_len = (time_window - 1) // 2
				padding_data = np.zeros((data.shape[0], data.shape[1], padding_len))
				# append padding to the front and the end on time axis
				data = np.append(padding_data, data, axis = 2)
				data = np.append(data, padding_data, axis = 2)
			max_time = data.shape[2] - time_window
			tw_start_pos = np.arange(0, data.shape[2] - time_window, step)
			upper_tri_dim = matop.find_tri_dim(data.shape[0])
			ts_tri = np.empty((max_time//step + 1, upper_tri_dim))
			tw_data = np.empty((data.shape[0], data.shape[1]*time_window, max_time//step + 1))
			# rearrange original data
			for tw_ind in range(tw_data.shape[2]):
				curr_start_pos = tw_start_pos[tw_ind]
				tw = data[:,:,curr_start_pos:curr_start_pos + time_window].reshape(data.shape[0], data.shape[1]*time_window)
				tw_data[:,:,tw_ind] = tw
			return tw_data, ts_tri		

###############################################################################
#							   Support functions		   					  #
###############################################################################

# Parameters:
#	data: 3D array of neural recording data, with the dimension of (trial,
#		channel, time points)
#	results: the 2D empty triangular array to record the results. It's dimension
#		should be (time points, triangular array dim)
# 	DFunc: a function that measures the similarity between two sets of data. It
#		takes a 2D array of dimension (trial, channel) as its input, and returns
#		a square 2D matrix of dimension (trial, trial)
def corr_tsRDM(data, results, DFunc):
	for t_ind in range(data.shape[2]):
		curr_data = data[:,:,t_ind]
		dis_mat = DFunc(curr_data)
		dim = dis_mat.shape[0]
		for row in range(dim):
			row_start = int((2*dim - row - 1)*(row/2))
			row_end = int(row_start + dim - row - 1)
			results[t_ind][row_start:row_end] = dis_mat[row,row+1:]

# Similar to corr_tsRDM, but optimized for np.corrcoef
@njit(parallel = True)
def quick_pearsonr_tsRDM(data, results):
	for t_ind in prange(data.shape[2]):
		curr_data = data[:,:,t_ind]
		dis_mat = 1 - np.corrcoef(curr_data)
		dim = dis_mat.shape[0]
		for row in range(dim):
			row_start = int((2*dim - row - 1)*(row/2))
			row_end = int(row_start + dim - row - 1)
			results[t_ind][row_start:row_end] = dis_mat[row,row+1:]

# @njit(parallel = True)
# def quick_pearsonr_tstri(ts_tri, candidate_tri, result_ts_tri):
# 	for t_ind in prange(ts_tri.shape[0]):
# 		target_tri = ts_tri[t_ind, :]
# 		result_ts_tri[t_ind] = np.corrcoef(target_tri, candidate_tri)[0,1]

# @njit
# def quick_pearsonr_tsRDM(data, results):
# 	for index in range(data.shape[2]):
# 		curr_data = data[:,:,index]
# 		dis_mat = 1 - np.corrcoef(curr_data)
# 		dim = dis_mat.shape[0]
# 		for row in range(dim):
# 			row_start = int((2*dim - row - 1)*(row/2))
# 			row_end = int(row_start + dim - row - 1)
# 			results[index][row_start:row_end] = dis_mat[row,row+1:]