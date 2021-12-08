# univariate.py

import numpy as np
from scipy import stats
from tqdm import tqdm


###############################################################################
#						   	subject_regression class		   				  #
###############################################################################

'''
Class subject_regression

	Attributes:
		- sub_data: the subject data with shape = (, N). 
		predictors: the predictors as a 2D numpy array with shape = 
		(variables, values)
		- betas: the beta values of regression with shape = (, N, variables)

	Functions:
		- fit(weight_mask = None): calculate the betas and weighted betas. The calculation is completed recursively, with each iteration using slices of data in the last dimension independently.
		- weighted_beta(weight_mask): calculate the weighted betas given the
		weight_masks
'''
class subject_regression:
	'''
	__init__(self, sub_data, predictors)

		- sub_data: the subject data used for regression. sub_data must be an
		instance of np.ndarray with at least 2 dimensional, with the last
		dimension containing values for regression analysis.

		- predictors: the predictors data for fitting. predictors must be an
		instance of np.ndarray with exactly 2 dimensions: (variables, values)
	'''
	def __init__(self, sub_data, predictors):
		assert type(sub_data) is np.ndarray and len(sub_data.shape) >= 2, "sub_data must be an instance of numpy ndarray with at least 2 dimensions"
		assert type(predictors) is np.ndarray and len(predictors.shape) == 2, "predictors must be an instance of numpy ndarray with exactly 2 dimensions"
		assert sub_data.shape[-1] == predictors.shape[-1], str(sub_data.shape) + " does not match " + str(predictors.shape) + " at the last dimension"
		
		# Class Variables
		self.sub_data = sub_data
		self.predictors = predictors
		self.betas = None

	'''
	fit(self, weight_mask = None)
		
		- weight_mask: the mask to select a subset of predictor data for 
		calculating the weighted beta. weight_mask must be an instance of
		np.ndarray with exactly 2 dimensions (weight type, mask). For example,
		suppose the  predictor is [[1,2,3,4,5,6], [6,5,4,3,2,1]] and the 
		weight_mask is [[1,1,1,0,0,0]], then the weighted betas of the
		predictors will be avg([1,2,3])*beta and avg([6,5,4]), as
		the weight_mask only specify the variables of the first three trials
		to be included. Each weight mask (weight type dimension) will be
		applied to all predictors.
		Default = None; all trials will be included to calculate the weighted
		beta. If weight_mask != None, then the return object weighted_beta will
		be an numpy ndarray with shape = (weight type, variables, N)
		Note: This parameter only affects how weighted beta is calculated; all 
		trials in the predictor will be used to produce beta. 

		returns: betas, weighted_betas
	'''
	def fit(self, weight_mask = None):
		self.betas = self.__recursive_regression(self.sub_data)
		if weight_mask is None:
			average_x = np.average(self.predictors, axis = 1).reshape(1, -1)
			weighted_betas = self.betas * average_x
		else:
			weighted_betas = self.weighted_beta(weight_mask)
		return self.betas.copy(), weighted_betas

	'''
	weighted_beta(self, weight_mask)
		
		- weight_mask: the mask to select subset of predictor data for 
		calculating the weighted beta. weight_mask must be an instance of
		np.ndarray with exactly 2 dimensions (weight type, mask). For example,
		suppose the  predictor is [[1,2,3,4,5,6], [6,5,4,3,2,1]] and the 
		weight_mask is [[1,1,1,0,0,0]], then the weighted betas of the
		predictors will be avg([1,2,3,0,0,0])*beta and avg([6,5,4,0,0,0]), as
		the weight_mask only specify the variables of the first three trials
		to be included. Each weight mask (weight type dimension) will be
		applied to all predictors.
		Note: please run fit() before using weighted_beta()

		returns: weighted_betas
	'''		
	def weighted_beta(self, weight_mask):
		assert self.betas is not None, "betas not calculated; run fit() before using weighted_beta()"
		assert weight_mask.shape[-1] == self.predictors.shape[-1], str(weight_mask.shape) + " does not match " + str(self.predictors.shape) + " at the last dimension"
		assert len(weight_mask.shape) == 2, "weight_mask must be an instance of numpy ndarray with exactly 2 dimensions"
		weighted_betas = []
		for mask in weight_mask:
			curr_average = np.average(self.predictors[:,mask], axis = 1).reshape(1,-1)
			weighted_betas.append(self.betas * curr_average)
		return np.array(weighted_betas)

	def __recursive_regression(self, subproblem):
		# reach the final subproblem
		if len(subproblem.shape) == 1:
			return np.linalg.lstsq(np.swapaxes(self.predictors, 0, 1), subproblem, rcond = None)[0]		
		# not yet reach the final subproblem
		curr_result = []
		for index in range(subproblem.shape[0]):
			subresult = self.__recursive_regression(subproblem[index])
			curr_result.append(subresult)
		return np.array(curr_result)

###############################################################################
#						   	  group_regression class		   				  #
###############################################################################

# Todo: test weight_mask feature
class group_regression:
	def __init__(self, name, sub_data, predictors, weight_mask = None):
		# Class Variables
		self.name = name
		self.subjects = []
		self.subject_betas = []
		self.subject_weighted_betas = []
		self.group_beta = []
		self.group_weighted_beta = []

		for index in range(len(sub_data)):
			self.subjects.append(subject_regression(sub_data[index], predictors[index], weight_mask = weight_mask))
		self.subjects = np.array(self.subjects)

	def run_regression(self):
		pbar = tqdm(total = len(self.subjects))
		for sub_regression in self.subjects:
			sub_beta, sub_weighted_beta = sub_regression.regression()
			self.subject_betas.append(sub_beta)
			self.subject_weighted_betas.append(sub_weighted_beta)
			pbar.update(1)
		self.subject_betas = np.array(self.subject_betas)
		self.subject_weighted_betas = np.array(self.subject_weighted_betas)
		self.group_beta = np.average(self.subject_betas, axis = 0)
		self.group_weighted_beta = np.average(self.subject_weighted_betas, axis = 0)
		return self.group_beta, self.group_weighted_beta

###############################################################################
#						   		helpful functions			   				  #
###############################################################################

# Note: var_data should be in the shape of (variable, values); sub_ind starts from 0
def get_conditions(mask, var_data, sub_ind, sub_eeg, weight_mask = None):
	mask = mask[sub_ind]
	var_data = var_data[:,sub_ind]
	if weight_mask is None:
		return var_data[mask], sub_eeg[mask]
	else:
		weight_mask = weight_mask[:,sub_ind]
		return var_data[:,mask], sub_eeg[mask], weight_mask[:,mask]

# Note: the predictors are in the shape of (variable, values)
def z_transform(predictors):
	assert len(predictors.shape) == 2
	z_scores = []
	for row in predictors:
		z_scores.append(stats.zscore(row))
	return np.array(z_scores)

# Note: the predictors are in the shape of (variable, values), and the intercept is added to the front row
def add_intercept(predictors):
	assert len(predictors.shape) == 2
	intercept = np.ones((1, predictors.shape[1]), dtype = float)
	predictors = np.vstack((intercept, predictors))
	return predictors

# Recursively apply baseline correction to the last dimension.
def recursive_baseline_correction(data, baseline_index):
	# reach the final subproblem
	if len(data.shape) == 1:
		return baseline_correction(data, baseline_index)		
	# not yet reach the final subproblem
	curr_result = []
	for index in range(data.shape[0]):
		subresult = recursive_baseline_correction(data[index], baseline_index)
		curr_result.append(subresult)
	return np.array(curr_result)

# Note: the time should be the first dimension for this function
def baseline_correction(ERP, baseline_index):
	baseline = np.sum(ERP[:baseline_index])/baseline_index
	return ERP - baseline


