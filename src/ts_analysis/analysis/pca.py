import numpy as np
from scipy import linalg

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
	