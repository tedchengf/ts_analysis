# func.py

import numpy as np
from numba import njit, jit, prange

#------------------------		Distance Functions		-----------------------#

def corr_dist(A):
	return 1 - np.corrcoef(A)

def abs_diff(A):
	target_matrix = np.zeros((len(A), len(A)))
	mat_dim = target_matrix.shape[0]
	for r in range(mat_dim):
		for c in range(r, mat_dim):
			target_matrix[r,c] = np.absolute(np.subtract(A[r], A[c]))
			target_matrix[c,r] = target_matrix[r,c]
	return target_matrix

def cond_diff(A):
	target_matrix = np.ones((len(A), len(A)), dtype = bool)
	mat_dim = target_matrix.shape[0]
	for r in range(mat_dim):
		for c in range(r, mat_dim):
			target_matrix[r,c] = (A[r] == A[c])
			target_matrix[c,r] = target_matrix[r,c]
	return target_matrix

def weighted_euclidian(A, weights):
	matrices = []
	for arr in A:
		mat = np.zeros((len(arr), len(arr)))
		matrix_iteration(arr, mat, squared_dist)
		matrices.append(mat)
	weighted_dist = np.zeros((len(arr), len(arr)))
	for ind in range(len(weights)):
		weighted_dist = weighted_dist + weights[ind] * matrices[ind]
	return np.sqrt(weighted_dist)

#------------------------		Transform Functions		-----------------------#

def corrcoef_z_transform(A):
	A = np.subtract(1, A)
	results = np.empty(len(A), dtype = A.dtype)
	quick_z_transform(A, results)
	return results

def invert_corrcoef(A):
	return np.subtract(1, A)

def z_transform(A):
	results = np.empty(len(A), dtype = A.dtype)
	quick_z_transform(A, results)
	return results

@njit(parallel = True)
def quick_z_transform(A, results):
	for i in prange(len(A)):
		results[i] = np.log((1+A[i])/(1-A[i]))/2
