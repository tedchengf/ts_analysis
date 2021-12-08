# matops.py

import numpy as np
from numba import njit, jit, prange

#-------------------------		Matrix Operations   	-----------------------#

# Parameter input_matrix must be an instance of numpy ndarray
def extract_upper_triangular(input_matrix):
	dim = input_matrix.shape[0]
	array_len = (dim * dim - dim) // 2
	array_result = np.empty(array_len, dtype = input_matrix.dtype)
	fast_extract(input_matrix, array_result)
	return array_result

# Parameter triangular_array must be an instance of numpy ndarray
def expand_lower_triangular(triangular_array, mat_dim, diag_val = 0):
	mat = np.zeros((mat_dim,mat_dim), dtype = triangular_array.dtype)
	fast_expand(triangular_array, mat, diag_val)
	return mat

# Parameter arr_ind must be an instance of numpy ndarray
def extract_tri_ind(arr_ind, mat_dim):
	tri_ind = np.empty(find_tri_dim(len(arr_ind)), dtype = int)
	fast_convert_ind(np.array(arr_ind, dtype = int), mat_dim, tri_ind)
	return tri_ind

def find_tri_dim(dim):
	return (dim * dim - dim)//2

@jit
def matrix_iteration(data_array, target_matrix, function, skip_diagonal=True):
	mat_dim = target_matrix.shape[0]
	if skip_diagonal == True:
		for r in range(mat_dim):
			for c in range(r+1, mat_dim):
				target_matrix[r,c] = function(data_array[r], data_array[c])
				target_matrix[c,r] = target_matrix[r,c]
	else:
		for r in range(mat_dim):
			for c in range(r, mat_dim):
				target_matrix[r,c] = function(data_array[r], data_array[c])
				target_matrix[c,r] = target_matrix[r,c]

#------------------------	Numba Optimized Functions	-----------------------#

# Parameters:
#	matrix: the 2D matrix to extract the triangular array from
#	result: the empty 1D triangular array to store the matrix values (its dtype
#		and dimensions must be specified)
@njit(parallel = True)
def fast_extract(matrix, result):
	dim = matrix.shape[0]
	for row in prange(dim):
		# Calculate the starting and ending index of each row's cell relative to the flattened triangular array
		row_start = int((2*dim - row - 1)*(row/2))
		row_end = int(row_start + dim - row - 1)
		result[row_start:row_end] = matrix[row,row+1:]

# Parameters:
#	triangular_array: the 1D triangular array to expand into a matrix
#	mat: the empty 2D matrix to store the expanded triangular values (its dtype
#		and dimensions must be specified)
#	diag_val: the value of cells on the diagonal
@njit
def fast_expand(triangular_array, mat, diag_val):
	mat_dim = mat.shape[0]
	counter = mat_dim - 1
	start_ind = 0
	for row in range(mat_dim):
		mat[row, row] = diag_val
		mat[row, row+1:] = triangular_array[start_ind:start_ind+counter]
		start_ind += counter
		counter -= 1
		for col in range(row):
			mat[row, col] = mat[col, row]

# Parameters:
# 	arr_ind: 1D array of row/column indices to slice the data (as if obtaining)
# 		the submatrices of the specified rows & columns
# 		and dimensions must be specified)
# 	mat_dim: the dimension of the original matrix
# 	tri_ind: the empty 1D array to store the transformed triangular indices
@njit
def fast_convert_ind(arr_ind, mat_dim, tri_ind):
	arr_dim = len(arr_ind)
	tri_ptr = 0
	for r_ind in range(arr_dim - 1):
		curr_r = arr_ind[r_ind]
		for c_ind in range(r_ind + 1, arr_dim):
			curr_c = arr_ind[c_ind]
			if curr_r < curr_c:
				curr_ind = (2*mat_dim-curr_r-1)*(curr_r/2)+curr_c-curr_r-1
			else:
				curr_ind = (2*mat_dim-curr_c-1)*(curr_c/2)+curr_r-curr_c-1
			tri_ind[tri_ptr] = curr_ind
			tri_ptr += 1