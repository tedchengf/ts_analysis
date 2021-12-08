# DataFrmae

import numpy as np
from ts_analysis.utilities import aux
import warnings

class Dim:
	def __init__(self, name, identifier, identity_dict = None):
		assert type(name) is str, "the name must be a string"
		self.name = name     					# str
		self.identifier = np.array(identifier)	# array

		self.__identity_dict = None

		if identity_dict is not None:
			self.__identity_dict = identity_dict
		else:
			self.__identity_dict = dict(zip(identifier, np.arange(len(identifier))))

	def slice(self, key, ktype = "identity", disable_warning = False):
		assert ktype in ("index", "identity"), "The parameter ktype must be one from (index, identity)"
		if ktype == "index": arr_ind = key
		else:			
			arr_ind, missing_keys = aux.dict_arr_query(key,self.__identity_dict)
			if len(missing_keys) > 0:
				if disable_warning == False:
					warnings.warn("The following identifiers are undefined: " + str(missing_keys))
		return arr_ind, Dim(self.name, self.identifier[arr_ind])

	def copy(self):
		return Dim(self.name,self.identifier.copy(),self.__identity_dict.copy())

	def __len__(self):
		return len(self.identifier)

	def __str__(self):
		return "Dim '" + self.name + "' with " + str(len(self)) + " identifiers as " + str(self.identifier.dtype)

	def __getitem__(self, key):
		return self.identifier[key]

	def __eq__(self, other):
		if isintance(other, Dim):
			return other.name == self.name and np.array_equal(other.identifier, self.identifier)
		return False

class Cond:
	def __init__(self, cond_lists, identifier, cond_dim=None, cond_dict=None):
		self.conditions = None
		self.cond_dict = None
		self.cond_dim = None

		self.__initialize(cond_lists, identifier, cond_dim, cond_dict)

	def slice(self, key, ktype = "identity"):
		arr_ind, new_dim = self.cond_dim.slice(key, ktype)
		new_dict = self.cond_dict.copy()
		for k in new_dict.keys():
			new_dict[k] = new_dict[k][arr_ind]
		return Cond(self.conditions.copy(), None, new_dim, new_dict)
	
	def add_cond(self, cond_lists):
		cond_lists = np.array(cond_lists)
		assert len(cond_lists.shape) < 3, "cond_lists can have at most two dimensions"
		if len(cond_lists.shape) == 1:
			assert len(self.cond_dim) == cond_lists.shape[0], "Mismatch in dimension between condition and identifier"
			self.__add_cond(cond_lists)
		else:
			assert len(self.cond_dim) == cond_lists.shape[1], "Mismatch in dimension between condition and identifier"
			for cond in cond_lists: self.__add_cond(cond)

	def del_cond(self, conditions):
		missing_keys = []
		if type(conditions) is list or type(conditions) is tuple or type(conditions) is np.ndarray:
			for cond in conditions:
				res = self.__find_cond_ind(cond)
				if type(res) is int:
					self.conditions.pop(res)
					self.cond_dict.pop(cond)
				else: missing_keys.append(cond)
		else:
			res = self.__find_cond_ind(conditions)
			if type(res) is int:
				self.conditions.pop(res)
				self.cond_dict.pop(conditions)
			else: missing_keys.append(conditions)
		return missing_keys

	def redefine_cond(self, conditions, new_conditions):
		missing_keys = []
		if type(conditions) is list or type(conditions) is tuple or type(conditions) is np.ndarray:
			assert len(conditions) == len(new_conditions)
			for cind in range(len(conditions)):
				res = self.__find_cond_ind(conditions[cind])
				curr_cond = conditions[cind]
				new_cond = new_conditions[cind]
				if type(res) is int:
					self.cond_dict[new_cond] = self.cond_dict.pop(curr_cond)
					self.conditions[res] = new_cond
				else: missing_keys.append(curr_cond)
		else:
			res = self.__find_cond_ind(conditions)
			if type(res) is int:
				self.cond_dict[new_conditions] = self.cond_dict.pop(conditions)
				self.conditions[res] = new_conditions
			else: missing_keys.append(conditions)
		return missing_keys

	def change_cond_masks(self, conditions, masks):
		masks = np.array(masks, dtype = bool)
		assert masks.shape[-1] == len(self.cond_dim), "Masks have the wrong dimension"	
		missing_keys = []
		if type(conditions) is list or type(conditions) is tuple or type(conditions) is np.ndarray:	
			assert len(conditions) == len(masks)
			for cind in range(len(conditions)):
				self.cond_dict[conditions[cind]] = masks[cind]
		else:
			self.cond_dict[conditions] = masks	

	def __getitem__(self, key):
		if type(key) is tuple:
			base_cond = np.ones(len(self.cond_dim), dtype = bool)
			for k in key:
				base_cond = np.multiply(base_cond, self.cond_dict[k])
			return base_cond
		else:
			return self.cond_dict[key]

	def __str__(self):
		repr_str = "Cond object with " + str(len(self.cond_dict.keys())) + " conds"
		for cond in self.conditions:
			repr_str += "\n\t - " + str(cond)
		return repr_str

	def __initialize(self, cond_lists, identifier, cond_dim, cond_dict):
		if cond_dim is not None and cond_dict is not None: 
			self.cond_lists = cond_lists
			self.cond_dim = cond_dim
			self.cond_dict = cond_dict
			return
		cond_lists = np.array(cond_lists)
		identifier = np.array(identifier)
		self.conditions = []
		assert len(cond_lists.shape) < 3, "cond_lists can have at most two dimensions"
		assert len(identifier.shape) == 1, "identifier must be one dimensional"
		self.cond_dim = Dim("Conditions", identifier)
		self.cond_dict = dict({})
		if len(cond_lists.shape) == 1:
			assert identifier.shape[0] == cond_lists.shape[0], "Mismatch in dimension between condition and identifier"
			self.__add_cond(cond_lists)
		else:
			assert identifier.shape[0] == cond_lists.shape[1], "Mismatch in dimension between condition and identifier"
			for cond in cond_lists: self.__add_cond(cond)

	def __add_cond(self, condition):
		conds = np.unique(condition)
		curr_dict = dict({})
		for cond in conds:
			assert cond not in self.cond_dict, "Repeated cond key: " + str(cond)
			cond_mask = condition == cond
			curr_dict.update({cond: cond_mask})
		self.cond_dict.update(curr_dict)
		self.conditions.extend(conds)

	def __find_cond_ind(self, cond):
		for ind, elem in enumerate(self.conditions):
			if elem == cond:
				return ind
		return False

class DFrame:
	def __init__(self, data, dim_names, dim_identifiers, dims = None, dim_dict = None):
		self.data = None		# numpy ndarray
		self.shape = None		# tuple
		self.dim_names = None	# array
		self.dims = None		# array of Dims instances
		
		self.__dim_dict = None	# dictionary Dim names and their positions

		self.__initialize(data, dim_names, dim_identifiers, dims, dim_dict)

	def getdim(self, dim):
		return self.dims[self.__getdim(dim)]

	def slice(self, dim, key, ktype = "identity", squeeze = True, disable_warning = False):
		dim_ind = self.__getdim(dim)
		arr_ind, new_dim = self.getdim(dim_ind).slice(key, ktype, disable_warning)
		if squeeze == True:
			if len(arr_ind) == 1:
				new_data = np.squeeze(np.take(self.data, arr_ind, axis = dim_ind), axis = dim_ind)
				new_dim_names = np.delete(self.dim_names, dim_ind)
				new_dims = []
				for ind in range(len(self.dims)):
					if ind != dim_ind: new_dims.append(self.dims[ind].copy())
				new_dict = self.__dim_dict.copy()
				new_dict.pop(new_dim.name)
				return DFrame(new_data, new_dim_names, None, new_dims, new_dict)
		new_data = np.take(self.data, arr_ind, axis = dim_ind)
		new_dims = []
		for ind in range(len(self.dims)):
			if ind == dim_ind: new_dims.append(new_dim)
			else: new_dims.append(self.dims[ind].copy())
		return DFrame(new_data, self.dim_names.copy(), None, new_dims, self.__dim_dict.copy())

	def swapdims(self, dim1, dim2):
		dim1 = self.__getdim(dim1)
		dim2 = self.__getdim(dim2)
		self.data = np.swapaxes(self.data, dim1, dim2)
		self.shape = self.data.shape
		self.dims[[dim1, dim2]] = self.dims[[dim2, dim1]]
		self.__update_dim_dicts()

	def redefine_dim(self, dim, dim_name, identifier):
		dim_ind = self.__getdim(dim)
		assert len(identifier) == self.shape[dim_ind], "Incorrect length for the new identifier"
		new_dim = Dim(dim_name, identifier)
		new_dim_names = self.dim_names.copy()
		new_dim_names[dim_ind] = dim_name
		new_dims = []
		for ind in range(len(self.dims)):
			if ind == dim_ind: new_dims.append(new_dim)
			else: new_dims.append(self.dims[ind].copy())
		new_dim_dict = self.__dim_dict.copy()
		new_dim_dict[dim_name] = new_dim_dict.pop(self.dim_names[dim_ind])
		return DFrame(self.data, new_dim_names, None, new_dims, new_dim_dict)

	def copy(self):
		new_dims = []
		for dim in self.dims: new_dims.append(dim.copy())
		new_dim_dict = self.__dim_dict.copy()
		return DFrame(self.data.copy(), self.dim_names.copy(), None, new_dims, new_dim_dict)

	def __getitem__(self, key):
		return self.data[key]

	def __str__(self):
		repr_str = "DFrame Object with shape " + str(self.shape)
		for ind, dim in enumerate(self.dims):
			repr_str +=  "\n\t" + "- " + str(ind) + ": " + str(dim)
		return repr_str

	def __initialize(self, data, dim_names, dim_identifiers, dims, dim_dict):
		self.data = np.array(data, dtype = object)
		self.shape = data.shape
		self.dim_names = np.array(dim_names, dtype = str)
		assert len(dim_names) == len(self.shape), "Incorrect number of dimensions specified"
		if dims is not None and dim_dict is not None:
			self.dims = dims
			self.__dim_dict = dim_dict
		else:
			assert len(dim_names) == len(dim_identifiers), "Mismatch between dim_names and dim_identifiers"
			dims = np.empty(len(dim_names), dtype = object)
			for dind in range(len(dim_names)):
				assert len(dim_identifiers[dind]) == self.shape[dind], "Identifiers of dim " + str(dim_names[dind] + " mismatches with its corresponding dimension")
				dims[dind] = Dim(dim_names[dind], dim_identifiers[dind])
			self.dims = dims
			self.__update_dim_dicts()

	def __update_dim_dicts(self):
		self.__dim_dict = dict({})
		for dind in range(len(self.dims)): self.__dim_dict.update({self.dims[dind].name : dind})

	def __getdim(self, dim):
		assert type(dim) is str or type(dim) is int, "invalid dim type; must be either str or int"
		if type(dim) is str: return self.__dim_dict[dim]
		else: return dim
