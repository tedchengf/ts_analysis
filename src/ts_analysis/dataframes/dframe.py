# DataFrmae

import numpy as np
from ts_analysis.utilities import aux
import warnings

def union(identifier_1, identifier_2):
	return np.array(list(set(identifier_1) | set(identifier_2)))

def intersection(identifier_1, identifier_2):
	return np.array(list(set(identifier_1) & set(identifier_2)))

def exclude(identifier, exclude_list):
	exclude_set = set(exclude_list)
	final_list = []
	for idt in identifier:
		if idt not in exclude_list: final_list.append(idt)
	return np.array(final_list)

class Dim:
	def __init__(self, name, identifier, identity_dict = None):
		assert type(name) is str, "the name must be a string"
		self.name = name     					# str
		self.identifier = np.array(identifier)	# array

		self.__identity_dict = None

		self.__initialize(name, identifier, identity_dict)

	def slice(self, key, ktype = "identity", loose_match = False, return_self = True):
		return self.__slice(key, ktype, loose_match, return_self)

	def exclude(self, key, ktype = "identity", return_self = True):
		key = self.__key_handler(key, ktype)
		if ktype == "index": exclude_list = self.identifier[key]
		else: exclude_list = key
		select_key = exclude(self.identifier, exclude_list)
		return self.__slice(select_key, "identity", False, return_self)

	def check_identifiers(self, keys):
		keys = self.__key_handler(keys, ktype = "identity")
		arr_ind, missing_keys = aux.dict_arr_query(keys,self.__identity_dict)
		return self.identifier[arr_ind], missing_keys

	def redefine(self, name = None, identifier = None):
		if name is not None: self.name = name
		if identifier is not None:
			new_identifier = np.array(identifier)
			identifier_shape = self.identifier.shape
			if new_identifier.shape != identifier_shape:
				raise RuntimeError("the shape of the new identifier does not match the one that have been previously defined")
			new_identity_dict = self.__rebuild_identity_dict(identifier)
			self.identifier = new_identifier
			self.__identity_dict = new_identity_dict

	def equal(self, other):
		if isinstance(other, Dim):
			return other.name == self.name and np.array_equal(other.identifier, self.identifier)
		return False

	def copy(self):
		return Dim(self.name,self.identifier.copy(),self.__identity_dict.copy())

	def wrap(self, data_obj):
		return WrapperDim(data_obj, self.name, self.identifier.copy(), self.__identity_dict.copy())

	def get_identity_dict(self):
		return self.__identity_dict

	def __len__(self):
		return len(self.identifier)

	def __str__(self):
		return "Dim '" + self.name + "' with " + str(len(self)) + " identifiers as " + str(self.identifier.dtype)

	def __getitem__(self, key):
		return self.__slice(key, ktype = "index", loose_match = False)

	def __iter__(self):
		return self.identifier.__iter__()

	def __lt__(self, other): return self.identifier < other
	def __le__(self, other): return self.identifier <= other
	def __eq__(self, other): return self.identifier == other
	def __ne__(self, other): return self.identifier != other
	def __gt__(self, other): return self.identifier > other
	def __ge__(self, other): return self.identifier >= other

	def __initialize(self, name, identifier, identity_dict):
		if identity_dict is not None:
			self.__identity_dict = identity_dict
		else:
			self.__identity_dict = self.__rebuild_identity_dict(identifier)	

	def __slice(self, key, ktype, loose_match, return_self):
		key = self.__key_handler(key, ktype)
		if ktype == "index":
			arr_ind = np.arange(len(self))[key]
		else:			
			arr_ind, missing_keys = aux.dict_arr_query(key,self.__identity_dict)
			if len(missing_keys) > 0:
				if loose_match == False:
					raise KeyError("The following identifiers are undefined: " + str(missing_keys))
		if return_self == True:
			return arr_ind, Dim(self.name, self.identifier[arr_ind])
		else:
			return arr_ind

	def __rebuild_identity_dict(self, identifier):
		identity_dict = dict({})
		repeated_identifier = []
		for ind, curr_identi in enumerate(identifier):
			if curr_identi in identity_dict: 
				repeated_identifier.append(curr_identi)
			identity_dict.update({curr_identi:ind})
		if len(repeated_identifier) > 0:
			raise KeyError("The following identifiers are repeated: " + str(repeated_identifier))
		return identity_dict

	def __key_handler(self, key, ktype):
		if ktype not in ("index", "identity"):
			raise KeyError("The parameter ktype must be one from (index, identity)")
		# if ktype == "identity":
		# 	if key is None: return self.identifier
		# 	if type(key) is list or type(key) is np.ndarray: return key
		# 	if type(key) is slice:
		# 		if key.start is None and key.stop is None and key.step is None:
		# 			return self.identifier
		# 		raise KeyError("ktype identity does not support slice indexing")
		# 	else: return [key]
		# else:
		# 	if key is None: return np.arange(len(self.identifier),dtype = int)
		# 	if type(key) is list or type(key) is np.ndarray: return key
		# 	if type(key) is slice: return aux.slice_handler(key, len(self.identifier))
		# 	else: return [key]
		if ktype == "identity":
			if key is None: return self.identifier
			if type(key) is slice:
				if key.start is None and key.stop is None and key.step is None:
					return self.identifier
				raise KeyError("ktype identity does not support slice indexing")
			if type(key) is str or type(key) is np.str_:
				return [key]
			try:
				iterator = iter(key)
				return key
			except:
				return [key]
		else:
			if key is None: return np.arange(len(self.identifier),dtype = int)
			if type(key) is slice: return aux.slice_handler(key, len(self.identifier))
			if type(key) is str or type(key) is np.str_:
				return [key]
			try:
				return list(iter(key))
			except:
				return [key]

class WrapperDim(Dim):
	def __init__(self, data_obj, name, identifier, identity_dict = None):
		super().__init__(name, identifier, identity_dict)
		if type(data_obj) is np.ndarray or type(data_obj) is list:
			data_obj = np.array(data_obj)
			if len(data_obj) != len(identifier): raise RuntimeError("The first dimension of the data_obj does not match the length of the identifier")
		elif type(data_obj) is DFrame: 
			obj_dim = data_obj.getdim_obj(name)
			if len(obj_dim.identifier) != len(identifier):
				raise RuntimeError("the dimension of the identifier does not match the corresponding dimension in DFrame")
		else: raise TypeError("invalid type for data_obj")
		self.data_obj = data_obj

	def slice(self, key, ktype = "identity", loose_match = False, return_self = False):
		res = super().slice(key, ktype, loose_match, return_self)
		return self.__slice_res_handler(res, loose_match, return_self)

	def exclude(self, key, ktype = "identity", return_self = False):
		res = super().exclude(key, ktype, return_self)
		return self.__slice_res_handler(res, False, return_self)

	def copy(self):
		return WrapperDim(self.data_obj, self.name,self.identifier.copy(),self.__identity_dict.copy())

	def __str__(self):
		return "WrapperDim '" + self.name + "' with " + str(len(self)) + " identifiers as " + str(self.identifier.dtype)

	def __getitem__(self, key):
		return self.slice(key, return_self = False)

	def __lt__(self, other): 
		return self.slice(self.identifier < other, ktype = "index")
	def __le__(self, other): 
		return self.slice(self.identifier <= other, ktype = "index")
	def __eq__(self, other): 
		return self.slice(self.identifier == other, ktype = "index")
	def __ne__(self, other): 
		return self.slice(self.identifier != other, ktype = "index")
	def __gt__(self, other): 
		return self.slice(self.identifier > other, ktype = "index")
	def __ge__(self, other): 
		return self.slice(self.identifier >= other, ktype = "index")

	def __slice_res_handler(self, res, loose_match, return_self):
		if return_self == True:
			arr_ind, new_dim = res
			new_data_obj = self.__data_obj_handler(arr_ind)
			return new_dim.wrap(new_data_obj)
		else:
			arr_ind = res
			new_data_obj = self.__data_obj_handler(arr_ind)
			return new_data_obj

	def __data_obj_handler(self, arr_ind):
		if type(self.data_obj) is DFrame:
			return self.data_obj.slice(arr_ind, dim = self.name, ktype="index")
		else:
			return self.data_obj[arr_ind]			

class DFrame:
	def __init__(self, data, dim_names, dim_identifiers, dims = None, dim_dict = None, dtype = object):
		self.data = None		# numpy ndarray
		self.shape = None		# tuple
		self.dim_names = None	# array
		
		self.__dims = None		# array of Dims instances
		self.__dim_dict = None	# dictionary Dim names and their positions

		self.__initialize(data, dim_names, dim_identifiers, dims, dim_dict, dtype)

	def getdim(self, dim):
		return self.__dims[self.__dim_ind(dim)].wrap(self)

	def getdim_obj(self, dim):
		return self.__dims[self.__dim_ind(dim)]

	def slice(self, key, dim = None, ktype = "identity", squeeze = True, loose_match = False):
		if dim is None: dim_ind = 0
		else: dim_ind = self.__dim_ind(dim)
		arr_ind, new_dim = self.__dims[dim_ind].slice(key,ktype,loose_match)
		return self.__slice(dim_ind, arr_ind, new_dim, squeeze)

	def exclude(self, key, dim = None, ktype = "identity", squeeze = True):
		if dim is None: dim_ind = 0
		else: dim_ind = self.__dim_ind(dim)
		arr_ind, new_dim = self.__dims[dim_ind].exclude(key, ktype)
		return self.__slice(dim_ind, arr_ind, new_dim, squeeze)

	def swapdims(self, dim1, dim2):
		dim1 = self.__dim_ind(dim1)
		dim2 = self.__dim_ind(dim2)
		if dim1 != dim2:
			self.data = np.swapaxes(self.data, dim1, dim2)
			self.shape = self.data.shape
			dim_obj_1 = self.__dims[dim1]
			dim_obj_2 = self.__dims[dim2]
			self.__dims[dim1] = dim_obj_2
			self.__dims[dim2] = dim_obj_1
			# self.__dims[[dim1, dim2]] = self.__dims[[dim2, dim1]]
			self.__dim_dict = self.__rebuild_dim_dict(self.__dims)
		return self

	def redefine_dim(self, dim, name = None, identifier = None):
		dim_ind = self.__dim_ind(dim)
		if name is None: name = self.__dims[dim_ind].name
		if identifier is None: identifier = self.__dims[dim_ind].identifier
		assert len(identifier) == self.shape[dim_ind], "Incorrect length for the new identifier"
		new_dim = Dim(name, identifier)
		self.__dims[dim_ind] = new_dim
		self.__dim_dict[name] = self.__dim_dict.pop(self.dim_names[dim_ind])
		self.dim_names[dim_ind] = name
		return self

	def copy(self):
		new_dims = []
		for dim in self.__dims: new_dims.append(dim.copy())
		new_dim_dict = self.__dim_dict.copy()
		return DFrame(self.data.copy(), self.dim_names.copy(), None, new_dims, new_dim_dict)

	def __getitem__(self, key):
		if key is None: return self
		if type(key) is tuple:
			return_DFrame = self.copy()
			for key_ind, curr_slice in enumerate(key):
				curr_dim_name = self.__dims[key_ind].name
				return_DFrame = return_DFrame.slice(curr_slice, dim=curr_dim_name, ktype = "index")
			return return_DFrame
		else: return self.slice(key, ktype = "index")

	def __getattr__(self, name):
		if name not in self.__dim_dict: raise AttributeError
		return self.getdim(name)

	def __getstate__(self):
		return dict({"data": self.data, "shape": self.shape, "dim_names": self.dim_names,"__dims": self.__dims, "__dim_dict": self.__dim_dict})

	def __setstate__(self, state_dict):
		self.data = np.array(state_dict["data"])
		self.shape = state_dict["shape"]
		self.dim_names = state_dict["dim_names"]
		self.__dims = state_dict["__dims"]
		self.__dim_dict = state_dict["__dim_dict"]

	def __len__(self):
		return self.shape[0]

	def __iter__(self):
		proxy_dim = []
		for dim in self.__dims: proxy_dim.append(dim.wrap(self))
		return proxy_dim.__iter__()

	def __str__(self):
		repr_str = "DFrame Object with shape " + str(self.shape)
		for ind, dim in enumerate(self):
			repr_str +=  "\n\t" + "- " + str(ind) + ": " + str(dim)
		return repr_str

	def __initialize(self, data, dim_names, dim_identifiers, dims, dim_dict, dtype):
		if type(data) is np.ndarray: self.data = data
		else:
			self.data = np.array(data, dtype = dtype)
		self.shape = self.data.shape
		self.dim_names = np.array(dim_names, dtype = str)
		assert len(dim_names) == len(self.shape), "Incorrect number of dimensions specified"
		if dims is not None and dim_dict is not None:
			self.__dims = dims
			self.__dim_dict = dim_dict
		else:
			assert len(dim_names) == len(dim_identifiers), "Mismatch between dim_names and dim_identifiers"
			dims = np.empty(len(dim_names), dtype = object)
			for dind in range(len(dim_names)):
				assert len(dim_identifiers[dind]) == self.shape[dind], "Identifiers of dim " + str(dim_names[dind] + " mismatches with its corresponding dimension")
				dims[dind] = Dim(dim_names[dind], dim_identifiers[dind])
			self.__dims = dims
			self.__dim_dict = self.__rebuild_dim_dict(self.__dims)

	def __rebuild_dim_dict(self, dims):
		dim_dict = dict({})
		for dind in range(len(dims)): dim_dict.update({dims[dind].name: dind})
		return dim_dict

	def __dim_ind(self, dim):
		if type(dim) is not str and type(dim) is not int:
			raise TypeError("invalid dim type; must be either str or int")
		if type(dim) is str: return self.__dim_dict[dim]
		else: return dim

	def __slice(self, dim_ind, arr_ind, new_dim, squeeze):
		if squeeze == True:
			if len(arr_ind) == 1:
				new_data = np.squeeze(np.take(self.data, arr_ind, axis = dim_ind), axis = dim_ind)
				# Return data if there is only one elem in the data
				if len(new_data.shape) == 0: return new_data.item()
				new_dim_names = np.delete(self.dim_names, dim_ind)
				new_dims = []
				for ind in range(len(self.__dims)):
					if ind != dim_ind: new_dims.append(self.__dims[ind].copy())
				new_dict = self.__rebuild_dim_dict(new_dims)	
				return DFrame(new_data, new_dim_names, None, new_dims, new_dict)
		new_data = np.take(self.data, arr_ind, axis = dim_ind)
		new_dims = []
		for ind in range(len(self.__dims)):
			if ind == dim_ind: new_dims.append(new_dim)
			else: new_dims.append(self.__dims[ind].copy())
		return DFrame(new_data, self.dim_names.copy(), None, new_dims, self.__dim_dict.copy())

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
