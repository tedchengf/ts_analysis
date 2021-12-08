# aux.py

import numpy as np

def dict_arr_query(keys, dictionary):
	values = []
	missing_keys = []
	for k in keys:
		try:
			values.append(dictionary[k])
		except KeyError:
			missing_keys.append(k)
	return np.array(values), np.array(missing_keys)

def subset_dict(keys, sup_dict):
	sub_dict = dict({})
	missing_keys = []
	for k in keys:
		try: 
			sub_dict.update({k:sup_dict[k]})
		except KeyError:
			missing_keys.append(k)
	return sub_dict, missing_keys
