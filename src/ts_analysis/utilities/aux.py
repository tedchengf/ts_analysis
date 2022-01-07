# aux.py

import numpy as np

def check_duplicate(arr):
	item_dict = dict({})
	dup_dict = dict({})
	for ind, item in enumerate(arr):
		if item in item_dict:
			if item not in dup_dict:
				dup_dict.update({item: [item_dict[item], ind]})
			else:
				dup_dict[item].append(ind)
		else:
			item_dict.update({item: ind})
	return dup_dict

def slice_handler(slice_obj, arr_len):
	start = slice_obj.start
	if start is None: start = 0
	if start < 0: start = arr_len + start
	stop = slice_obj.stop
	if stop is None: stop = arr_len
	if stop < 0: stop = arr_len + stop
	step = slice_obj.step
	return np.arange(start, stop, step)

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
