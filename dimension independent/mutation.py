import numpy as np

def create_grouping(input_data, group_size, input_is_multi_ch = True):
	input_shape = input_data.shape
	if input_is_multi_ch:
		input_shape = input_shape[0:-1]
	
	input_volume = np.product(input_shape)
		
	if group_size < 1:
		return[[i] for i in range(input_volume)]
	
	group_numbers = [(length + group_size -1) // group_size 
		for length in input_shape]
	
	pre_cut = np.reshape(np.arange(input_volume),input_shape)
	
	for group_index in np.product(group_numbers):
		