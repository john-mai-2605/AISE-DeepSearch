def deep_search(input, model, group_size, batch_size, input_is multi_ch = True, verbose = False):
	input_dimension = len(input.shape)
	input_channels = input.shape[-1]
	# Assume last dimension is channels
	if input_is_multi_ch:
		input_dimension -= 1
	
	
	grouping = create_grouping(input.shape, group_size, input_is multi_ch = True)
	