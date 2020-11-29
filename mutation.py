import numpy as np
import matplotlib.pyplot as plt

def group_generation(size = (3,3), group_size = 2, options = ""):
	"""
	size: Size of the image to be divided into groups.
	group_size: Width of group if the group was square.
	options: Reserved parameter in case of other grouping patterns.
	
	This method outputs  a list with groups of indices. For example,
	 _______________
	|0  |1  |  2|  3|	If this square was grouped into 4 pixels,
	|___|___|___|___|	[[0, 1, 4, 5],
	|4  |5  |  6|  7|	 [2, 3, 6, 7],
	|___|___|___|___|	 [8, 9, 12, 13],
	|   |   |   |   |	 [10, 11, 14, 15]]
	|8__|9__|_10|_11|	will be the outcome.
	|   |   |   |   |	
	|12_|13_|_14|_15|	
	
	If the groups cannot be sized equally, the rightmost and bottom
	groups will be cropped."""
	"""
	idea behind the implementation is:
	1. Fill the pixels with sequential indexes
	2. Slice and return the slices.
	
	Channel invariant
	"""
	size_y, size_x = size
	
	# trivial case
	if group_size < 1:
		return[[i] for i in range(size_x*size_y)]
		
	if options =="" or options.lower() == "square":
		
		# the "+ group_size -1" is for ceiling to integer
		group_number_x = (size_x + group_size - 1) // group_size
		group_number_y = (size_y + group_size - 1) // group_size
		
		pre_cut = np.reshape(np.arange(size_x*size_y),(size_y, size_x))
		
		# j*gs is starting index_x of a group
		# i*gs is starting index_y of a group
		# To maintain x as horizontal, it is secondary index.
		gs = group_size
		return [np.reshape(pre_cut[i*gs:(i+1)*gs , j*gs:(j+1)*gs],-1) for i in range(group_number_y) for j in range(group_number_x)]
		#breakdown of above^
		# 1. for i in range(group_number_x) 
		#		for j in range(group_number_y)
		#			pre_cut[i*gs:(i+1)*gs , j*gs:(j+1)*gs]
		# 2. reshaped into single dimensional array
	else: # no option match found
		print("[group_generation]Unavailable option: ", end = str(option) + "\n")
		return[[i] for i in range(size_x*size_y)]
	
def create_boundary_palette(image, distortion_cap):
	"""
	image: image in floating point 0~1 values
	distortion_cap: maximum distortion value in floating point 0~1
	
	lower, upper = create_boundary_palette(image, distortion_cap)
	returns image that has all lower and all upper boundary values.
	If the modified value exceeds the range[0,1], it is clipped.
	
	Channel invariant
	"""
	lower = image - distortion_cap
	upper = image + distortion_cap
	lower[lower < 0] = 0
	upper[upper > 1] = 1
	return(lower,upper)
	
def single_mutate(image, group_index, grouping_scheme, lower, upper, direction = True, channel = 0):
	"""
	image: Targetting image to mutate
	group_index: Used for instructing which group to mutate
	grouping_scheme: list of grouped indices of pixels
		for example, a first 3 by 3 group will be:
		[0, 		1, 			2,
		 w, 		w+1, 		w+2,
		 w*2, 		(w*2)+1,	(w*2)+2]
		(w is width of image)
		grouping_scheme is a list of these.
	lower, upper: each are images that is entirely upper bounded or lower bounded
	direction: True for upper, False for lower
		(later changed to to_upper for readability)
	channel: For color images. 0 for red, 1 for green, 2 for blue.
	
	A new patch(group) is taken from either upper or lower image and is replaced of the original image.

	mutated_image, mutated = single_mutated()

	If the image is not mutated since it's already at the
	boundary for the specified group, it will output the
	original image and 'False' (bool) for 'mutated'.
	"""
	to_upper = direction
	is_color = len(np.shape(image)) == 3
	original_shape = np.shape(image)[:2]
	
	# Channel separation
	if is_color:
		mutated = np.copy(image[:,:,channel])
	# Single channel
	else:
		mutated = np.copy(image)
	
	# Flatten to a single row to use group indices
	mutated = np.reshape(mutated,-1)
	if to_upper:
		alternative_image = np.reshape(upper, -1)
	else:
		alternative_image = np.reshape(lower, -1)
	# Get the pixel indices to replace
	replacing_group_indices = grouping_scheme[group_index]
	
	# Channel remerge
	if is_color:
		if np.product(mutated[replacing_group_indices] == alternative_image[replacing_group_indices*3 + channel]):
			return (image, False)
		mutated[replacing_group_indices] = alternative_image[replacing_group_indices*3 + channel]
		temp = np.copy(image)
		temp[:,:,channel] = np.reshape(mutated, original_shape)
		return (temp, True)

	# Replace a part of original image to an alternate one.		
	mutated[replacing_group_indices] = alternative_image[replacing_group_indices]
	return (np.reshape(mutated, original_shape), True)

def image_mutate(image, grouping_scheme, lower, upper,  direction_array = None):
	"""
	image: targetting image
	direction_array: An array(or list) that has all the directions for all the groups.
		For color images with three channels direction_array is shaped as [:][3] for each channels.
	grouping_scheme: list of group indices created by group_generation()
	This method mutates every group(pixel) of an image accordingly
	"""
	# If no direction is give, all goes to upper
	is_color = len(np.shape(image)) == 3
	image = np.copy(image)
	original_shape = np.shape(image)
	
	# direction_array was not specified
	if not type(direction_array) == np.ndarray:
		if is_color:
			direction_array = np.array([[True,True,True] for i in range(len(grouping_scheme))])
		else:
			direction_array = [True for i in range(len(grouping_scheme))]
	
	# Converting direction_array from boolean to 0,1 integer
	# in order to avoid if statement and use indexing
	direction_binary = np.zeros(np.shape(direction_array), dtype = int)
	# False is lower, True is upper
	direction_binary[direction_array] = 1
	# bound[0] is lower, bound[1] is upper
	bounds = np.stack((lower,upper))
	
	
	# To understand how this works, read grayscale part first
	if is_color: 
		# Channel separation
		img_r = np.reshape(image[:,:,0],-1)
		img_g = np.reshape(image[:,:,1],-1)
		img_b = np.reshape(image[:,:,2],-1)
		direction_r = direction_binary[:,0]
		direction_g = direction_binary[:,1]
		direction_b = direction_binary[:,2]
		bound_r = np.reshape(bounds[:,:,:,0],(2,-1))
		bound_g = np.reshape(bounds[:,:,:,1],(2,-1))
		bound_b = np.reshape(bounds[:,:,:,2],(2,-1))
		for group_number in range(len(direction_binary)):
			group_indices = grouping_scheme[group_number]
			dir_r = direction_r[group_number]
			dir_g = direction_g[group_number]
			dir_b = direction_b[group_number]
			img_r[group_indices] = bound_r[dir_r][group_indices]
			img_g[group_indices] = bound_g[dir_g][group_indices]
			img_b[group_indices] = bound_b[dir_b][group_indices]
		# Channel remerging
		image[:,:,0] = np.reshape(img_r, original_shape[:2])
		image[:,:,1] = np.reshape(img_g, original_shape[:2])
		image[:,:,2] = np.reshape(img_b, original_shape[:2])
	else: #Grayscale
		# Flattening image
		image = np.reshape(image,-1)
		bounds = np.reshape(bounds,(2,-1))
		for group_number in range(len(direction_binary)):
			group_indices = grouping_scheme[group_number]
			dir = direction_binary[group_number]
			image[group_indices] = bounds[dir][group_indices]
		image = np.reshape(image,original_shape)
	return image
			
if __name__ == "__main__":
	grouping = group_generation((6, 5), 2)
	
	# Color test
	test_image = np.reshape((np.arange(90) + 0.5 ) / 91, (6, 5, 3))
	lower, upper = create_boundary_palette(test_image, 0.1)
	single_mutated = single_mutate(test_image, 0, grouping, lower, upper)
	# Yellow, Blue, Red
	# Green, Magenta, Cyan
	# Red, Green, Blue
	print("Yellow, Blue, Red\nGreen, Magenta, Cyan\nRed, Green, Blue\n")
	color_directions = np.array([[True,True, False],[False,False,True], [True,False,False],[False, True,False],[True,False,True],[False,True,True],[True,False,False],[False,True,False],[False,False,True]])
	image_mutated = image_mutate(test_image, grouping, lower, upper, color_directions)
	
	plt.subplot(231)
	plt.imshow(lower, vmin=0, vmax=1)
	plt.subplot(232)
	plt.imshow(test_image, vmin=0, vmax=1)
	plt.subplot(233)
	plt.imshow(upper, vmin=0, vmax=1)
	plt.subplot(235)
	plt.imshow(single_mutated, vmin=0, vmax=1)
	plt.subplot(236)
	plt.imshow(image_mutated, vmin = 0, vmax = 1)
	plt.show()
	
	# Grayscale test
	g_test_image = np.reshape((np.arange(30) + 0.5 ) / 31, (6, 5))
	g_lower, g_upper = create_boundary_palette(g_test_image, 0.1)
	g_single_mutated = single_mutate(g_test_image, 0, grouping, lower, upper)
	# 1 0 1
	# 0 1 0
	# 1 0 0
	print("101\n010\n100\n")
	g_directions = np.array([True,False,True,False,True,False,True,False,False])
	g_image_mutated = image_mutate(g_test_image, grouping, g_lower, g_upper, g_directions)
	
	plt.subplot(231)
	plt.imshow(g_lower, "gray", vmin=0, vmax=1)
	plt.subplot(232)
	plt.imshow(g_test_image, "gray", vmin=0, vmax=1)
	plt.subplot(233)
	plt.imshow(g_upper, "gray", vmin=0, vmax=1)
	plt.subplot(235)
	plt.imshow(g_single_mutated, "gray", vmin=0, vmax=1)
	plt.subplot(236)
	plt.imshow(g_image_mutated, "gray", vmin = 0, vmax = 1)
	plt.show()
	