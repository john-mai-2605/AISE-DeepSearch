from mutation import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt

def deepSearch(image, model, distortion_cap, group_size= 16, max_calls = 10000):
	"""
	"""
	e = Evaluator(model)
	original_probability = e.evaluate(image)
	original_class = np.argmax(original_probability)
	current_class_prob = original_probability[original_class]
	
	# Same as relative_evaluation(image, original_class)
	# For more information, look up in evaluation.py
	# This way saves one evaluation_calls
	relative_score = current_class_prob - original_probability
	relative_score[original_class] = 1
	
	lower, upper = create_boundary_palette(image, distortion_cap)
	img_size = np.shape(image)[:2]
	
	# Algorithm 2: line 5
	rel_eval = lambda image : e.relative_evaluate(image, original_class)
	
	# Initialize before loop
	current_class = original_class
	current_image = image
	
	while True and e.evaluation_count < max_calls:
		# Algorithm 2: line 2 
		grouping = group_generation(img_size, group_size, options = "square")
		regroup = False 
		
		# Main algorithm starts here
		while True and e.evaluation_count < max_calls and not regroup:
			# Line 7
			target_class = np.argmin(relative_score)
			# Line 8
			mutated_image = approx_min(current_image, lower, upper, rel_eval, grouping, target_class)
			
			current_image = mutated_image
			current_class = np.argmax(e.evaluate(current_image))
			print(e.evaluation_count)
	success = current_class == original_class
	counts = e.evaluation_count
	return (success, counts, current_image)
		
			
def approx_min(image, lower, upper, rel_eval, grouping, target_class):
	number_of_groups = len(grouping)
	
	# Initialize direction_array
	# This array keeps all the decision
	direction_array = np.zeros(number_of_groups, dtype = bool)
	
	# This loop is ApproxMin
	for group_index in range(number_of_groups):
		# single channel (grayscale)
		upper_exploratory = single_mutate(image, group_index, grouping, lower, upper, direction = True)
		lower_exploratory = single_mutate(image, group_index, grouping, lower, upper, direction = False)
		upper_score = rel_eval(upper_exploratory)[target_class]
		lower_score = rel_eval(lower_exploratory)[target_class]
		direction_array[group_index] = upper_score > lower_score
	mutated_image = image_mutate(image, grouping, lower, upper, direction_array)
	return(mutated_image)
			
if __name__ == "__main__":
	g_test_image = np.reshape((np.arange(30) + 0.5 ) / 31, (6, 5))
	success, counts, res = deepSearch(g_test_image, model = None, distortion_cap = 0.1, group_size = 2, max_calls = 10000)
	print(res)
	plt.imshow(res,"gray",vmin = 0, vmax = 1)
	plt.show()