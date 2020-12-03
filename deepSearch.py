from mutation import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
import time


def deepSearch(image, model, distortion_cap, group_size= 16, max_calls = 10000, batch_size = 64, verbose = False):
	"""
	"""
	# You may skip initial part
	e = Evaluator(model, max_calls)
	original_probability = e.evaluate(image)
	original_class = np.argmax(original_probability)
	print("Original class: {}".format(original_class))
	current_class_prob = original_probability[original_class]

	# Used for verbose display
	s_max_calls = str(max_calls)
	new_score = 1
	
	# Same as relative_evaluation(image, original_class)
	# For more information, look up in evaluation.py
	# This way saves one evaluation_calls
	relative_score = current_class_prob - original_probability
	relative_score[original_class] = 1
	
	lower, upper = create_boundary_palette(image, distortion_cap)
	img_size = np.shape(image)[:2]
	
	# Algorithm 2: line 5
	rel_eval = lambda image : e.evaluate(image)[original_class]
	#rel_eval = lambda image : e.relative_evaluate(image, original_class)
	
	# Initialize before loop
	current_class = original_class
	# Push image to lower bound
	grouping = group_generation(img_size, group_size, options = "square")
	current_image = image_mutate(image, grouping, lower, lower)
	while original_class==current_class and e.evaluation_count < max_calls:
		# Algorithm 2: line 2 
		grouping = group_generation(img_size, group_size, options = "square")
		
		regroup = False 
		
		# Main algorithm starts here
		while original_class==current_class and e.evaluation_count < max_calls and not regroup:
			if verbose:
				print("Call count(can overshoot)\t"+str(e.evaluation_count)+"/"+s_max_calls + " Score: {:.4f}".format(new_score), end = "\r")
			# Line 7
			target_class = np.argmin(relative_score)
			# Line 8
			mutated_image, new_score = approx_min(current_image, lower, upper, rel_eval, grouping, batch_size, target_class)
			# If nothing changed, change the grouping
			if True:#np.product(current_image == mutated_image):
				regroup = True
				group_size = group_size//2
				print("\nGroup size: {}".format(group_size))
			current_image = mutated_image
			current_class = np.argmax(e.evaluate(current_image))
	if verbose:
		print()

	success = not current_class == original_class
	print("Current class: {}".format(current_class))
	counts = e.evaluation_count
	return (success, current_image, counts)
		
			

def approx_min(image, lower, upper, rel_eval, grouping, batch_size, target_class):
	number_of_groups = len(grouping)
	ch = 0
	is_color = len(np.shape(image)) == 3

	base_score = rel_eval(image)
	
	# Initialize direction_array
	# This array keeps all the decision
	if is_color:
		direction_array = np.zeros((number_of_groups,3), dtype = bool)
	else:
		direction_array = np.zeros(number_of_groups, dtype = bool)
	
	# Reserved in case the final result is less robust than one of the exploratory steps
	da_keep = direction_array
	minimum = 1
	# group_index, ch, direction, da
	minimum_record = (0, 0, False, direction_array)
	batch_count = 0
	# This loop is ApproxMin
	for group_number in np.random.permutation(number_of_groups * 3):
		if is_color:
			group_index = group_number // 3
			ch = group_number % 3
			#started_time = time.process_time()
			upper_exploratory, u_mutated = single_mutate(image, group_index, grouping, lower, upper, direction = True, channel = ch)
			lower_exploratory, l_mutated = single_mutate(image, group_index, grouping, lower, upper, direction = False, channel = ch)
			if u_mutated:
				upper_score = rel_eval(upper_exploratory)
			else:
				upper_score = base_score
			if l_mutated:
				lower_score = rel_eval(lower_exploratory)
			else:
				lower_score = base_score				
			u_target_score = np.min(upper_score)
			l_target_score = np.min(lower_score)
			# If adversarial input is found during exploration,
			# Stop there
			if u_target_score < 0:
				return(upper_exploratory, np.min(upper_score))
			if l_target_score < 0:
				return(lower_exploratory, np.min(lower_score))
			dir = u_target_score < l_target_score
			direction_array[group_index,ch] = dir
			if min((u_target_score,l_target_score))<minimum:
				minimum_record = (group_index, ch, dir, da_keep)
				minimum = min((u_target_score,l_target_score))
			batch_count += 1
			if batch_count == batch_size:
				da_keep = direction_array.copy()
				batch_count = 0
				image = image_mutate(image, grouping, lower,upper, da_keep)
				base_score = rel_eval(image)
		else:# single channel (grayscale)
			upper_exploratory = single_mutate(image, group_index, grouping, lower, upper, direction = True)
			lower_exploratory = single_mutate(image, group_index, grouping, lower, upper, direction = False)
			upper_score = rel_eval(upper_exploratory)
			lower_score = rel_eval(lower_exploratory)
			u_target_score = upper_score[target_class]
			l_target_score = lower_score[target_class]
			# If adversarial input is found during exploration,
			# Stop there
			if u_target_score < 0:
				return(upper_exploratory, np.min(upper_score))
			if l_target_score < 0:
				return(lower_exploratory, np.min(lower_score))
			dir = u_target_score < l_target_score
			direction_array[group_index] = dir
			if min((u_target_score,l_target_score))<minimum:
				minimum_record = (group_index, ch, dir)
				minimum = min((u_target_score,l_target_score))
	mutated_image = image_mutate(image, grouping, lower, upper, direction_array)
	total_rel_score = rel_eval(mutated_image)
	if(minimum < np.min(total_rel_score)):
		group_index, ch, direction, da =minimum_record
		da[group_index,ch] = direction
		mutated_image = image_mutate(image, grouping, lower, upper, da)
	return(mutated_image, np.min((minimum,np.min(total_rel_score))))

			
if __name__ == "__main__":
	g_test_image = np.reshape((np.arange(30) + 0.5 ) / 31, (6, 5))
	success, counts, g_res = deepSearch(g_test_image, model = None, distortion_cap = 0.1, group_size = 2, max_calls = 10000, verbose = True)
	magic_number = 150*150*3
	test_image = np.reshape((np.arange(magic_number) + 0.5 ) / magic_number, (150, 150, 3))
	success, counts, res = deepSearch(test_image, model = None, distortion_cap = 10/255, group_size = 16, max_calls = 10000, verbose = True)
	plt.subplot(121)
	plt.imshow(g_res,"gray",vmin = 0, vmax = 1)
	plt.subplot(122)
	plt.imshow(res, vmin = 0, vmax = 1)
	plt.show()