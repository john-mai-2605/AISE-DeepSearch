from mutation import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
import time

def deepSearch(image, label, model, distortion_cap, group_size= 16, max_calls = 10000, batch_size = 64, verbose = False, targeted = False, target = None, proba = True):
	"""
	"""
	# You may skip initial part
	e = Evaluator(model, max_calls)
	original_probability = e.evaluate(image)
	original_class = label
	print("Original class: {}".format(e.idx2name(original_class)))
	current_class_prob = original_probability[original_class]
	
	# Same as relative_evaluation(image, original_class)
	# For more information, look up in evaluation.py
	# This way saves one evaluation_calls
	relative_score = current_class_prob - original_probability
	relative_score[original_class] = 1
	
	lower, upper = create_boundary_palette(image, distortion_cap)
	img_size = np.shape(image)[:2]
	
	# Algorithm 2: line 5
	#rel_eval = lambda image : e.evaluate(image)[original_class]
	rel_eval = lambda image : e.relative_evaluate(image, original_class, proba)
	
	# Initialize before loop
	current_class = original_class
	# Push image to lower bound
	grouping = group_generation(img_size, group_size, options = "square")
	current_image = image_mutate(image, grouping, lower, lower)

	if not targeted:
		while original_class==current_class and e.evaluation_count < max_calls:
			# Algorithm 2: line 2 
			grouping = group_generation(img_size, group_size, options = "square")

			regroup = False 

			# Main algorithm starts here
			while original_class==current_class and e.evaluation_count < max_calls and not regroup:
				# Line 7
				target_class = np.argmin(rel_eval(current_image))
				# Line 8
				mutated_image = approx_min(current_image, lower, upper, rel_eval, grouping, batch_size, targeted, target_class, e ,verbose)
				# If nothing changed, change the grouping
				if True:#np.product(current_image == mutated_image):
					regroup = True
					group_size = group_size//2
					if verbose:
						print("\nGroup size: {}".format(group_size))
				current_image = mutated_image
				current_class = np.argmax(e.evaluate(current_image))
		success = not current_class == original_class
	else:
		print("  Target class: {}".format(e.idx2name(target)))
		rel_eval = lambda image :e.targeted_evaluate(image, target)
		while current_class!=target and e.evaluation_count < max_calls:
			# Algorithm 2: line 2 
			grouping = group_generation(img_size, group_size, options = "square")

			regroup = False 

			# Main algorithm starts here
			while current_class != target and e.evaluation_count < max_calls and not regroup:
				# Line 7
				target_class = target
				# Line 8
				mutated_image = approx_min(current_image, lower, upper, rel_eval, grouping, batch_size, targeted, target, e, verbose)
				# If nothing changed, change the grouping
				if np.product(current_image == mutated_image):
					regroup = True
					group_size = group_size//2
					if verbose:
						print("\nGroup size: {}".format(group_size))
				current_image = mutated_image
				current_class = np.argmax(e.evaluate(current_image))
		success = current_class == target

	counts = e.evaluation_count
	return (success, current_image, counts)
		
			

def approx_min(image, lower, upper, rel_eval, grouping, batch_size, targeted,  target_class, e, verbose):
	number_of_groups = len(grouping)
	ch = 0
	is_color = len(np.shape(image)) == 3

	base_score = rel_eval(image)
	
	# Initialize direction_array
	# This array keeps all the decision
	if is_color:
		direction_array = read_direction(image,lower,upper,grouping)
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
			u_target_score = upper_score[target_class]
			l_target_score = lower_score[target_class]
			# If adversarial input is found during exploration,
			# Stop there
			if u_target_score < 0:
				return(upper_exploratory)
			if l_target_score < 0:
				return(lower_exploratory)
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
				if targeted:
					current_class = e.current_class(image)
				else:
					target_class = np.argmin(base_score)
				if verbose:
					probabilities = e.evaluate(image)
					current_class = np.argmax(probabilities)
					new_score = base_score[target_class]
					print(str(e.evaluation_count) + "/" + str(e.max_count) + " Score: {:.3e}".format(new_score),end="\t\t\t\t\t\n")
					print("currentC: {0}...  currentP: {2:.3e}  targetP: {1:.3e}".format(e.idx2name(current_class)[:7], probabilities[target_class], np.max(probabilities)), end = '\r\b\r')
					
					
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
				return(upper_exploratory, upper_score[target_class])
			if l_target_score < 0:
				return(lower_exploratory, lower_score[target_class])
			dir = u_target_score < l_target_score
			direction_array[group_index] = dir
			if min((u_target_score,l_target_score))<minimum:
				minimum_record = (group_index, ch, dir)
				minimum = min((u_target_score,l_target_score))
	mutated_image = image_mutate(image, grouping, lower, upper, direction_array)
	total_rel_score = rel_eval(mutated_image)
	if(minimum < total_rel_score[target_class]):
		group_index, ch, direction, da = minimum_record
		da[group_index,ch] = direction
		mutated_image = image_mutate(image, grouping, lower, upper, da)
	return(mutated_image)

			
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