from mutation import *
from evaluation import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def deepSearch(cifar_, spectro_, image, label, model, distortion_cap, group_size= 16, max_calls = 10000, batch_size = 64, verbose = False, targeted = False, target = None, proba = True):
	"""
	cifar_: A boolean value. Whether we are using a CIFAR model or not (using Imgnt)
	image: The image to adverse one
	model: The machine learning model to output prediction results.
	distortion_cap: The maximum distortion per pixel. Value should be between 0 and 1 for reasonable results.
	group_size: The width of group if it was grouped in square.
	max_calls: The maximum number of calls to make. Actual call counts may overshoot.
	batch_size: The number of groups to batch mutate in approx_min(). Not batch of images.
	verbose: True will output progressions: scores/original class probability/target class probability
	targeted: Boolean value.
	target: The index of the target class. 0 to 10 for CIFAR, 0 to 999 for Imgnt
	proba: Boolean value. True will use raw probability output from the model, False will use statistic data. Go to evaluate() in evaluation.py for more detail.
	
	To understand the algorithm, lookup for Algorithm 2 in the original paper. The batching is not explained in detail in the original paper.
	
	Actual mutation and evaluations are done in approx_min() and this method provides some facilitations:
		initailization, evaluation function, (re)grouping, checking success/query-budget
	"""
	# You may skip initial part
	e = Evaluator(model, max_calls, cifar_, spectro_)
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
	current_class = np.argmax(original_probability)
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
				mutated_image = approx_min(current_image, lower, upper, rel_eval, grouping, batch_size, targeted, target_class, e, verbose)
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
		rel_eval = lambda image :e.targeted_evaluate(image, target, proba)
		while current_class!=target and e.evaluation_count < max_calls:
			# Algorithm 2: line 2 
			grouping = group_generation(img_size, group_size, options = "square")

			regroup = False 

			# Main algorithm starts here
			while current_class != target and e.evaluation_count < max_calls and not regroup:
				# Line 7 but for fixed target
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
	print("Final class: ", e.idx2name(e.current_class(current_image)))
	return (success, current_image, counts)
		
			

def approx_min(image, lower, upper, rel_eval, grouping, batch_size, targeted, target_class, e, verbose):
	"""
	image: The image to evaluate and mutate.
	lower/upper: Lower and upper bounded images to use in mutation.
	rel_eval: Method handle to use for evaluation of the image
	grouping: The grouping scheme for mutaion. Contains a group of indices. Go to generate_group() in mutation.py for more information.
	batch_size: Batching in this algorithm is for batch mutation. This is covered in algorithm detail below.
	targeted: Boolean value wether the attack is targeted or not.
	target_class: Index of the targeting class. 0 to 10 for CIFAR, 0 to 999 for Imgnt
	e: Evaluator class. Responsible of keeping the evaluation count providing an interface for the model. Go to evaluation.py for details.
	verbose: verbose
	
	Big picure of the algorithm is to take exploratory steps and keep note of the direction that made a desirable change (score going down).
	Then the noted directions are applied to the input and evaluated. The process is itereated until termination critirea is met.
	
	1. The input image is initialized with read_direction() from mutation.py
		The direction will be registered as 'lower' for a spot if no mutation was done to the spot.
		So for the first call of approx_min, all the groups will be pushed to lower bound.
	2. The picture is divided into groups (each channels are also separated for grouping)
		3. In randomly permutated groups, batch_size amount of group is mutated and evaluated for direction decisions individually.
			In a single batch, a single mutation doesn't affect other single mutation in the batch.
		4. After a batch, the whole direction decision is applied to the image and goes back to step 3 for remaining groups.
			The mutations after this is applied to the mutated image. 
			Every mutation/evaluation are all independent except the mutations from the batches before.
		5. During the process, keep track of the fittest condition
	6. After iterating through all the groups, apply the whole direction decisions
	7. Evaluate the new picture and the fittest condition during the process
	8. Return the more fitted one.
	"""
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
			
			# Explorary step and evaluations
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
				
			# Keeping decision note based on explorary step.
			dir = u_target_score < l_target_score
			if u_target_score == l_target_score:
				dir = bool(random.getrandbits(1))
			direction_array[group_index,ch] = dir
			
			# Keep track of minimum during process
			if min((u_target_score,l_target_score))<minimum:
				minimum_record = (group_index, ch, dir, da_keep)
				minimum = min((u_target_score,l_target_score))
			batch_count += 1
			
			# Apply intermediate direction decisions.
			if batch_count == batch_size:
				da_keep = direction_array.copy()
				batch_count = 0
				image = image_mutate(image, grouping, lower,upper, da_keep)
				base_score = rel_eval(image)
				if not targeted: # For normal cases, go for the nearest hill
					target_class = np.argmin(base_score)
				if verbose:
					probabilities = e.evaluate(image)
					current_class = np.argmax(probabilities)
					new_score = base_score[target_class]
					print(str(e.evaluation_count) + "/" + str(e.max_count) + " Score: {:.3e}".format(new_score),end="\t\t\t\t\t\n")
					print("currentC: {0}...  currentP: {2:.3e}  targetP: {1:.3e}".format(e.idx2name(current_class)[:7], probabilities[target_class], np.max(probabilities)), end = '\r\b\r')
				if e.evaluation_count > e. max_count:
					break
					
					
		else:# single channel (grayscale) # this part of the code is outdated
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
	
	# Apply the whole direction decisions and return the fittest compared against minimum tracked during process.
	mutated_image = image_mutate(image, grouping, lower, upper, direction_array)
	total_rel_score = rel_eval(mutated_image)
	if(minimum < total_rel_score[target_class]): # If the final image is not fit enough
		# Apply the minimu conditions found in process
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