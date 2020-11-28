import numpy as np

class Evaluator:
	def __init__(self, model):
		self.evaluation_count = 0
		self.model = model
	def evaluate(self, image):
		self.evaluation_count +=1
		return np.random.rand(30)
		
	def relative_evaluate(self, image, class_number):
		new_probability = self.evaluate(image)
		class_prob = new_probability[class_number] 
		
		relative_score = class_prob - new_probability
		# Set current class fitness so it won't become target_class
		relative_score[class_number] = 1
		return relative_score


	def decide_direction(self, original_probability, mutated_probability):
		pass