import numpy as np
import time

class Evaluator:
	def __init__(self, model, max_count):
		self.evaluation_count = 0
		self.model = model
		self.max_count = max_count
		
	def evaluate(self, image):
		self.evaluation_count +=1
		shape = (1,) + image.shape
		#started_time = time.process_time()
		prediction = self.model.predict(np.reshape(image,shape))
		#print(time.process_time() - started_time)
		return prediction.reshape(-1)
		
	def relative_evaluate(self, image, class_number):
		
		new_probability = self.evaluate(image)
		class_prob = new_probability[class_number] 
		
		relative_score = class_prob - new_probability

    # Set current class fitness to 1 so it won't become target_class
		relative_score[class_number] = 1
		return relative_score


	def decide_direction(self, original_probability, mutated_probability):
		pass