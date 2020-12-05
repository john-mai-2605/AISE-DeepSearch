import numpy as np
import time
import json

class Evaluator:
	def __init__(self, model, max_count):
		self.evaluation_count = 0
		self.model = model
		self.max_count = max_count
		self.classes = json.load(open("classes.json","r"))
		self.classes = {int(k):v for k,v in self.classes.items()}
		
	def current_class(self,image):
		return(np.argmax(self.evaluate(image)))
		
	def idx2name(self,class_index):
		return(self.classes[class_index])
		
	def evaluate(self, image, proba = True):
		shape = (1,) + image.shape
		if proba:
			self.evaluation_count +=1
			#started_time = time.process_time()
			prediction = self.model.predict(np.reshape(image,shape))
			#print(time.process_time() - started_time)
			return prediction.reshape(-1)
		else:
			predictions = np.array([])
			img = np.reshape(image,shape)
			predictions = [self.model.predict(img + np.random.normal(0, 30, shape), False).reshape(-1).tolist() for i in range(100)]
			self.evaluation_count +=100
			predictions = np.array(predictions)
			prediction = np.mean(predictions, axis = 0)
			return prediction
		
	def relative_evaluate(self, image, class_number, proba = True):
		
		new_probability = self.evaluate(image, proba)
		class_prob = new_probability[class_number] 
		
		relative_score = class_prob - new_probability

    # Set current class fitness to 1 so it won't become target_class
		relative_score[class_number] = 1
		return relative_score

	def targeted_evaluate(self, image, target):
		prob = self.evaluate(image)
		if np.argmax(prob) == target:
			return -prob
		return 1/prob

	def decide_direction(self, original_probability, mutated_probability):
		pass