import numpy as np
import matplotlib.pyplot as plt
import time
import json

class Evaluator:
	def __init__(self, model, max_count, cifar_):
		self.evaluation_count = 0
		self.model = model
		self.max_count = max_count
		self.classes = json.load(open("classes.json","r"))
		self.classes = {int(k):v for k,v in self.classes.items()}
		self.cifar_ = cifar_
		
	def current_class(self,image):
		return(np.argmax(self.evaluate(image)))
		
	def idx2name(self,class_index):
		cifar_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		if self.cifar_:
			return(cifar_names[class_index])
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
			predictions = [
				self.top_rank(
					self.model.predict(
						img + np.random.normal(0, 14/256, shape)
					)[0]
				).reshape(-1).tolist() for i in range(50)]
			self.evaluation_count +=50
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

	def targeted_evaluate(self, image, target, proba = True):
		prob = self.evaluate(image, proba)
		if np.argmax(prob) == target:
			return -prob
		return 1/prob

	def top_rank(self, array_, top_number = 3):
		# with help from
		# https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice/5284703#5284703
		temp = array_.argsort()
		ranks = np.empty_like(temp)
		ranks[temp] = np.arange(len(array_))
		# ^ ranks is rank of array_, higher number gets higher rank
		ranks = ranks - (len(ranks) - top_number) + 1
		cutout = ranks < 0
		ranks[cutout] = 0
		return 2 * ranks/(top_number**2 + top_number)
		