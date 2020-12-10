import numpy as np
import time
import json

class Evaluator:
	def __init__(self, model, max_count, cifar_, spectro_):
		"""
		"""
		self.evaluation_count = 0
		self.model = model
		self.max_count = max_count
		# Reads imagenet class label names from the file.
		self.classes = json.load(open("classes.json","r"))
		self.classes = {int(k):v for k,v in self.classes.items()}
		self.cifar_ = cifar_
		self.spectro_ = spectro_
		
	def current_class(self,image, **kwargs):
		"""
		Test the image in the model and return the maximum class index.
		"""
		return(np.argmax(self.evaluate(image, **kwargs)))
		
	def idx2name(self,class_index):
		"""
		Reads class_index and returns it's name.
		Range for each networks:
		Imgnet	(0 ~ 999)
		CIFAR	(0~10)
		"""
		cifar_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		audio_names = ["Cat", "dog", "parrot", "human", "kid"]
		if self.cifar_:
			return(cifar_names[class_index])
		elif self.spectro_:
			return(audio_names[class_index])
		return(self.classes[class_index])
		
	def evaluate(self, image, proba = True, **kwargs):
		"""
		image: Image you want to get probability array for.
		proba: Whether to use raw probability output (True)
			or to use categorical statistic output (False)
			The later option will use 30~100 times more query count and will be inaccurate.
			This option is added to simulate more realistic blackbox condition.
			
		This method takes the image into the model and gets the output.
		Categorical output is handled in this method so it is usable in similar way as probability output.
		"""
		"""
		The idea of converting categorical output to a probability-like output is to try for many times and used frequency distribution of the category output.
		"""
		shape = (1,) + image.shape
		if proba: # normal probability output
			self.evaluation_count +=1
			
			# model.predict comes from *Wrapper.py
			prediction = self.model.predict(np.reshape(image,shape), **kwargs)
			return prediction.reshape(-1)
		else: # no probability, only ranked categorical output. Go to top_rank() for more detail.
			predictions = np.array([])
			img = np.reshape(image,shape)
			predictions = [
				self.top_rank( # Only top few categories are left with higher nmber for higher rank
					self.model.predict(
						# Random noise is added to test categorical confidence
						img + np.random.normal(0, 14/256, shape) 
					)[0]
				).reshape(-1).tolist() for i in range(50)]
			# Frequency distribution is used as output.
			self.evaluation_count +=50
			predictions = np.array(predictions)
			prediction = np.mean(predictions, axis = 0)
			return prediction
		
	def relative_evaluate(self, image, class_number, proba = True, **kwargs):
		"""
		image: Image to evaluate it's score. The goal is to get the score below 0.
		class_number: Index of class that we are measuring the probability to.
			It's usually the original class.
		proba: Boolean option to use different method for prediction output.
			goto evaluate() for more detail.
			
		Relative output is probability gap between a specified class and all the other ones.
		We will be trying to minimize the gap until one reaches negative in deepsearch
		"""
		new_probability = self.evaluate(image, proba, **kwargs)
		class_prob = new_probability[class_number] 
		
		relative_score = class_prob - new_probability

		# Set current class fitness to 1 so it won't become target_class
		relative_score[class_number] = 1
		return relative_score

	def targeted_evaluate(self, image, target, proba = True, **kwargs):
		"""
		This is just like regualr evaluate(), with a little more to detect the target has been reached.
		It outputs absolute probabilities until target class is most prominent.
		"""
		prob = self.evaluate(image, proba)
		if np.argmax(prob) == target:
			return -prob
		return 1/prob

	def top_rank(self, array_, top_number = 3):
		"""
		array_: Array to rank and socre.
		top_number: Number of elements to survive from array_
		
		This method ranks the elements in array_ and only scores the ones that make in to the top_number.
		Scores are normalized so the sum is 1
		"""
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
		