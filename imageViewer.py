import os
import pickle
import matplotlib.pyplot as plt

def select_directory():
	directory_list = [directory for directory in os.listdir("./DSBatched/")]
	if len(directory_list) == 0:
		print("No directory found")
		return
	for dir_number in range(len(directory_list)):
		print("{0:02d}\t{1}".format(dir_number, directory_list[dir_number][:-7].replace("_",":")))
	selection = int(input("Type in index of the directory\n>>> "))
	return("./DSBatched/" + directory_list[selection])
	
def load_pkl(path):
	pkl_list = [file for file in os.listdir(path) if file[-3:]=="pkl"]
	if len(pkl_list) == 0:
		print("No .pkl file found")
		return
	for pkl_number in range(len(pkl_list)):
		print("{0:02d}\t{1}".format(pkl_number, pkl_list[pkl_number][:-4]))
	selections=[int(x) for x in input("Type in indices of files, each separated by spacing\n>>> ").split()]
	return [path+"/"+pkl_list[selection] for selection in selections]
	
if __name__ == "__main__":
	files = load_pkl(select_directory())
	imgs = []
	for file_path in files:
		with open(file_path,'rb') as file:
			temp = pickle.load(file)
			size = temp.shape[1:]
			imgs.append(temp.reshape(size))
	#imgs = [pickle.load(open(file_path, 'rb')).reshape(pickle.load(open(file_path, 'rb')).shape[1:]) for file_path in files]
	for image in imgs:
		plt.imshow(image)
		plt.show()
			
	