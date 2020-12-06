import os
import pickle
import matplotlib.pyplot as plt

def select_directory(view_DSbatched):
	if view_DSbatched:
		out_most = "./DSBatched/"
		cut = -7
	else:
		out_most = "./Results/"
		cut = -1
	directory_list = [directory for directory in os.listdir(out_most)]
	if len(directory_list) == 0:
		print("No directory found")
		return
	for dir_number in range(len(directory_list)):
		print("{0:02d}\t{1}".format(dir_number, directory_list[dir_number][:cut].replace("_",":")))
	selection = int(input("Type in index of the directory\n>>> "))
	return(out_most + directory_list[selection])
	
def load_pkl(path):
	pkl_list = [file for file in os.listdir(path) if file[-3:]=="pkl" and file[0] != "d"]
	if len(pkl_list) == 0:
		print("No .pkl file found")
		return
	for pkl_number in range(len(pkl_list)):
		print("{0:02d}\t{1}".format(pkl_number, pkl_list[pkl_number][:-4]))
	selections=[int(x) for x in input("Type in indices of files, each separated by spacing\n>>> ").split()]
	return [path+"/"+pkl_list[selection] for selection in selections]
	
if __name__ == "__main__":
	view_DSbatched = bool(int(input("0: View Organized Results\n1: View DSBatched\n>>> ")))
	files = load_pkl(select_directory(view_DSbatched))
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
			
	