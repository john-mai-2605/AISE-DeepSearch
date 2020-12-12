# Copyright 2020 Max Planck Institute for Software Systems

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pickle import dump,load
import sys
from datetime import datetime
from os import mkdir
from os.path import exists
from tqdm import tqdm
import argparse

from deepSearch import *

# Parse argument
parser = argparse.ArgumentParser()
    
# Argument lists
parser.add_argument('--targeted', type=int, default=0, choices = [0,1], help="Tartgeted (1) or non-targeted (0, default) attack")
parser.add_argument('--target', type=int, default=0, help="Tartgeted class (applicable for targeted attack)")
parser.add_argument('--cifar', action ='store_true', default=False, help="turn on for cifar")
parser.add_argument('--undef', action ='store_true', default=False, help="turn on for undefended")
parser.add_argument('--spectro', action ='store_true', default=False, help="turn on for spectro")
parser.add_argument('--proba', type=int, default=1, choices=[0,1], help="Output from model is probability (1, default) or class (0)")


# Read the arguments
args = parser.parse_args()
targeted = args.targeted == 1
target = args.target
undefended = args.undef
cifar_ = args.cifar
proba = args.proba == 1
spectro_ = args.spectro

log_entry = ""

# Initial conditions for Imgnet
img_x, img_y = 256, 256
grs= 32
batch_size = 64


if targeted:
	print(f'Targeted attack with target of class @{target}')
	log_entry += "Targeted "
else:
	print('Non-targeted attack')
	log_entry += "Non-targeted "
	
if cifar_:
	log_entry += "Cifar "
	if undefended:
		from madryCifarUndefWrapper import *
		target_set=load(open("indices.pkl","rb"))
		log_entry += "Undefended "
	else:
		from madryCifarWrapper import *
		target_set=load(open("def_indices.pkl","rb"))
		log_entry += "Defended "
	img_x, img_y = 32,32
	grs = 4
	batch_size = 64
elif spectro_:
	from spectroWrapper2 import*
	target_set = range(50)
	log_entry += "spetcro"
	img_x, img_y = 103, -1
	#img_x, img_y = 480, 640
	grs = 50
	batch_size = 15
else:
	from imgntWrapper import *
	target_set = range(50)
	log_entry += "Imagenet "

# Creating folder to store results
if not exists("DSBatched"):
    mkdir("DSBatched")
path="DSBatched/"+str(datetime.now()).replace(":","_")+"/"
mkdir(path)
with open(path+"log.txt","w") as log_path:
	# Comment this line to see results in console!
	sys.stdout=log_path
	# ^!!!important!!!^
	
	Data={}
	succ=0
	tot=0
	print("group_size ",grs," batch_size ", batch_size)
	print(log_entry)
	for j in tqdm(target_set[:50]):
		print("\nStarting attack on image", tot, " ", j)
		tot+=1
		#def deepSearch(cifar_, image, label, model, distortion_cap, 
		#	group_size= 16, max_calls = 10000, batch_size = 64, verbose = False,
		#	targeted = False, target = None, proba = True):
		if spectro_: 
			kwargs = {'j': j}
		ret = deepSearch(cifar_, spectro_, x_test[j], y_test[j], mymodel, 5/256, 
			group_size = grs, max_calls = 10000, batch_size = batch_size, verbose = False, 
			targeted = targeted, target = target, proba = proba, **kwargs)
		if spectro_:
			dump(ret[1].reshape(1,img_x,img_y),open(path+classes[j//items_per_class]+"_"+"{:05d}".format(inds[j%items_per_class])+".pkl","wb"))
		else:
			dump(ret[1].reshape(1,img_x,img_y,3),open(path+"image_"+"{:05d}".format(j)+".pkl","wb"))
		Data[j]=(ret[0],ret[2])
		if ret[0]:
			succ+=1
			print("Attack Succeeded with",ret[2],"queries, success rate is\t",100*succ/tot)
		else:
			print("Attack Failed using",ret[2],"queries, success rate is\t",100*succ/tot)
		dump(Data,open(path+"data.pkl","wb"))
