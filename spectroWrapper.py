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
from torchvision.models import resnet50, alexnet
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import time
from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm


transform = transforms.Compose([
      # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

AUDDIR="audios/"
INDICES=""
# normalize = lambda x:(x-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
class CompatModel:
    def __init__(self):
        ############################################################
        self.model = resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5) 
        self.model.load_state_dict(torch.load('./audio_classifier/model_spectrogram_resnet50.pt'))
        ############################################################
        self.model.cuda()
        self.model.eval()
        self.calls=0
    def predict(self, sigs, f, t):
    	images = sig2spec(Sxx_dB, f, t)
        self.calls+=1
        images = np.reshape(images, images.shape[1:])
        with torch.no_grad():
            #images = Image.fromarray(np.uint8(images[:,:,:3] * 256 - 0.5))
            images = Image.fromarray(np.uint8(images[:,:,:3]*255))
            t_images = transform(images).cuda()             
            res=self.model(t_images[None, ...].float())
            res=torch.nn.functional.softmax(res,dim=1)
        model_output = res.cpu().detach().numpy()
        print(model_output)
        return model_output
        
mymodel=CompatModel()

def read_wave(wav, label):
    path = AUDDIR + label + "/" + str(wav) + ".wav"
    input_data = read(path)
    audio = input_data[1]
    sr = input_data[0]
    f, t, Sxx = signal.spectrogram((np.mean(audio, axis=1)), sr, scaling='spectrum')
    #print(Sxx.shape)
    Sxx_dB = np.log10(Sxx)
    return Sxx_dB, f, t

def sig2spec(Sxx_db, f, t):
    plt.pcolormesh(t, f, Sxx_dB)
    save_path = './audio_classifier/intermediary/' + label + "/" + str(wav) + '.png'
    plt.axis("off")
    plt.savefig(save_path, pad_inches = 0, bbox_inches = "tight")
    #plt.savefig(save_path)
    plt.close()
    pil_image = Image.open(save_path)   
    image = np.array(pil_image)
    return image[:,:,:3]

classes = ['cat', 'dog', 'parrot', 'human', 'kid'] # add more in the correct order of class 0, 1, ...
inds = range(4,14)

x_test=[]
y_test=[]
if INDICES=="":
  for j, label in enumerate(classes):
    x_test += [(read_wave(i, label)[0] + 0.5)/256 for i in inds]
    y_test += (j*np.ones(len(inds),dtype="int32")).tolist() 
    fs += [read_wave(i, label)[1] for i in inds]
    ts += [read_wave(i, label)[2] for i in inds]
if INDICES=="ALL":
  for j, label in enumerate(classes):
    x_test.append(np.stack([read_wave(i, label)[0] for i in tqdm(range(100))]).tolist())
    y_test.append(j*np.ones(len(x_test)).tolist())
    fs += [read_wave(i, label)[1] for i in range(100)]
    ts += [read_wave(i, label)[2] for i in range(100)]
x_test = np.array(x_test)
y_test = np.array(y_test)
fs = np.array(fs)
ts = np.array(ts)