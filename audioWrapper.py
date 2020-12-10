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
import matplotlib.pyplot as plt

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
        self.model=alexnet()
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2) 
        self.model.load_state_dict(torch.jit.load('./audio_classifier/best_model.pt'))
        ############################################################
        self.model.cpu()
        self.model.eval()
        self.calls=0
    def predict(self, wavs, **kwargs):
        images = sig2plot(wavs)
        self.calls+=images.shape[0]
        with torch.no_grad():
            t_images = transform(images).cpu()             
            res=self.model(t_images)
            res=torch.nn.functional.softmax(res,dim=1)
        return res.cpu().detach().numpy()
        
#mymodel=CompatModel()

def read_wave(wav, label):
    path = AUDDIR + label + "/" + str(wav) + ".wav"
    input_data = read(path)
    audio = input_data[1]
    return audio

def sig2plot(audio):
    plt.plot(audio)
    plt.savefig('./audio_classifier/intermediary/plot.png')
    plt.close()
    return Image.open('./audio_classifier/intermediary/plot.png')

classes = ['dog'] # add more in the correct order of class 0, 1, ...
inds=[360]

x_test = []
y_test = []

if INDICES=="":
  for j, label in enumerate(classes):
    x_test.append(np.stack([read_wave(i, label) for i in inds]).tolist())
    y_test.append(j*np.ones(len(x_test)).tolist())
if INDICES=="ALL":
  for j, label in enumerate(classes):
    x_test.append(np.stack([read_wave(i, label) for i in tqdm(range(100))]).tolist())
    y_test.append(j*np.ones(len(x_test)).tolist())
x_test = np.array(x_test)
y_test = np.array(y_test)
