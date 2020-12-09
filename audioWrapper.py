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
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import time
from scipy.io.wavfile import read

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
        self.model=resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5) 
        self.model.load_state_dict(torch.load('./audio_classifier/best_model_resnet50.pt'))
		############################################################
        self.model.cpu()
        self.model.eval()
        self.calls=0
    def predict(self, wavs):
        images = sig2plot(wavs)
        self.calls+=images.shape[0]
        with torch.no_grad():
            t_images = transform(images).cpu()             
            res=self.model(t_images)
            res=torch.nn.functional.softmax(res,dim=1)
        return res.cpu().detach().numpy()
        
mymodel=CompatModel()

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

if INDICES=="":
  for j, label in enumerate(classes):
    x_test=np.stack([read_wave(i, label) for i in inds])
    y_test=j*np.ones(len(x_test))
if INDICES=="ALL":
  for j, label in enumerate(classes):
    x_test=np.stack([read_wave(i, label) for i in tqdm(range(400))])
    y_test=j*np.ones(len(x_test))
