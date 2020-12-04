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
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import time

IMGDIR="images/req_images/"
INDICES=""
normalize = lambda x:(x-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
class CompatModel:
    def __init__(self):
		############################################################
        self.model=inception_v3(pretrained=True)
		############################################################
        self.model.cuda()
        self.model.eval()
        self.calls=0
    def predict(self,images,proba=True):
        #start_time = time.process_time()
        self.calls+=images.shape[0]
        with torch.no_grad():
            t_images=torch.tensor(normalize(images),dtype=torch.float).cuda()             
            t_images=t_images.permute(0,3,1,2)
            res=self.model(t_images)
            res=torch.nn.functional.softmax(res,dim=1)
        #print("{:.3f}".format(time.process_time()-start_time))
        if proba:
            return res.cpu().detach().numpy()
        else:
            c = torch.argmax(res).cpu().detach().numpy()
            output = torch.zeros(10)
            output[c] = 1
            return output
mymodel=CompatModel()

def load_image(id):
    path=IMGDIR+"ILSVRC2012_val_000"+str(id).zfill(5)+".JPEG"
    image = Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((256,256))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img
inds=[37860, 5869]

labels=[0, 1]
labels=np.array(labels)
if INDICES=="":
    x_test=np.stack([load_image(i+1) for i in inds])
    y_test=labels
if INDICES=="ALL":
    x_test=np.stack([load_image(i+1) for i in tqdm(range(50000))])
    y_test=labels
