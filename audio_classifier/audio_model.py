from __future__ import unicode_literals
import youtube_dl
from pydub import AudioSegment

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

#DEFINE root dir
root = ""

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl':root + '/kids.%(ext)s',
}

url_dog = "https://www.youtube.com/watch?v=muAiqLbblVQ"
url_dog = "https://www.youtube.com/watch?v=7ej1ur8amCo"
url_cat = "https://www.youtube.com/watch?v=P9AY5rc5M28"
url_parrot = "https://www.youtube.com/watch?v=JkvRP-WMU9o"
url_human = "https://www.youtube.com/watch?v=R-deIwiFyuw"
url_kid = "https://www.youtube.com/watch?v=j5-6bI3hR2M"

# Extract mp3 audio files from youtube video link
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url_kid])

#convert mp3 to wav
# classes = ['cat', 'dog', 'parrot', 'human', 'kid']
# for x in classes:
#   sound = AudioSegment.from_mp3(root+"/"+x+"s.mp3")
#   sound.export(root+"/"+x+"s.wav", format="wav")

# for x in classes:
#   path = root + '/' + x
#   count=1
#   for i in range(1,3600,10):
#       t1 = i * 1000 #Works in milliseconds
#       t2 = (i+10) * 1000
#       newAudio = AudioSegment.from_wav(path+"s.wav")
#       newAudio = newAudio[t1:t2]
#       newAudio.export(path+'/'+str(count)+'.wav', format="wav") #Exports to a wav file in the current path.
#       count+=1

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import walk
import os

cat_wavs, dog_wavs, parrot_wavs, human+wavs, kid_wavs = ([] for i in range(5))
for (_,_,filenames) in walk(root+'/cat'):
    cat_wavs.extend(filenames)
    break
for (_,_,filenames) in walk(root+'/dog'):
    dog_wavs.extend(filenames)
    break

def savePng(wavs, animal):
  newDir = root+"/plots/"+animal + "Plots"
  if not os.path.exists(newDir):
    os.makedirs(newDir)
  count = 0
  for wav in wavs:
    if count < 100:
      input_data = read(root+"/" +animal+ "/" + wav)
      audio = input_data[1]
      plt.plot(audio)
      plt.savefig(newDir + "/" + wav.split('.')[0] + '.png')
      plt.close('all')
      count += 1


# savePng(cat_wavs, 'cat')
# savePng(dog_wavs, 'dog')


dataset = datasets.ImageFolder(
    root+'/plots',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 70, 70])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)

model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

NUM_EPOCHS = 10
best_accuracy = 0.0
model_path = root + '/best_model.pt'

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('Epoch %d: Accuracy is %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), model_path)
        best_accuracy = test_accuracy