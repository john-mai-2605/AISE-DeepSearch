import youtube_dl
from pydub import AudioSegment

#DEFINE root dir
root = "../audios"

# ydl_opts = {
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '192',
#     }],
#     'outtmpl':root + '/kids.%(ext)s',
# }

# url_dog = "https://www.youtube.com/watch?v=muAiqLbblVQ"
# url_dog = "https://www.youtube.com/watch?v=7ej1ur8amCo"
# url_cat = "https://www.youtube.com/watch?v=P9AY5rc5M28"
# url_parrot = "https://www.youtube.com/watch?v=JkvRP-WMU9o"
# url_human = "https://www.youtube.com/watch?v=R-deIwiFyuw"
# url_kid = "https://www.youtube.com/watch?v=j5-6bI3hR2M"

# # Extract mp3 audio files from youtube video link
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url_kid])

#convert mp3 to wav
classes = ['cat', 'dog', 'parrot', 'human', 'kid']
# for x in classes:
#   sound = AudioSegment.from_mp3(root+"/"+x+"s.mp3")
#   sound.export(root+"/"+x+"s.wav", format="wav")

# for x in classes:
#   path = root + '/' + x
#   count=1
#   for i in range(10,2000,10):
#       t1 = i * 1000 #Works in milliseconds
#       t2 = (i+10) * 1000
#       newAudio = AudioSegment.from_wav(path+"s.wav")
#       newAudio = newAudio[t1:t2]
#       newAudio.export(path+'/'+str(count)+'.wav', format="wav") #Exports to a wav file in the current path.
#       count+=1


from scipy.io.wavfile import read
import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
from os import walk
import os
import librosa

# cat_wavs, dog_wavs, parrot_wavs, human_wavs, kids_wavs  
wavs = [[] for i in range(5)]

for i in range(5):
  curpath = root+'/'+classes[i]
  # print(curpath)
  for (_,_,filenames) in walk(curpath):
    wavs[i].extend(filenames)
    print(wavs[i])
    break

for i,cur_wavs in enumerate(wavs): #100, 100, 60, 100, 91 
  animal = classes[i]
  newDir = root+"/plots/"+animal + "Plots"
  if not os.path.exists(newDir):
    os.makedirs(newDir)
  for wav in cur_wavs:
      path = root+"/" +animal+ "/" + wav
      audio, sr =  librosa.load(path, sr = None)
      n = len(audio)
      n_fft = 204
      audio_pad = librosa.util.fix_length(audio, n + n_fft // 2)
      stft = librosa.stft(audio_pad, n_fft = n_fft)
      magnitude, phase = librosa.magphase(stft)
      magnitude_db = librosa.amplitude_to_db(magnitude)
      plt.imshow(magnitude_db, interpolation='nearest', aspect='auto')
      plt.axis("off")
      plt.savefig(newDir + "/" + wav.split('.')[0] + '.png',  pad_inches = 0, bbox_inches = "tight") 
      plt.close('all')


for i,cur_wavs in enumerate(wavs): #100, 100, 60, 100, 91 
  animal = classes[i]
  newDir = root+"/plots/"+animal + "PlotsTimes"
  if not os.path.exists(newDir):
    os.makedirs(newDir)
  count = 0
  for wav in cur_wavs:
    if count < 100:
      _, audio = read(root+"/" +animal+ "/" + wav)
      plt.plot(audio)
      plt.savefig(newDir + "/" + wav.split('.')[0] + '.png')
      plt.close('all')
      count += 1

