import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
import pywt
import cv2
import scipy.io.wavfile
import soundfile as sf


def save_image(type, data, wave_path = 'test.wav'):
    newDir = f'./Audio_Results/{type}'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    np.save(f"{newDir}/{wave_path.split('.')[0]}", data)

def resize(data, size = (224, 224)):
    return cv2.resize(data, dsize = size, interpolation = cv2.INTER_CUBIC)

def normalize(vect_in, percent_acceptation=80, not_clip_until_acceptation_time_factor=1.5):
    percent_val = np.percentile(abs(vect_in).reshape(-1), percent_acceptation)
    percent_val_matrix = not_clip_until_acceptation_time_factor * np.repeat(percent_val,
                                                                            vect_in.shape[0]*vect_in.shape[1],
                                                                            axis=0).reshape((vect_in.shape[0], vect_in.shape[1]))
    matrix_clip = np.maximum(np.minimum(vect_in, percent_val_matrix), -percent_val_matrix)
    return np.divide(matrix_clip, percent_val_matrix)

def sig2spec(wave_path, n_fft = 1024, hop_length = 512, norm = True, save = False):
    audio, sr =  librosa.load(wave_path, sr = 22500)
    stft = librosa.stft(audio, n_fft = n_fft, hop_length = hop_length)
    magnitude, phase = librosa.magphase(stft)
    magnitude_db = librosa.amplitude_to_db(magnitude)
    if save:
        save_image('spectrogram', spectrogram)
    return magnitude_db, phase
    
def sig2scalo(wave_path, scales = np.arange(1, 128), wavelet = 'morl', norm = True, save = False):
    audio, sr =  librosa.load(wave_path, sr = 22500)
    coeff, freq = pywt.cwt(audio, scales, wavelet)
    #scalogram = cv2.resize(coeff, dsize=(224, 224), interpolation = cv2.INTER_CUBIC)
    if save:
        save_image('scalogram', scalo)
    return coeff

def sig2mfcc(wave_path, n_mfcc=200, norm = True, save = False):
    audio, sr =  librosa.load(wave_path, sr = 22500)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    if save:
        save_image('mfcc', mfcc)
    return mfcc

def spec2sig(spec, name):
    audio = librosa.griffinlim(spec)
    newDir = './Audio_Results/audio_from_spec'
    if not os.path.exists(newDir):
        os.makedirs(newDir)        
    sf.write(f"{newDir}/{name}.wav", audio, 22500)

def scalo2sig(scalo, name, scales = np.arange(1, 128), wavelet = 'morl'):
    mwf = pywt.ContinuousWavelet(wavelet).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
    r_sum = np.transpose(np.sum(np.transpose(scalo)/ scales ** 0.5, axis=-1))
    audio = r_sum * (1 / y_0)
    newDir = './Audio_Results/audio_from_scalo'
    if not os.path.exists(newDir):
        os.makedirs(newDir)        
    sf.write(f"{newDir}/{name}.wav", audio, 22500)


def mfcc2sig(mfcc, name):
    audio = librosa.feature.inverse.mfcc_to_audio(mfcc)
    newDir = './Audio_Results/audio_from_mfcc'
    if not os.path.exists(newDir):
        os.makedirs(newDir)    
    sf.write(f"{newDir}/{name}.wav", audio, 22500)


if __name__ == '__main__':
    spec = sig2mfcc('360.wav')
    plt.figure()
    plt.axis('off')
    plt.imshow(spec, cmap='Reds', interpolation='nearest', aspect='auto')
    plt.show()
    mfcc2sig(spec, 'test')