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
#import tensorflow as tf
from model import Model
from pickle import dump, load
from sys import argv
import tensorflow as tf
from tensorflow.compat.v1 import keras
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.disable_v2_behavior()

_, (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test=x_test/255.0
x_test=x_test.reshape(10000,32,32,3)
y_test=y_test.reshape(-1)
class CompatModel:
    def __init__(self,folder):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.84)
        #config = tf.ConfigProto()#gpu_options=gpu_options)
        #config.gpu_options.allow_growth=True
        self.sess=tf.compat.v1.Session()#config=config)
        model_file = tf.train.latest_checkpoint(folder)
        model=Model("eval")
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)
        self.model=model
        self.calls=0
    def predict(self,images,**kwargs):
        self.calls+=images.shape[0]
        res=np.exp(self.sess.run(self.model.pre_softmax,feed_dict={self.model.x_input:images*255,self.model.y_input:[1]}))
        return res/np.sum(res,axis=1).reshape(-1,1)
mymodel=CompatModel("model_defended/")
