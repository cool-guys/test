import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential,Model
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from imageloader import data_process
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

number = False


if(number):
  dp = data_process('./DATA/aug/all/train/number',number)
  dp.point_data_load()
  #dp.image_read()
  dp.sequence_50()
  dp.image_make()
  dp.data_shuffle()

  class_num = 10

else:
  dp_train = data_process('./DATA/aug/all/train/Alphabet',number)
  dp_train.point_data_load()
  #dp.image_read()
  dp_train.sequence_50()
  dp_train.image_make()
  dp_train.data_shuffle()
  
  dp_test = data_process('./DATA/aug/all/test/Alphabet',number)
  dp_test.point_data_load()
  #dp.image_read()
  dp_test.sequence_50()
  dp_test.image_make()
  dp_test.data_shuffle()
 
  class_num = 26


X_train = dp_train.images[:]
X_test = dp_test.images[:]
Y_train = dp_train.label[:]
Y_test = dp_test.label[:]


Y_train = keras.utils.to_categorical(Y_train,num_classes=class_num, dtype='float32')
Y_test = keras.utils.to_categorical(Y_test,num_classes=class_num, dtype='float32')

# cudnn 오류 해결용(RTX에서만 생기는 문제로 보임))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = load_model('./model/cnn_alpha.hdf5')

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1,28,28,1))

 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

display_activation(activations, 4, 8, 1)
