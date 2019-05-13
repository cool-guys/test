import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization,Activation, ZeroPadding2D,CuDNNLSTM, Bidirectional,LSTM,concatenate,Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
import tensorflow as tf

from keras.callbacks import ModelCheckpoint,EarlyStopping

from utils import random_sine, plot_prediction


latent_dim = 256
j = 0
x_data = []
y_data = []
x_img = []
xt_img = []
scaler = MinMaxScaler((0,500))

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))

    x_data.append(df[['x','y']].to_numpy())
    y_data.append(df[['label']].to_numpy())

    
X_DATA = np.array(x_data)
Y_DATA = np.array(y_data)

for i in range(np.size(Y_DATA,0)):
    Y_DATA[i] = np.unique(Y_DATA[i],axis=0)
Y_DATA = Y_DATA.reshape((np.size(Y_DATA,0),1))


X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.3, random_state=52)

Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')

Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')

for i in range(np.size(X_train,0)):
  img = np.zeros((500, 500, 1), np.uint8)
  for k in range(len(X_train[i])):
    if(k != len(X_train[i])-1):
      cv2.line(img, (X_train[i][k][0],X_train[i][k][1]), (X_train[i][k+1][0],X_train[i][k+1][1]), (255,255,255), 25)
  img = cv2.flip(img, 1)
  img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)

  x_img.append(img)

X_train_img = np.array(x_img)
X_train_img = np.reshape(X_train_img,(-1,28,28,1))
X_train_img = X_train_img/255

for i in range(np.size(X_test,0)):
  img = np.zeros((500, 500, 1), np.uint8)
  for k in range(len(X_test[i])):
    if(k != len(X_test[i])-1):
      cv2.line(img, (X_test[i][k][0],X_test[i][k][1]), (X_test[i][k+1][0],X_test[i][k+1][1]), (255,255,255), 25)
  img = cv2.flip(img, 1)
  img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)

  xt_img.append(img)

X_test_img = np.array(xt_img)
X_test_img = np.reshape(X_test_img,(-1,28,28,1))
X_test_img = X_test_img/255
'''
for i in range(np.size(X_train,0)):
  X_train[i] = scaler.fit_transform(X_train[i])
  
for i in range(np.size(X_test,0)):
  X_test[i] = scaler.fit_transform(X_test[i])
'''

layers =[35,35]
regulariser = None
# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, 2))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]


# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(2,
                                   activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer='adam', loss="mse")

train_data_generator = random_sine(batch_size=16,
                                   steps_per_epoch=200,
                                   input_sequence_length=input_sequence_length,
                                   target_sequence_length=target_sequence_length,
                                   min_frequency=0.1, max_frequency=10,
                                   min_amplitude=0.1, max_amplitude=1,
                                   min_offset=-0.5, max_offset=0.5,
                                   num_signals=num_signals, seed=1969)

model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)