import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization,Activation, ZeroPadding2D,CuDNNLSTM, Bidirectional,LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
import tensorflow as tf

from keras.callbacks import ModelCheckpoint,EarlyStopping

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',save_best_only=True,
                                verbose=1)

columns = ['x','y', 'label']
j = 0
x_data = []
y_data = []
scaler = MinMaxScaler((0,100))
t_data = []

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((500, 500, 1), np.uint8)
    for k in range(len(df_img)):
      if(k != len(df_img)-1):
        cv2.line(img, (df_img[k][0],df_img[k][1]), (df_img[k+1][0],df_img[k+1][1]), (255,255,255), 30)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)

    x_data.append(img)
    y_data.append(scaler.fit_transform(df[['x','y']].to_numpy()))

    
X_DATA = np.array(x_data)
X_DATA = np.reshape(X_DATA,(-1,64,64,1))
X_DATA = X_DATA/255
Y_DATA = np.array(y_data)






X_train, X_test, Y_train_, Y_test_ = train_test_split(X_DATA, Y_DATA, test_size=0.1, random_state=32)


X_val = X_train[0:18]
X_train = X_train[18:]


Y_val = keras.preprocessing.sequence.pad_sequences(Y_train_[0:18], maxlen=45, padding='post', dtype='int32')
Y_train = keras.preprocessing.sequence.pad_sequences(Y_train_[18:], maxlen=45, padding='post', dtype='int32')
Y_test = keras.preprocessing.sequence.pad_sequences(Y_test_, maxlen=45, padding='post', dtype='int32')

for i in range(18):
  lens = np.size(Y_train_[i],0)
  if(45 - lens) > 0:
    for j in range(45 - lens):
      Y_val[i][j+lens-1] = Y_train_[i][lens-1]


for i in range(162):
  lens = np.size(Y_train_[i+18],0)
  if(45 - lens) > 0:
    for j in range(45 - lens):
      Y_train[i][j+lens-1] = Y_train_[i+18][lens-1]

for i in range(20):
  lens = np.size(Y_test_[i],0)
  if(45 - lens) > 0:
    for j in range(45 - lens):
      Y_test[i][j+lens-1] = Y_test_[i][lens-1]

for i in range(162):
    for k in range(45):
        img = np.zeros((100, 100, 1), np.uint8)
        Y_train = Y_train.astype(int)
        for k in range(len(df_img)):
            if(k != len(Y_train[i])-1):
                cv2.line(img, (Y_train[i][k][0],Y_train[i][k][1]), (Y_train[i][k+1][0],Y_train[i][k+1][1]), (255,255,255), 3)
        img = cv2.flip(img, 1)
        img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)

        t_data.append(img)

for i in range(100):
  cv2.imshow("draw{}".format(i), t_data[i])
cv2.waitKey(0)
