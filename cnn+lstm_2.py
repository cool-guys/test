
import pandas as pd
import numpy as np
import os
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, TimeDistributed,CuDNNLSTM,BatchNormalization,Flatten,Dropout
from keras.layers import Input,concatenate

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras
from keras.layers.advanced_activations import LeakyReLU

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

j = 0
x_data = []
y_data = []
x_img = []
xt_img = []
scaler = MinMaxScaler((0,100))

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

for i in range(np.size(X_train,0)):
  X_train[i] = scaler.fit_transform(X_train[i])
  
for i in range(np.size(X_test,0)):
  X_test[i] = scaler.fit_transform(X_test[i])

def train_generator():
  n = 0
  while True:
    x_train = np.reshape(X_train[n],(1,np.size(X_train[n],0),2))
    x_train_img = X_train_img[n]
    x_train_img = np.reshape(x_train_img,(1,28,28,1))
    y_train = Y_train[n].reshape(1,10)
    #print(y_train)
    if(n < np.size(X_train,0)-1):
      n += 1
    else:
      n = 0
    yield [x_train_img,x_train], y_train

def test_generator():
  n = 0
  while True:
    x_test = np.reshape(X_test[n],(1,np.size(X_test[n],0),2))
    x_test_img = X_test_img[n]
    x_test_img = np.reshape(x_test_img,(1,28,28,1))
    y_train = Y_test[n].reshape(1,10)
    #print(y_train)
    if(n < np.size(X_test,0)-1):
      n += 1
    else:
      n = 0
    yield [x_test_img,x_test], y_train
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

input_1 = Input(shape=(28, 28, 1))

x_1 = Conv2D(40, kernel_size=5, padding="same", activation = 'relu')(input_1)
x_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = Conv2D(70, kernel_size=3, padding="same", activation = 'relu')(x_1)
x_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = Conv2D(200, kernel_size=3, padding="same", activation = 'relu')(x_1)
x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)
x_1 = Dropout(0.5)(x_1)

x_1 = Conv2D(512, kernel_size=3, padding="valid", activation = 'relu')(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)
x_1 = Flatten()(x_1)
x_1 = Dense(units=1000, activation='relu')(x_1)
x_1 = Dropout(0.5)(x_1)

input_2 = Input(shape=(None, 2))
x_2 = CuDNNLSTM(256, return_sequences=True)(input_2)
x_2 = CuDNNLSTM(128)(x_2)


merged = concatenate([x_1,x_2])
m = Dense(256, activation='relu')(merged)
m = Dense(10, activation='softmax')(m)

model = Model(inputs=[input_1, input_2], outputs = m)

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit_generator(train_generator(),steps_per_epoch=200, epochs=100,validation_data=test_generator(),validation_steps=60)

score = model.evaluate_generator(test_generator(),steps=60)
scores = model.predict_generator(test_generator(),steps=60)
print(np.argmax(Y_test,1))
print(np.argmax(scores,1))
print(score[1]*100)

ROW = 5
COLUMN = 6
for i in range(ROW * COLUMN):
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, i+1)          # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {}'.format(np.argmax(scores[i],0)))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('mnist_plot.png')
plt.show()

