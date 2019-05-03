import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf

columns = ['x','y', 'label']
j = 0
x_data = []
y_data = []
scaler = MinMaxScaler()

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((500, 500, 1), np.uint8)
    for k in range(len(df_img)):
      if(k != len(df_img)-1):
        cv2.line(img, (df_img[k][0],df_img[k][1]), (df_img[k+1][0],df_img[k+1][1]), (255,255,255), 25)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)

    x_data.append(img)
    y_data.append(df[['label']].to_numpy())

    
X_DATA = np.array(x_data)
X_DATA = np.reshape(X_DATA,(-1,28,28,1))
X_DATA = X_DATA/255
Y_DATA = np.array(y_data)

for i in range(np.size(Y_DATA,0)):
    Y_DATA[i] = np.unique(Y_DATA[i],axis=0)
Y_DATA = Y_DATA.reshape((np.size(Y_DATA,0),1))


X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.1, random_state=42)

Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')

Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')

model = Sequential()
model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(Conv2D(200, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Conv2D(512, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

model.summary()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, Y_train,
                 batch_size=16,
                 epochs=30,
                 verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])