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
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from imageloader import data_process
columns = ['x','y', 'label']
j = 0
x_data = []
y_data = []
scaler = MinMaxScaler()

dp = data_process('./DATA/test')
dp.point_data_load()
dp.image_make()
dp.data_shuffle()

size = int(np.size(dp.images,0) * 0.7)

X_train = dp.images[:size]
X_test = dp.images[size:]
Y_train = dp.label[:size]
Y_test = dp.label[size:]
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
                validation_data=(X_test,Y_test),
                epochs=100,
                verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
pred = model.predict(X_test)
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(pred,1)
print(np.argmax(pred,axis=1))
#print("복원된 모델의 정확도: {:5.2f}%".format(100*score))
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    list_.append(i)



ROW = 5
COLUMN = 4
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {}'.format(np.argmax(pred[i],0)))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('mnist_plot.png')
plt.show()