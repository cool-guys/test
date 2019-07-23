import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation,ReLU,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from imageloader import data_process
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

number = False

columns = ['x','y', 'label']

MODEL_SAVE_FOLDER_PATH = './model/cnn_alpha'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)
                                
j = 0
x_data = []
y_data = []

'''
dp_train = data_process('./DATA/aug/all/train')
dp_test = data_process('./DATA/aug/all/test')

dp_train.point_data_load()
dp_train.image_make()
dp_train.sequence_50()

dp_test.point_data_load()
dp_test.image_make()
dp_test.sequence_50()

X_train = dp_train.images
X_test = dp_test.images
Y_train = dp_train.label
Y_test = dp_test.label
print(Y_train)
'''

dp = data_process('./DATA/aug/all/train/Alphabet',False)
dp.point_data_load()
#dp.image_read()
dp.sequence_50()
dp.image_make()
dp.data_shuffle()


size = int(np.size(dp.point,0) * 0.7)


X_train = dp.images[:size]
X_test = dp.images[size:]
Y_train = dp.label[:size]
Y_test = dp.label[size:]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

Y_train = keras.utils.to_categorical(Y_train,num_classes=26, dtype='float32')
Y_test = keras.utils.to_categorical(Y_test,num_classes=26, dtype='float32')

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding="valid",input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.45))

model.add(Conv2D(64, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(128, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(256, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.45))

model.add(Dense(26,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, Y_train,
                batch_size=64,
                validation_data=(X_test,Y_test),
                epochs=250,
                verbose=1,
                callbacks=[cb_checkpoint])

model = load_model('./model/cnn_alpha.hdf5')

score = model.evaluate(X_test, Y_test, verbose=0)
pred = model.predict(X_test)
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(pred,1)
print(np.argmax(pred,axis=1))
print(score[1]*100)
#print("복원된 모델의 정확도: {:5.2f}%".format(100*score))
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    if(number):
      print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    else:
      print('t:{},p:{}'.format(chr(97+true_value[i]),chr(97+predict_value[i])))
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
    if(number):
      plt.title('predict = {}'.format((np.argmax(pred[i],0))))
    else:
      plt.title('predict = {}'.format(chr(97+ np.argmax(pred[i],0))))  
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('cnn.png')
plt.show()