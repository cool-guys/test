
import pandas as pd
import numpy as np
import os
from keras.models import Sequential,Model
from keras.layers import *

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import load_model
import keras
from keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import time
from imageloader import data_process
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import copy
from keras.preprocessing import sequence

#reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',factor=0.5, patience=20)
#model save
MODEL_SAVE_FOLDER_PATH = '../model/lstm_only'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc',
                                verbose=1, save_best_only=True)


#early_stopping = EarlyStopping()

# Minmax 
scaler = MinMaxScaler()
scaler_ = MinMaxScaler()

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
  train_len = np.size(dp_train.point,0)
  print(train_len)

  
  dp_test = data_process('./DATA/aug/all/test/Alphabet',number)
  dp_test.point_data_load()
  #dp.image_read()
  dp_test.sequence_50()
  dp_test.image_make()
  dp_test.data_shuffle()
  test_len = np.size(dp_test.point,0)

  class_num = 26
  
  
# minmax 실행
train_x = dp_train.point[:,:,0].reshape((-1,1))
test_x = dp_test.point[:,:,0].reshape((-1,1))

train_y = dp_train.point[:,:,1].reshape((-1,1))
test_y = dp_test.point[:,:,1].reshape((-1,1))

data_x = scaler.fit_transform(train_x).reshape((-1,64,1))
data_test_x = scaler.transform(test_x).reshape((-1,64,1))

data_y = scaler_.fit_transform(train_y).reshape((-1,64,1))
data_test_y = scaler_.transform(test_y).reshape((-1,64,1))

data = np.hstack((data_x,data_y)).reshape((-1,64,2),order='F')
data_test = np.hstack((data_test_x,data_test_y)).reshape((-1,64,2),order='F')

#train set test set 나누기
X_train = data[:,:,:]
#X_train_ = copy.deepcopy(dp.point)
X_test = data_test[:,:,:]
X_test_ = copy.deepcopy(data[:,:,:])

X_train_img = dp_train.images[:]
X_test_img = dp_test.images[:]
Y_train = dp_train.label[:]
Y_test = dp_test.label[:]


Y_train = keras.utils.to_categorical(Y_train,num_classes=class_num, dtype='float32')
Y_test = keras.utils.to_categorical(Y_test,num_classes=class_num, dtype='float32')

# cudnn 오류 해결용(RTX에서만 생기는 문제로 보임))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


input_2 = Input(shape=(64, 2))

x_2 = Conv1D(32,kernel_size=4, padding='same',strides=1)(input_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = AveragePooling1D()(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(64,kernel_size=5, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(alpha=0.2)(x_2)
x_2 = AveragePooling1D()(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(128, kernel_size=6, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = AveragePooling1D()(x_2)
x_2 = Dropout(0.5)(x_2)


x_2 = Conv1D(256, kernel_size=7, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = AveragePooling1D()(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(512, kernel_size=8, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = AveragePooling1D()(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = GlobalAveragePooling1D()(x_2)

x_2 = Dense(class_num,activation='softmax')(x_2)

model = Model(inputs=input_2, outputs = x_2)

model.summary()
model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.001,epsilon=1e-04),
                metrics=['accuracy'])

results = model.fit(X_train,Y_train,epochs=250,batch_size=128,validation_data=(X_test,Y_test),callbacks=[cb_checkpoint,lr_reduce])

model = load_model('../model/lstm_only.hdf5')

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

    f, ax = plt.subplots(5, 4, figsize=(18,8))

for i,j in enumerate(list_):
    if(i > 19):
        break
    image = X_test[j].reshape(28,28)
    ax[i//4,i%4].imshow(image, cmap = cmp.gray)
    ax[i//4,i%4].set_title('predict = {},{}'.format(chr(97+ np.argmax(pred[j],0)),chr(97 + true_value[j])),fontsize= 15)
    ax[i//4,i%4].axis('off')
plt.tight_layout()
plt.savefig('LSTM_only_image.png')
plt.show()

cm = confusion_matrix(true_value, predict_value)
df_cm = pd.DataFrame(cm, index = [i for i in "abcdefghijklmnopqrstuvwxyz"],
                  columns = [i for i in "abcdefghijklmnopqrstuvwxyz"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_map_lstm_only.png')
plt.show()