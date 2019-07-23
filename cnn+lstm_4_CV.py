
import pandas as pd
import numpy as np
import os
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, TimeDistributed,CuDNNLSTM,BatchNormalization,Flatten,Dropout,CuDNNGRU,MaxPooling1D,Conv1D,GlobalMaxPooling1D
from keras.layers import Input,concatenate

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import keras

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
cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

#reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')

early_stopping = EarlyStopping()

scaler = MinMaxScaler((0,100))

dp = data_process('./DATA/aug/all/train/Alphabet',False)
dp.point_data_load()
#dp.image_read()
dp.sequence_50()
dp.image_make()
dp.data_shuffle()



size = int(np.size(dp.point,0) * 0.7)
X_val_ = dp.point
X_train = copy.deepcopy(dp.point)
X_test = dp.point[size:]

X_train_img = dp.images
X_test_img = dp.images[size:]
Y_train = dp.label
Y_test = dp.label[size:]
#Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')
#Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(X_train, Y_train))

for i in range(np.size(X_train,0)):
  X_train[i] = scaler.fit_transform(X_train[i])
'''
  for j in range(np.size(X_train[i],0)):
    X_train[i][j][0] -= np.mean(X_train[i], axis=0)[0]
    X_train[i][j][1] -= np.mean(X_train[i], axis=0)[1]
    X_train[i][j][0] /= np.std(X_train[i],axis=0)[0]
    X_train[i][j][1] /= np.std(X_train[i],axis=0)[1]
''' 
#for i in range(np.size(X_test,0)):
#  X_test[i] = scaler.fit_transform(X_test[i])
'''
  for j in range(np.size(X_test[i],0)):
    X_test[i][j][0] -= np.mean(X_test[i], axis=0)[0]
    X_test[i][j][1] -= np.mean(X_test[i], axis=0)[1]
    X_test[i][j][0] /= np.std(X_test[i],axis=0)[0]
    X_test[i][j][1] /= np.std(X_test[i],axis=0)[1]
'''

'''
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
'''    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
acc_list = []

for l, (train_idx, val_idx) in enumerate(folds):
  print('\n Fold',l)

  X_val_ =dp.point[val_idx]

  X_train_cv = X_train[train_idx]
  X_train_img_cv = X_train_img[train_idx]
  X_valid_cv = X_train[val_idx]
  X_valid_img_cv = X_train_img[val_idx]

  Y_train_cv = Y_train[train_idx]
  Y_valid_cv = Y_train[val_idx]

  Y_train_cv = keras.utils.to_categorical(Y_train_cv,num_classes=26, dtype='float32')
  Y_valid_cv = keras.utils.to_categorical(Y_valid_cv,num_classes=26, dtype='float32')

  input_1 = Input(shape=(28, 28, 1))

  x_1 = Conv2D(32, kernel_size=5, padding="same", activation = 'relu')(input_1)
  x_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_1)
  x_1 = Dropout(0.3)(x_1)

  x_1 = Conv2D(64, kernel_size=3, padding="same", activation = 'relu')(x_1)
  x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)
  x_1 = Dropout(0.3)(x_1)

  x_1 = Conv2D(128, kernel_size=3, padding="same", activation = 'relu')(x_1)
  x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)

  x_1 = Dropout(0.3)(x_1)

  x_1 = Flatten()(x_1)
  x_1 = Dense(units=512, activation='relu')(x_1)


  input_2 = Input(shape=(50, 2))
  '''
  x_2 = Conv1D(256,3,padding='valid',activation='relu',strides=1)(input_2)
  x_2 = Conv1D(256,3,padding='valid',activation='relu',strides=1)(x_2)
  x_2 = MaxPooling1D(3)(x_2)
  x_2 = Conv1D(512,3,padding='valid',activation='relu',strides=1)(x_2)
  x_2 = Conv1D(512,3,padding='valid',activation='relu',strides=1)(x_2)
  x_2 = MaxPooling1D(3)(x_2)
  x_2 = CuDNNLSTM(128)(x_2)
  '''
  
  x_2 = CuDNNLSTM(128,return_sequences=True)(input_2)
  x_2 = Dropout(0.3)(x_2)
  x_2 = CuDNNLSTM(128,return_sequences=True)(x_2)
  x_2 = Dropout(0.3)(x_2)
  x_2 = CuDNNLSTM(128)(x_2)
  

  merged = concatenate([x_1,x_2])
  m = Dense(256, activation='relu')(merged)

  m = Dense(26, activation='softmax')(m)

  model = Model(inputs=[input_1, input_2], outputs = m)

  #model.summary()
  model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

  #model.fit_generator(train_generator(),steps_per_epoch=2700, epochs=20,validation_data=test_generator(),validation_steps=300)
  model.fit([X_train_img_cv,X_train_cv],Y_train_cv,epochs=40,batch_size=32,validation_data=([X_valid_img_cv,X_valid_cv],Y_valid_cv))
  score = model.evaluate([X_valid_img_cv,X_valid_cv],Y_valid_cv)
  scores = model.predict([X_valid_img_cv,X_valid_cv])
  true_value = np.argmax(Y_valid_cv,1)
  predict_value = np.argmax(scores,1)
  
  list_ = []
  for i in range(np.size(true_value,0)):
    if(true_value[i] != predict_value[i]):
      print('t:{},p:{}'.format(true_value[i],predict_value[i]))
      list_.append(i)

  print(score[1]*100)
  upper = np.array([0,0,255])
  lower = np.array([0,0,5])
  acc_list.append(score[1]*100)
  '''
  ROW = 5
  COLUMN = 4
  j = 1
  for i in list_:
    img = np.ones((550, 550, 3), np.uint8) * 255
    for k in range(len(X_val_[i])):
      if(k != len(X_val_[i])-1):
        cv2.line(img, (X_val_[i][k][0],X_val_[i][k][1]), (X_val_[i][k+1][0],X_val_[i][k+1][1]), (0,0,5*k+5), 20)
    img = cv2.flip(img, 1)
    mask = cv2.inRange(img, lower, upper)
    #img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    cv2.imwrite('./plot/lstm/{}lstm_seq{}{}.jpg'.format(i,true_value[i],predict_value[i]),img)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.title.set_text('Unchanged')
    ax.plot(X_valid_cv[i])
    ax.set_xlim([0,50])
    ax.set_ylim([-50,150])
    ax.title.set_fontsize(20)
    plt.savefig('./plot/lstm/lstm+cnn{}{}.png'.format(true_value[i],predict_value[i]))


  for i in list_:
      if(j> 20):
        j = 20
      # train[i][0] is i-th image data with size 28x28
      image = X_valid_img_cv[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
      plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
      j +=1
      plt.imshow(image, cmap=plt.cm.Blues)  # cmap='gray' is for black and white picture.
      # train[i][1] is i-th digit label
      plt.title('predict = {}'.format(np.argmax(scores[i],0)))
      plt.axis('off')  # do not show axis value

  plt.tight_layout()   # automatic padding between subplots
  plt.savefig('./plot/lstm/all/LSTM{}.png'.format(l))
  plt.clf()
  '''

print(acc_list)
'''  
score = model.evaluate([X_test_img,X_test],Y_test)
scores = model.predict([X_test_img,X_test])
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(scores,1)
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    list_.append(i)

print(score[1]*100)
upper = np.array([0,0,255])
lower = np.array([0,0,5])

for i in list_:
  img = np.ones((550, 550, 3), np.uint8) * 255
  for k in range(len(X_test[i])):
    if(k != len(X_test[i])-1):
      cv2.line(img, (X_test[i][k][0],X_test[i][k][1]), (X_test[i][k+1][0],X_test[i][k+1][1]), (0,0,5*k+5), 20)
  img = cv2.flip(img, 1)
  mask = cv2.inRange(img, lower, upper)
  #img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
  cv2.imwrite('./plot/lstm/{}lstm_seq.jpg'.format(i),img)
ROW = 5
COLUMN = 6
j = 1
for i in list_:
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {}'.format(np.argmax(scores[i],0)))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('lstm+cnn.png')
cm = confusion_matrix(true_value, predict_value)
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.show()

'''