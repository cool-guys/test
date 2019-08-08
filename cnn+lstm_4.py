
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
MODEL_SAVE_FOLDER_PATH = '../model/lstm'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc',
                                verbose=1, save_best_only=True)


#early_stopping = EarlyStopping()

# Minmax 
scaler = MinMaxScaler()
scaler_ = MinMaxScaler()


def conv1_layer(x_2):
  x_2 = ZeroPadding1D(padding=3)(x_2)
  x_2 = Conv1D(64, kernel_size=7, strides=2)(x_2)
  x_2 = BatchNormalization()(x_2)
  x_2 = ReLU()(x_2)
  x_2 = ZeroPadding1D(padding=1)(x_2)

  return x_2

def conv2_layer(x_2):
  x_2 = MaxPool1D(1,2)(x_2)

  shortcut = x_2

  for i in range(3):
    if(i == 0):
      x_2 = Conv1D(64, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(64, kernel_size=16, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(256, kernel_size=1, strides=1, padding='valid')(x_2)
      shortcut = Conv1D(256, kernel_size=1, strides=1, padding='valid')(shortcut)
      x_2 = BatchNormalization()(x_2)
      shortcut = BatchNormalization()(shortcut)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2
      
    else:
      x_2 = Conv1D(64, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(64, kernel_size=16, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(256, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2
  
  return x_2

def conv3_layer(x_2):
  shortcut = x_2

  for i in range(4):
    if(i == 0):
      x_2 = Conv1D(128, kernel_size=1, strides=2, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(128, kernel_size=15, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(512, kernel_size=1, strides=1, padding='valid')(x_2)
      shortcut = Conv1D(512, kernel_size=1, strides=2, padding='valid')(shortcut)
      x_2 = BatchNormalization()(x_2)
      shortcut = BatchNormalization()(shortcut)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2

    else:
      x_2 = Conv1D(128, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(128, kernel_size=15, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(512, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2
  
  return x_2
def conv4_layer(x_2):
  shortcut = x_2

  for i in range(6):
    if(i == 0):
      x_2 = Conv1D(256, kernel_size=1, strides=2, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(256, kernel_size=15, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(1024, kernel_size=1, strides=1, padding='valid')(x_2)
      shortcut = Conv1D(1024, kernel_size=1, strides=2, padding='valid')(shortcut)
      x_2 = BatchNormalization()(x_2)
      shortcut = BatchNormalization()(shortcut)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2

    else:
      x_2 = Conv1D(256, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(256, kernel_size=15, strides=1, padding='same')(x_2)
      x_2 = BatchNormalization()(x_2)
      x_2 = ReLU()(x_2)

      x_2 = Conv1D(1024, kernel_size=1, strides=1, padding='valid')(x_2)
      x_2 = BatchNormalization()(x_2)

      x_2 = Add()([x_2, shortcut])
      x_2 = ReLU()(x_2)

      shortcut = x_2

  return x_2


def conv1_layer_2(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 
    
 
def conv2_layer_2(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer_2(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
#숫자인지 아닌지
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

# cnn part
input_1 = Input(shape=(28, 28, 1))

#x_1 = conv1_layer_2(input_1)
#x_1 = conv2_layer_2(x_1)

x_1 = Conv2D(32, kernel_size=3, padding="valid", activation = 'relu')(input_1)
x_1 = BatchNormalization()(x_1)
x_1 = ReLU()(x_1)
x_1 = Dropout(0.45)(x_1)

x_1 = Conv2D(64, kernel_size=3, padding="valid", activation = 'relu')(x_1)
x_1 = BatchNormalization()(x_1)
x_1 = ReLU()(x_1)
x_1 = Dropout(0.45)(x_1)

x_1 = Conv2D(128, kernel_size=3, padding="valid", activation = 'relu')(x_1)
x_1 = BatchNormalization()(x_1)
x_1 = ReLU()(x_1)
x_1 = Dropout(0.45)(x_1)

x_1 = Conv2D(256, kernel_size=3, padding="valid", activation = 'relu')(x_1)
x_1 = BatchNormalization()(x_1)
x_1 = ReLU()(x_1)
x_1 = Dropout(0.45)(x_1)

x_1 = GlobalAveragePooling2D()(x_1)
x_1 = Dropout(0.45)(x_1)


# lstm part
input_2 = Input(shape=(64, 2))
'''
x_2 = conv1_layer(input_2)
x_2 = conv2_layer(x_2)


x_2 = GlobalAveragePooling1D()(x_2)
'''


x_2 = Conv1D(32,kernel_size=4, padding='same',strides=1)(input_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(64,kernel_size=5, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(alpha=0.2)(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(128, kernel_size=6, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = Dropout(0.5)(x_2)


x_2 = Conv1D(256, kernel_size=7, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = Conv1D(512, kernel_size=8, strides=1, padding='same')(x_2)
x_2 = BatchNormalization(momentum=0.8)(x_2)
x_2 = LeakyReLU(0.2)(x_2)
x_2 = Dropout(0.5)(x_2)

x_2 = GlobalAveragePooling1D()(x_2)




'''
x_2 = Bidirectional(CuDNNLSTM(32,return_sequences=True))(input_2)
x_2 = BatchNormalization()(x_2)
x_2 = Dropout(0.45)(x_2)
x_2 = Bidirectional(CuDNNLSTM(64,return_sequences=True))(x_2)
x_2 = BatchNormalization()(x_2)
x_2 = Dropout(0.45)(x_2)
x_2 = CuDNNLSTM(128)(x_2)
'''
# merge part
merged = concatenate([x_1,x_2])

m = Dense(class_num, activation='softmax')(merged)

model = Model(inputs=[input_1, input_2], outputs = m)

model.summary()
model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.001,epsilon=1e-04),
                metrics=['accuracy'])

results = model.fit([X_train_img,X_train],Y_train,epochs=250,batch_size=128,validation_data=([X_test_img,X_test],Y_test),callbacks=[cb_checkpoint,lr_reduce])

model = load_model('../model/lstm.hdf5')

score = model.evaluate([X_test_img,X_test],Y_test)
pred = model.predict([X_test_img,X_test])
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(pred,1)
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    list_.append(i)

print(score[1]*100)
'''
ROW = 5
COLUMN = 4
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
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
plt.savefig('cnn+lstm.png')
'''

ROW = 5
COLUMN = 4
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    if(number):
      plt.title('predict = {},{}'.format((np.argmax(pred[i],0)),chr(97+true_value[i])))
    else:
      plt.title('predict = {},{}'.format(chr(97+ np.argmax(pred[i],0)),chr(97+true_value[i])))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('cnn+lstm{}.png'.format(score[1]*100))
plt.show()

plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn+lstm_acc.png')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn+lstm_loss.png')
plt.show()



cm = confusion_matrix(true_value, predict_value)
df_cm = pd.DataFrame(cm, index = [i for i in "abcdefghijklmnopqrstuvwxyz"],
                  columns = [i for i in "abcdefghijklmnopqrstuvwxyz"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_map.png')
plt.show()

'''
upper = np.array([0,0,255])
lower = np.array([0,0,5])
if(number):
  for i in list_:
    img = np.ones((550, 550, 3), np.uint8) * 255
    for k in range(len(X_test[i])):
      if(k != len(X_test[i])-1):
        cv2.line(img, (X_test_[i][k][0],X_test_[i][k][1]), (X_test_[i][k+1][0],X_test_X_test[i][k+1][1]), (0,0,5*k+5), 20)
    img = cv2.flip(img, 1)
    mask = cv2.inRange(img, lower, upper)
    #img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    cv2.imwrite('../plot/lstm/{}lstm_seq{}{}.jpg'.format(i,true_value[i],predict_value[i]),img)
else:
  for i in list_:
    img = np.ones((550, 550, 3), np.uint8) * 255
    for k in range(len(X_test[i])):
      if(k != len(X_test[i])-1):
        cv2.line(img, (X_test_[i][k][0],X_test_[i][k][1]), (X_test_[i][k+1][0],X_test_[i][k+1][1]), (0,0,5*k+5), 20)
    img = cv2.flip(img, 1)
    mask = cv2.inRange(img, lower, upper)
    #img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    cv2.imwrite('../plot/lstm/{}lstm_seq{}{}.jpg'.format(i,chr(97 + true_value[i]),chr(97 + predict_value[i])),img)  

ROW = 5
COLUMN = 6
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {}'.format(chr(97+ np.argmax(predict_value[i],0))))
    print(i)
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('lstm+cnn.png')
cm = confusion_matrix(true_value, predict_value)
df_cm = pd.DataFrame(cm, index = [i for i in "abcdefghijklmnopqrstuvwxyz"],
                  columns = [i for i in "abcdefghijklmnopqrstuvwxyz"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


for i in list_:
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(1,1,1)
  ax.title.set_text('Unchanged')
  ax.plot(X_test[i])
  ax.set_xlim([0,50])
  ax.set_ylim([-50,150])
  ax.title.set_fontsize(20)
  plt.savefig('../plot/lstm/train-test/{}lstm+cnn{}{}.png'.format(i,true_value[i],predict_value[i]))
'''